"""Re-quantise the DiT block FFN linears from int4 to int2.

Picks the SwiGLU MLP linears (``*.mlp.w1``, ``*.mlp.w2``, ``*.mlp.w3``) out
of an existing int4 safetensors, dequants them via the GPTQ-v1 path, then
re-quantises with simple RTN at wbits=2 and writes a new safetensors that
keeps every other layer untouched. The new tensors live under the same
``_extra.<name>.*`` namespace the runtime already understands, but with a
new metadata key ``extra_quant_2bit_json`` describing the wbits=2 entries.

Why FFN-only
------------
The DiT FFN dominates parameter count (~70% of the block body) and is
empirically more tolerant of aggressive quantisation than attention output
projection or AdaLN, which the 4-bit kernel keeps untouched. See the
``docs/architecture.md`` rationale and external references for DiT
quantisation sensitivity.

Why RTN, not GPTQ/HQQ/AWQ
-------------------------
This is a scaffold. RTN min/max quant is the simplest 2-bit baseline; at
wbits=2 the SNR loss is non-trivial on random weights (~3-4 dB on N(0,1))
and audio TTS is more sensitive than LLM perplexity. A real deployment
should swap in calibration-aware 2-bit (HQQ / AWQ / AQLM) and verify with
UTMOS / MOS on JVS or JSUT samples — not yet done in this repo.

The shipped runtime supports the produced format end-to-end after
``configure(ffn_2bit=True)`` (see the companion loader changes).

CLI::

    python tools/requantize_dit_ffn_2bit.py \\
        --input dit_int4.safetensors \\
        --output dit_int4_ffn2bit.safetensors \\
        --groupsize 32 \\
        --layers w1,w2,w3        # default w1,w2,w3 (whole FFN). Use w1,w2 to spare w3.

Expected disk delta on the shipped checkpoint (12 blocks):
    w1, w2 (K=1280, N=3680): scales+zeros (40, 3680) fp16 each, qweight_u8 (3680, 320)
                              → before (per layer, gptq): ~2.4 MB
                                after  (per layer, int2): ~330 KB qweight + 290 KB scales/zeros
                              → ~1.5 MB saved per layer
    w3      (K=3680, N=1280): scales+zeros (115, 1280) fp16, qweight_u8 (1280, 920)
                              → ~2.4 MB → ~1.5 MB
Total for 12 blocks, w1+w2+w3: ~54 MB → ~22 MB, saving ~32 MB on disk.
The VRAM peak win is comparable since the runtime keeps weights packed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irodori_tts_lite.quant_utils import dequant_gptq_to_fp  # noqa: E402
from irodori_tts_lite.streaming_int2_linear import (  # noqa: E402
    pack_int2_u8,
    rtn_quantize_2bit,
)


def _is_ffn(name: str, allow: tuple[str, ...]) -> bool:
    """Match `blocks.{i}.mlp.{w1|w2|w3}` for the requested w-set."""
    for w in allow:
        suf = f".mlp.{w}"
        if name.endswith(suf) or f"{suf}." in name:
            return True
    return False


def _strip_ffn_from_state(state: dict, name: str) -> None:
    for s in ("qweight", "scales", "qzeros", "g_idx", "bias"):
        state.pop(f"{name}.{s}", None)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to existing int4 safetensors")
    p.add_argument("--output", required=True, help="path to write the int2-FFN safetensors")
    p.add_argument(
        "--groupsize",
        type=int,
        default=32,
        help="RTN group size for the new 2-bit layers (default 32; -1 = per-row)",
    )
    p.add_argument(
        "--layers",
        default="w1,w2,w3",
        help=(
            "Comma-separated SwiGLU MLP layer suffixes to convert to 2-bit "
            "(default `w1,w2,w3` = whole FFN; use `w1,w2` to spare the down "
            "projection, etc.)"
        ),
    )
    args = p.parse_args()

    allow = tuple(s.strip() for s in args.layers.split(",") if s.strip())
    for w in allow:
        if w not in ("w1", "w2", "w3"):
            raise SystemExit(
                f"--layers entries must be w1/w2/w3 (got {w!r})"
            )

    with safe_open(args.input, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        state = {k: f.get_tensor(k) for k in f.keys()}

    if md.get("quant_method") != "autobit" or "quant_layers_json" not in md:
        raise SystemExit("input is not a packed AutoBit checkpoint")

    quant_layers = json.loads(md["quant_layers_json"])
    v1 = md.get("checkpoint_format", "gptq") != "gptq_v2"

    new_int4_layers: list[dict] = []
    new_2bit_entries: list[dict] = []
    bytes_before = 0
    bytes_after = 0

    for entry in quant_layers:
        name = entry["name"]
        in_f = int(entry["in_features"])
        out_f = int(entry["out_features"])
        if not _is_ffn(name, allow):
            new_int4_layers.append(entry)
            continue
        if int(entry["wbits"]) != 4:
            new_int4_layers.append(entry)
            continue
        if args.groupsize > 0 and in_f % args.groupsize != 0:
            print(
                f"  [skip g-mismatch] {name} (in_features={in_f}, "
                f"--groupsize={args.groupsize})"
            )
            new_int4_layers.append(entry)
            continue

        qweight_old = state[f"{name}.qweight"]
        scales_old = state[f"{name}.scales"]
        qzeros_old = state[f"{name}.qzeros"]
        g_idx_old = state.get(f"{name}.g_idx")
        if g_idx_old is None:
            g_idx_old = (torch.arange(in_f) // (in_f // scales_old.shape[0])).to(torch.int32)

        bytes_before += sum(
            t.numel() * t.element_size()
            for t in (qweight_old, scales_old, qzeros_old)
        )

        w_fp = dequant_gptq_to_fp(
            qweight=qweight_old, scales=scales_old, qzeros=qzeros_old,
            g_idx=g_idx_old, in_features=in_f, out_features=out_f,
            dtype=torch.float16, v1=v1,
        )
        w_int, scales_new, zeros_new = rtn_quantize_2bit(w_fp, groupsize=args.groupsize)
        qweight_u8 = pack_int2_u8(w_int)

        _strip_ffn_from_state(state, name)
        state[f"_extra2.{name}.qweight_u8"] = qweight_u8
        state[f"_extra2.{name}.scales"] = scales_new
        state[f"_extra2.{name}.zeros"] = zeros_new
        bias = state.get(f"{name}.bias")
        if bias is not None:
            state[f"{name}.bias"] = bias  # keep where it was

        new_2bit_entries.append({
            "name": name,
            "in_features": in_f,
            "out_features": out_f,
            "wbits": 2,
            "groupsize": args.groupsize,
            "num_groups": (
                in_f // args.groupsize if args.groupsize > 0 else 1
            ),
            "has_bias": bias is not None,
        })

        bytes_after += qweight_u8.numel() * qweight_u8.element_size()
        bytes_after += scales_new.numel() * scales_new.element_size()
        bytes_after += zeros_new.numel() * zeros_new.element_size()

        print(f"  [int4 -> int2] {name}  K={in_f} N={out_f}")

    if not new_2bit_entries:
        raise SystemExit(
            f"no FFN layers matched --layers={args.layers} / --groupsize={args.groupsize}"
        )

    new_md = {k: v for k, v in md.items() if k != "quant_layers_json"}
    new_md["quant_layers_json"] = json.dumps(new_int4_layers)
    new_md["extra_quant_2bit_json"] = json.dumps(new_2bit_entries)

    save_file(state, args.output, metadata=new_md)
    print(
        f"\nconverted {len(new_2bit_entries)} layers to 2-bit; "
        f"bytes (qweight+scales+qzeros) "
        f"{bytes_before/1024**2:.1f} MB -> {bytes_after/1024**2:.1f} MB "
        f"({(bytes_before - bytes_after)/1024**2:.1f} MB saved)"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
