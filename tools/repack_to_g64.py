"""Re-pack DiT linear layers from groupsize=32 to groupsize=64.

For every selected layer that satisfies ``K % 64 == 0`` the tool:

1. dequantises the existing AutoGPTQ-v1 packed buffers back to fp16
   (via ``quant_utils.dequant_gptq_to_fp``);
2. re-quantises with simple RTN at groupsize=64 (per-group min/max);
3. re-packs into AutoGPTQ-v1 format (the same layout `FusedInt4Linear`
   reads after the kernel was opened up to g32/g64 in this PR);
4. writes a new safetensors file with the per-layer ``groupsize`` in
   ``quant_layers_json`` updated to 64.

Layout invariants (mirrors `quant_utils.unpack_int_weights` / `unpack_zeros`):

  qweight (K // 8, N)        int32 — 8 nibbles per int32, K-packed
  scales  (K // gs, N)       fp16  — per-group scale
  qzeros  (K // gs, N // 8)  int32 — 8 nibbles per int32, N-packed, v1 -1 offset
  g_idx   (K,)               int32 — implicit (k // gs) when actorder=False

Re-quantisation is RTN, not GPTQ. We do not have the GPTQ calibration
Hessian here, so we cannot replay the Hessian-aware solve at g=64. For
the layers the heuristic targets (SwiGLU gate is the documented safe
case; ``--scope all-divisible`` widens that with author discretion) the
RTN/GPTQ delta is typically small relative to fp16 quality. Verify with
SNR / UTMOS before shipping a re-packed checkpoint.

CLI::

    python tools/repack_to_g64.py \\
        --input dit_int4.safetensors \\
        --output dit_int4_g64.safetensors \\
        --scope swiglu-gate          # or all-divisible / pattern <regex>

The shipped Irodori-TTS-Lite-int4 checkpoint has shapes
  attention.{q,k,v,o}: K=N=1280   (K % 64 == 0  ✓)
  mlp.w1 (gate), mlp.w2 (up): K=1280, N=3680  (K % 64 == 0  ✓)
  mlp.w3 (down):              K=3680, N=1280  (K % 64 == 0  ✗ — kept g=32)
so ``swiglu-gate`` re-packs 12 layers, ``all-divisible`` re-packs ~72.

Untested on real hardware. Run `tools/measure_peak_memory.py` after
re-packing to confirm VRAM / quality match expectations.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Allow running directly from the repo without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irodori_tts_lite.quant_utils import dequant_gptq_to_fp  # noqa: E402


def _pack_int4_v1(w_int: torch.Tensor) -> torch.Tensor:
    """(out, in) int in [0, 15] → packed (in // 8, out) int32.

    Mirrors the inverse of `quant_utils._unpack_rows_int4`: row k of the
    transposed (in, out) tensor lands in nibble (k % 8) of packed row k // 8.
    """
    out_f, in_f = w_int.shape
    if in_f % 8 != 0:
        raise ValueError(f"in_features {in_f} must be a multiple of 8")
    w_t = w_int.t().contiguous().to(torch.int32)  # (in, out)
    packed = torch.zeros((in_f // 8, out_f), dtype=torch.int32)
    for i in range(8):
        packed |= (w_t[i::8, :] & 0xF) << (i * 4)
    return packed


def _pack_zeros_v1(zeros: torch.Tensor) -> torch.Tensor:
    """(num_groups, out) int in [0, 15] → packed (num_groups, out // 8) int32 with v1 -1 offset.

    The on-disk encoding stores `(z - 1) & 0xF` so dequant's `(z + 1) & 0xF`
    recovers the original. Packing direction matches `unpack_zeros`: 8 nibbles
    per int32 along the output-feature axis.
    """
    num_g, out_f = zeros.shape
    if out_f % 8 != 0:
        raise ValueError(f"out_features {out_f} must be a multiple of 8")
    z_v1 = ((zeros.to(torch.int32) - 1) & 0xF)  # AutoGPTQ-v1 on-disk form
    z_t = z_v1.t().contiguous()  # (out, num_groups)
    packed = torch.zeros((out_f // 8, num_g), dtype=torch.int32)
    for i in range(8):
        packed |= (z_t[i::8, :] & 0xF) << (i * 4)
    return packed.t().contiguous()  # (num_groups, out // 8)


def _rtn_quantize_g(w: torch.Tensor, groupsize: int):
    """(out, in) fp → ((out, in) int [0,15], (num_g, out) fp16, (num_g, out) int).

    Per-group symmetric-around-zero RTN: ``scale = (max - min) / 15``,
    ``zero = round(-min / scale)`` so ``q = clamp(round(w / scale) + zero, 0, 15)``.
    """
    out_f, in_f = w.shape
    if in_f % groupsize != 0:
        raise ValueError(f"in_features={in_f} not divisible by groupsize={groupsize}")
    num_g = in_f // groupsize
    w_g = w.float().reshape(out_f, num_g, groupsize)
    w_max = w_g.amax(dim=-1, keepdim=True)
    w_min = w_g.amin(dim=-1, keepdim=True)
    dead = (w_max == w_min)
    w_max = torch.where(dead, w_max + 1.0, w_max)
    w_min = torch.where(dead, w_min - 1.0, w_min)
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    zero = torch.round(-w_min / scale).clamp(0, 15)
    q = torch.clamp(torch.round(w_g / scale) + zero, 0, 15).to(torch.int32)
    w_int = q.reshape(out_f, in_f)
    scales = scale.squeeze(-1).t().contiguous().to(torch.float16)  # (num_g, out)
    zeros = zero.squeeze(-1).t().contiguous().to(torch.int32)  # (num_g, out)
    return w_int, scales, zeros


def _repack_layer(state: dict, name: str, in_f: int, out_f: int, *,
                  v1: bool, target_groupsize: int) -> tuple[dict, dict]:
    """Re-quantise a single GPTQ-v1 layer at the new groupsize.

    Returns ``(new_tensors, popped_keys)`` where popped_keys is the list of
    old tensor keys removed from ``state`` (caller writes the new ones).
    """
    qweight_old = state[f"{name}.qweight"]
    scales_old = state[f"{name}.scales"]
    qzeros_old = state[f"{name}.qzeros"]
    g_idx_old = state.get(f"{name}.g_idx")
    if g_idx_old is None:
        # actorder=False checkpoints often omit g_idx; reconstruct.
        g_idx_old = (torch.arange(in_f) // (in_f // scales_old.shape[0])).to(torch.int32)

    w_fp = dequant_gptq_to_fp(
        qweight=qweight_old, scales=scales_old, qzeros=qzeros_old, g_idx=g_idx_old,
        in_features=in_f, out_features=out_f, dtype=torch.float16, v1=v1,
    )
    w_int, scales_new, zeros_new = _rtn_quantize_g(w_fp, groupsize=target_groupsize)
    qweight_new = _pack_int4_v1(w_int)
    qzeros_new = _pack_zeros_v1(zeros_new)
    g_idx_new = (torch.arange(in_f) // target_groupsize).to(torch.int32)

    return (
        {"qweight": qweight_new, "scales": scales_new,
         "qzeros": qzeros_new, "g_idx": g_idx_new},
        ["qweight", "scales", "qzeros", "g_idx"],
    )


def _select(entry: dict, scope: str, pattern: re.Pattern | None) -> bool:
    name = entry["name"]
    in_f = int(entry["in_features"])
    if in_f % 64 != 0:
        return False
    if int(entry["wbits"]) != 4:
        return False
    if int(entry["groupsize"]) == 64:
        return False  # already done
    if bool(entry.get("actorder", False)):
        # Re-packing an actorder=True layer would require regenerating
        # the permutation; we only support actorder=False.
        return False
    if scope == "swiglu-gate":
        return name.endswith(".mlp.w1") or ".mlp.w1." in name
    if scope == "all-divisible":
        return True
    if scope == "pattern":
        return bool(pattern and pattern.search(name))
    raise ValueError(f"unknown scope {scope!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to existing int4 safetensors")
    p.add_argument("--output", required=True, help="path to write the re-packed safetensors")
    p.add_argument(
        "--scope",
        choices=["swiglu-gate", "all-divisible", "pattern"],
        default="swiglu-gate",
        help="which layers to re-pack at groupsize=64 (default: SwiGLU gate only)",
    )
    p.add_argument(
        "--pattern",
        default=None,
        help="regex (used when --scope pattern) to match layer names",
    )
    args = p.parse_args()

    pattern = re.compile(args.pattern) if args.pattern else None
    if args.scope == "pattern" and pattern is None:
        raise SystemExit("--scope pattern requires --pattern <regex>")

    with safe_open(args.input, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        state = {k: f.get_tensor(k) for k in f.keys()}

    if md.get("quant_method") != "autobit" or "quant_layers_json" not in md:
        raise SystemExit("input is not a packed AutoBit checkpoint")

    quant_layers = json.loads(md["quant_layers_json"])
    v1 = md.get("checkpoint_format", "gptq") != "gptq_v2"

    new_layers: list[dict] = []
    repacked = 0
    bytes_before = 0
    bytes_after = 0
    for entry in quant_layers:
        if _select(entry, args.scope, pattern):
            in_f = int(entry["in_features"])
            out_f = int(entry["out_features"])
            name = entry["name"]

            scales_old = state[f"{name}.scales"]
            qzeros_old = state[f"{name}.qzeros"]
            bytes_before += scales_old.numel() * scales_old.element_size()
            bytes_before += qzeros_old.numel() * qzeros_old.element_size()

            new_tensors, keys_to_pop = _repack_layer(
                state, name, in_f, out_f, v1=v1, target_groupsize=64,
            )
            for k in keys_to_pop:
                state.pop(f"{name}.{k}", None)
            for k, t in new_tensors.items():
                state[f"{name}.{k}"] = t

            bytes_after += new_tensors["scales"].numel() * new_tensors["scales"].element_size()
            bytes_after += new_tensors["qzeros"].numel() * new_tensors["qzeros"].element_size()

            new_entry = dict(entry)
            new_entry["groupsize"] = 64
            new_layers.append(new_entry)
            print(f"  [g32 -> g64] {name}  K={in_f} N={out_f}")
            repacked += 1
        else:
            new_layers.append(entry)

    if repacked == 0:
        raise SystemExit(
            f"no layers matched scope={args.scope!r} (check shapes / pattern)"
        )

    new_md = {k: v for k, v in md.items() if k != "quant_layers_json"}
    new_md["quant_layers_json"] = json.dumps(new_layers)

    save_file(state, args.output, metadata=new_md)
    print(
        f"\nrepacked {repacked} layers; "
        f"scales+qzeros bytes "
        f"{bytes_before/1024:.0f} KB -> {bytes_after/1024:.0f} KB "
        f"({(bytes_before - bytes_after)/1024:.0f} KB saved)"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
