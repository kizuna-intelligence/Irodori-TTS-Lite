"""Irodori-TTS-Lite — エンドツーエンドのピーク GPU メモリを計測する。

上流 TTS パイプラインの `infer.main()` 経路をそのまま走らせ、
`torch.cuda.max_memory_allocated()` / `max_memory_reserved()` を取る。

DiT 単体ではなく、DACVAE コーデック / トークナイザ / forward 中の
activation を **すべて含めた** 値を返す。

例::

    python tools/measure_peak_memory.py \\
        --mode int4 \\
        --text "こんにちは" \\
        --num-steps 6 \\
        --no-ref

int4 モードで `--checkpoint` を省略すると HF から自動 DL。`--mode bf16` /
`--mode fp32` を指定すると未量子化との比較が可能（この場合は `--checkpoint`
に元の checkpoint を渡す）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def _peek(label: str) -> dict[str, float]:
    alloc = torch.cuda.memory_allocated() / 1024**2
    res = torch.cuda.memory_reserved() / 1024**2
    print(f"[mem] {label:32s} allocated={alloc:8.1f} MB  reserved={res:8.1f} MB")
    return {"label": label, "allocated_MB": alloc, "reserved_MB": res}


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Allow running against an on-disk upstream TTS checkout that isn't
    # pip-installed (set TTS_UPSTREAM_PATH).
    _upstream = os.environ.get("TTS_UPSTREAM_PATH")
    if _upstream and os.path.isdir(_upstream) and _upstream not in sys.path:
        sys.path.insert(0, _upstream)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["int4", "bf16", "fp32"], required=True,
                        help="int4: 本リポジトリの量子化推論. bf16/fp32: 未量子化との比較用.")
    parser.add_argument("--checkpoint", default=None,
                        help="Local path or hf://<org>/<repo>/<file>. "
                             "int4 モードで省略時は kizuna-intelligence/Irodori-TTS-Lite-int4 "
                             "から自動ダウンロード.")
    parser.add_argument("--text", default="こんにちは、メラだよ。")
    parser.add_argument("--output-wav", default="/tmp/peak_mem_probe.wav")
    parser.add_argument("--num-steps", type=int, default=6)
    parser.add_argument("--no-ref", action="store_true")
    parser.add_argument("--codec-int4", action="store_true",
                        help="DACVAE コーデックも int4 (NormConv1d/ConvTranspose1d を packed 化)")
    parser.add_argument("--json", action="store_true", help="末尾に JSON 1 行を出力する")
    args, infer_argv = parser.parse_known_args()

    if not torch.cuda.is_available():
        sys.stderr.write("CUDA GPU が見つかりません\n")
        return 2

    # ---- patching ----
    if args.mode == "int4":
        import irodori_tts_lite
        irodori_tts_lite.configure(
            use_fused=True, force_fp16=True, codec_int4=args.codec_int4,
        )
        irodori_tts_lite.patch()
        checkpoint_path = irodori_tts_lite.resolve_checkpoint(args.checkpoint)
        precision = "fp32"   # patched loader overrides to fp16 regardless
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for bf16/fp32 modes")
        checkpoint_path = args.checkpoint
        precision = args.mode

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = _peek("baseline (CUDA context のみ)")

    # ---- argv assembly ----
    sys.argv = [
        sys.argv[0],
        "--checkpoint", checkpoint_path,
        "--text", args.text,
        "--output-wav", args.output_wav,
        "--num-steps", str(args.num_steps),
        "--model-precision", precision,
    ]
    if args.no_ref:
        sys.argv.append("--no-ref")
    sys.argv.extend(infer_argv)

    import infer
    infer.FIXED_SECONDS = 3.0  # determinism for the probe

    # 同期して測定の境界を明示
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    infer.main()

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_res = torch.cuda.max_memory_reserved() / 1024**2
    print()
    print(f"[mem] PEAK over full inference  allocated={peak_alloc:8.1f} MB  "
          f"reserved={peak_res:8.1f} MB")

    if args.json:
        print("RESULT " + json.dumps({
            "mode": args.mode,
            "baseline_allocated_MB": baseline["allocated_MB"],
            "baseline_reserved_MB": baseline["reserved_MB"],
            "peak_allocated_MB": peak_alloc,
            "peak_reserved_MB": peak_res,
            "num_steps": args.num_steps,
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
