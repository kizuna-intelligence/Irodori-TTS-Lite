"""Minimal end-to-end example: int4 TTS inference via Irodori-TTS-Lite.

Loads the int4-quantized DiT shipped under `weights/` and runs inference
through the upstream TTS pipeline's `infer.main()`.

Run from the repo root::

    python example/run_tts.py \\
        --checkpoint weights/dit_int4.safetensors \\
        --text "こんにちは、メラだよ！" \\
        --out /tmp/mera.wav \\
        --no-ref
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="Local path or hf://<org>/<repo>/<file>. "
                             "Omit to auto-download from "
                             "kizuna-intelligence/Irodori-TTS-Lite-int4.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-wav", default="/tmp/irodori_tts_lite_sample.wav")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Audio duration. Auto-derived from phoneme count when omitted.")
    parser.add_argument("--no-ref", action="store_true",
                        help="Voice-design checkpoint mode (speaker baked in).")
    parser.add_argument("--no-fused", action="store_true",
                        help="Skip FusedInt4Linear; eager-dequant every layer.")
    parser.add_argument("--no-eager-dequant", action="store_true",
                        help="Disable AdaLN eager-dequant.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable forced fp16 model dtype. Default is fp16, "
                             "which matches the kernel's native dtype and avoids "
                             "an fp32 wrapper-path shape check.")
    parser.set_defaults(fp16=True)
    args, infer_argv = parser.parse_known_args()

    if args.seconds is None:
        import pyopenjtalk
        phs = pyopenjtalk.g2p(args.text, kana=False).split()
        args.seconds = max(2.0, len(phs) / 11.0 + 0.6)
        print(f"[run_tts] phonemes={len(phs)} → seconds={args.seconds:.2f}")

    import irodori_tts_lite
    irodori_tts_lite.configure(
        use_fused=not args.no_fused,
        force_fp16=args.fp16,
        disable_eager=args.no_eager_dequant,
    )
    irodori_tts_lite.patch()

    checkpoint_path = irodori_tts_lite.resolve_checkpoint(args.checkpoint)
    if checkpoint_path != (args.checkpoint or ""):
        print(f"[run_tts] checkpoint: {checkpoint_path}")

    # Defer infer import until after the runtime is patched so it picks up
    # our hooks the first time it constructs an InferenceRuntime.
    import infer
    infer.FIXED_SECONDS = float(args.seconds)

    sys.argv = [sys.argv[0], "--checkpoint", checkpoint_path, "--text", args.text,
                "--output-wav", args.output_wav]
    if args.no_ref:
        sys.argv.append("--no-ref")
    sys.argv.extend(infer_argv)

    infer.main()
    print(f"[run_tts] wrote {args.output_wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
