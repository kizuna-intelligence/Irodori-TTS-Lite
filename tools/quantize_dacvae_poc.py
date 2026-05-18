"""DACVAE 4-bit 量子化 PoC.

DACVAECodec をロード → NormConv1d / NormConvTranspose1d を
`PackedInt4Conv1d` / `PackedInt4ConvTranspose1d` に in-place 置換 →
fp16 ベースラインと音声出力を比較 + 量子化版のピーク VRAM を計測.

  python tools/quantize_dacvae_poc.py [--groupsize 32]

依存:
  * 上流 TTS パイプラインのパスを `TTS_UPSTREAM_PATH` に設定すること
  * Irodori-TTS-Lite が editable install されていること
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

for _p in (os.environ.get("TTS_UPSTREAM_PATH"),
           os.path.join(os.path.dirname(__file__), "..")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))

from irodori_tts_lite.packed_conv import replace_conv_with_packed  # noqa: E402


def _decode(codec, latent: torch.Tensor) -> torch.Tensor:
    # DACVAECodec.decode_latent expects (B, T, D)
    return codec.decode_latent(latent)


def _peak_mb(reset: bool = False) -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return (
        torch.cuda.max_memory_allocated() / 1024**2,
        torch.cuda.max_memory_reserved() / 1024**2,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    p.add_argument("--groupsize", type=int, default=32)
    p.add_argument("--baseline-dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--latent-len", type=int, default=270)
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-iter", type=int, default=3)
    p.add_argument("--save-wavs", type=str, default=None,
                   help="prefix path; saves <prefix>_baseline.wav and <prefix>_int4.wav")
    args = p.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    base_dtype = dtype_map[args.baseline_dtype]

    from irodori_tts.codec import DACVAECodec

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    latent = torch.randn(1, args.latent_len, 32, dtype=base_dtype, device=device)
    latent_int4 = latent.clone()

    # ---- baseline ----
    print(f"\n=== baseline: dtype={args.baseline_dtype} ===")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    codec = DACVAECodec.load(repo_id=args.repo_id, device=device, dtype=base_dtype)
    inner = codec.model

    n_params = sum(p.numel() for p in inner.parameters())
    bytes_baseline = sum(p.numel() * p.element_size() for p in inner.parameters())
    print(f"params: {n_params/1e6:.1f}M   weight bytes: {bytes_baseline/1024**2:.1f} MB")

    with torch.no_grad():
        _ = _decode(codec, latent)  # warmup
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.n_iter):
            audio_base = _decode(codec, latent)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    dt_base = (time.perf_counter() - t0) / args.n_iter * 1000
    peak_alloc_base, peak_res_base = _peak_mb()
    print(f"decode: {dt_base:.1f} ms/it")
    print(f"peak alloc: {peak_alloc_base:.1f} MB   reserved: {peak_res_base:.1f} MB")
    audio_base_cpu = audio_base.detach().float().cpu()

    del codec, inner, audio_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- int4 ----
    print(f"\n=== int4 packed Conv (groupsize={args.groupsize}) ===")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load on CPU first so we don't hit GPU with fp32 transient
    codec = DACVAECodec.load(repo_id=args.repo_id, device="cpu", dtype=torch.float32)
    stats = replace_conv_with_packed(codec.model, groupsize=args.groupsize,
                                      cast_remaining_to=base_dtype)
    print(f"replaced {stats['replaced']} Conv modules")
    print(f"  bytes_before (fp32 conv weights): {stats['bytes_before']/1024**2:.1f} MB")
    print(f"  bytes_after  (int4 packed):       {stats['bytes_after']/1024**2:.1f} MB"
          f"  (ratio={stats['ratio']:.3f})")

    codec.model.to(device)
    # Update the codec's recorded dtype/device if it stores them
    if hasattr(codec, "device"):
        try:
            codec.device = torch.device(device)
        except Exception:
            pass
    if hasattr(codec, "dtype"):
        try:
            codec.dtype = base_dtype
        except Exception:
            pass

    bytes_int4_total = 0
    for p in codec.model.parameters():
        bytes_int4_total += p.numel() * p.element_size()
    for b in codec.model.buffers():
        bytes_int4_total += b.numel() * b.element_size()
    print(f"weight+buffer bytes after int4: {bytes_int4_total/1024**2:.1f} MB")

    with torch.no_grad():
        _ = _decode(codec, latent_int4)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.n_iter):
            audio_int4 = _decode(codec, latent_int4)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    dt_int4 = (time.perf_counter() - t0) / args.n_iter * 1000
    peak_alloc_int4, peak_res_int4 = _peak_mb()
    print(f"decode: {dt_int4:.1f} ms/it")
    print(f"peak alloc: {peak_alloc_int4:.1f} MB   reserved: {peak_res_int4:.1f} MB")
    audio_int4_cpu = audio_int4.detach().float().cpu()

    # ---- quality compare ----
    print(f"\n=== quality ===")
    print(f"audio shape: baseline={tuple(audio_base_cpu.shape)} int4={tuple(audio_int4_cpu.shape)}")
    if audio_base_cpu.shape == audio_int4_cpu.shape:
        diff = (audio_int4_cpu - audio_base_cpu).abs()
        ref = audio_base_cpu.abs()
        snr = 20 * torch.log10(
            ref.pow(2).mean().sqrt().clamp(min=1e-10)
            / diff.pow(2).mean().sqrt().clamp(min=1e-10)
        )
        print(f"max abs diff = {diff.max().item():.4e}")
        print(f"mean abs diff = {diff.mean().item():.4e}")
        print(f"baseline rms = {ref.pow(2).mean().sqrt().item():.4e}")
        print(f"SNR vs baseline = {snr.item():.2f} dB")
    else:
        print("shape mismatch — cannot compute SNR")

    if args.save_wavs:
        try:
            import soundfile as sf
            sr = 48000
            a0 = audio_base_cpu.squeeze().numpy()
            a1 = audio_int4_cpu.squeeze().numpy()
            sf.write(f"{args.save_wavs}_baseline.wav", a0, sr)
            sf.write(f"{args.save_wavs}_int4.wav", a1, sr)
            print(f"wrote {args.save_wavs}_baseline.wav / _int4.wav")
        except ImportError:
            print("soundfile not installed; skipping wav save")

    print(f"\n=== summary ===")
    print(f"baseline ({args.baseline_dtype}): {peak_alloc_base:.0f} MB peak, {dt_base:.1f} ms")
    print(f"int4                            : {peak_alloc_int4:.0f} MB peak, {dt_int4:.1f} ms")
    saved = peak_alloc_base - peak_alloc_int4
    print(f"VRAM saving: {saved:.0f} MB ({100*saved/max(peak_alloc_base,1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
