"""Irodori-TTS-Lite — int4-quantized TTS DiT inference runtime.

A self-contained drop-in that lets the standard `irodori_tts.inference_runtime`
load 4-bit quantized DiT checkpoints with the FusedInt4Linear Triton kernel.
~85% smaller on disk and ~71% lower peak GPU memory than fp32 at near-identical
audio quality.

Usage::

    import irodori_tts_lite
    irodori_tts_lite.patch()   # install hooks into irodori_tts.inference_runtime

    # Now use irodori_tts as normal; pass the int4 safetensors as `--checkpoint`.
    import infer
    infer.main()
"""
from __future__ import annotations

from .checkpoint_loader import configure, patch
from .fused_int4_linear import FusedInt4Linear, fused_int4_gemm
from .packed_conv import (
    PackedInt4Conv1d,
    PackedInt4ConvTranspose1d,
    replace_conv_with_packed,
)

__all__ = [
    "FusedInt4Linear",
    "fused_int4_gemm",
    "configure",
    "patch",
    "PackedInt4Conv1d",
    "PackedInt4ConvTranspose1d",
    "replace_conv_with_packed",
]

__version__ = "0.1.0"
