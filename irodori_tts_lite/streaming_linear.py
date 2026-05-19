"""Streaming (lazy) int4 dequant Linear for the AdaLN hot path.

Keeps the GPTQ-v1 packed weight resident on GPU as int32 / fp16 / int32, and
materialises the fp16 weight inside `forward()` for the current call only.
The materialised tensor is dropped immediately after `F.linear` returns, so
peak VRAM is governed by activation size rather than the sum of all AdaLN
projection weights kept simultaneously in fp16.

Tradeoff vs the existing eager-dequant path
-------------------------------------------
* Eager (default): each AdaLN projection is dequantised once at load time into
  an `nn.Linear`. The fp16 weight stays in VRAM permanently.
  - Pros: zero per-step dequant overhead, plain cuBLAS GEMM.
  - Cons: 12 blocks × ~12 small projections × (1280×1280 fp16) ≈ 30-60 MB of
    permanent fp16 weight residency, on top of the int4 DiT body.
* Streaming (this module): packed int4 stays in VRAM, dequant happens per call.
  - Pros: ~30-60 MB lower peak VRAM (matches the upper bound estimated in the
    PR description).
  - Cons: extra dequant arithmetic per AdaLN call (thousands per inference);
    ~5-15% latency regression expected, measured on a real device.

This is opt-in via `configure(adaln_streaming=True)`. The eager path remains
the default because the latency hit is non-trivial; choose streaming when
fitting under a tight VRAM budget matters more than throughput.

The path mirrors `packed_conv.PackedInt4Conv1d` (also "load int4, dequant in
forward"), so the runtime stays self-contained and uses the existing
`dequant_gptq_to_fp` helper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_utils import dequant_gptq_to_fp


class StreamingInt4Linear(nn.Module):
    """nn.Linear-compatible module that dequantises GPTQ-v1 int4 in forward().

    Built from the same GPTQ-v1 packed buffers as `FusedInt4Linear`. Designed
    for the AdaLN hot path where the per-launch Triton overhead of the fused
    kernel exceeds the cuBLAS fp16 cost, but the weight footprint of eager
    dequant matters.

    Buffers held verbatim:
        qweight: int32 (K // 8, N) — AutoGPTQ-v1 packed (8 nibbles per int32)
        scales : fp16  (K // groupsize, N)
        qzeros : int32 (K // groupsize, N // 8) — v1 -1 offset
        g_idx  : int32 (K,) — explicit group index per K row
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        g_idx: torch.Tensor,
        bias: torch.Tensor | None = None,
        groupsize: int = 32,
        v1: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self._v1 = v1
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("qzeros", qzeros.contiguous())
        self.register_buffer("g_idx", g_idx.contiguous())
        if bias is not None:
            self.register_parameter(
                "bias", nn.Parameter(bias.detach(), requires_grad=False)
            )
        else:
            self.bias = None

    def _materialize(self, dtype: torch.dtype) -> torch.Tensor:
        # dequant_gptq_to_fp returns (out, in) in the requested dtype.
        return dequant_gptq_to_fp(
            qweight=self.qweight,
            scales=self.scales,
            qzeros=self.qzeros,
            g_idx=self.g_idx,
            in_features=self.in_features,
            out_features=self.out_features,
            dtype=dtype,
            v1=self._v1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._materialize(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def _apply(self, fn, recurse=True):
        # Match FusedInt4Linear's behaviour: never let parent .to(dtype) cast
        # packed integer buffers, and keep `scales` in 16-bit float.
        qweight_before = self.qweight
        qzeros_before = self.qzeros
        g_idx_before = self.g_idx
        result = super()._apply(fn, recurse=recurse)
        if self.qweight.dtype != torch.int32:
            self.qweight = qweight_before.to(self.qweight.device)
        if self.qzeros.dtype != torch.int32:
            self.qzeros = qzeros_before.to(self.qzeros.device)
        if self.g_idx.dtype != torch.int32:
            self.g_idx = g_idx_before.to(self.g_idx.device)
        if self.scales.dtype not in (torch.float16, torch.bfloat16):
            self.scales = self.scales.to(torch.float16)
        return result
