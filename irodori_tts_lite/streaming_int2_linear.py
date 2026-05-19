"""Streaming 2-bit Linear for the DiT FFN.

Layout (deliberately a smaller cousin of the 4-bit `extra_quant_layers_json`
uint8-nibble pack — same packing direction, half the bits, four values per
byte instead of two):

    qweight_u8: shape (out_features, ceil(in_features / 4)), uint8
                — bits [0:2] = val 0, [2:4] = val 1, [4:6] = val 2, [6:8] = val 3
                  4-bit quant values are kept in [0, 3]
    scales    : shape (out_features, num_groups), fp16
    zeros     : shape (out_features, num_groups), fp16 / int

The quantisation grain is per-row, per-group RTN: ``num_groups`` is normally
``in_features // groupsize`` (groupsize 64 by default), with ``num_groups=1``
collapsing to per-row quant.

Why streaming, not eager
------------------------
DiT FFN (``mlp.w1/w2/w3``) is the body of every block: dequanting once at
load time would mean keeping a fp16 copy of the very weights we just shrank
to 2 bits, defeating the purpose. Streaming pays a per-call dequant cost,
which for FFN-shaped GEMMs (M ~ 64-256, K/N in {1280, 3680}) is dominated
by the subsequent F.linear call anyway.

This module deliberately keeps the dequant in pure PyTorch (no Triton). A
2-bit + fp16 GEMM kernel is the obvious next step but is left out of this
PR — the streaming dequant path lets the rest of the loader/tooling settle
first, and the kernel can drop in later behind the same module API.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_int2_u8(w_int: torch.Tensor) -> torch.Tensor:
    """(out, in) int in [0, 3] → packed (out, ceil(in / 4)) uint8.

    Layout: byte[i, j] = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6) where vk
    is the (4*j + k)-th 2-bit value. Padding (when in_features is not a
    multiple of 4) is zero-filled on the right and is dropped by
    `unpack_int2_u8`.
    """
    out_f, in_f = w_int.shape
    pad = (4 - in_f % 4) % 4
    if pad:
        w_int = torch.cat(
            [w_int, torch.zeros(out_f, pad, dtype=w_int.dtype, device=w_int.device)],
            dim=1,
        )
    in_padded = w_int.shape[1]
    w_4 = w_int.reshape(out_f, in_padded // 4, 4).to(torch.uint8)
    packed = (
        (w_4[..., 0] & 0x3)
        | ((w_4[..., 1] & 0x3) << 2)
        | ((w_4[..., 2] & 0x3) << 4)
        | ((w_4[..., 3] & 0x3) << 6)
    )
    return packed.contiguous()


def unpack_int2_u8(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """(out, k) uint8 → (out, in_features) int16.

    Drops the trailing padding to match `pack_int2_u8`'s round-trip.
    """
    out_f, k = packed.shape
    in_padded = k * 4
    p = packed.to(torch.int16)
    v0 = p & 0x3
    v1 = (p >> 2) & 0x3
    v2 = (p >> 4) & 0x3
    v3 = (p >> 6) & 0x3
    interleaved = torch.stack([v0, v1, v2, v3], dim=-1).reshape(out_f, in_padded)
    if in_padded != in_features:
        interleaved = interleaved[:, :in_features]
    return interleaved


def rtn_quantize_2bit(w: torch.Tensor, groupsize: int = 64):
    """(out, in) fp → ((out, in) int [0,3], (out, num_g) fp16 scales, (out, num_g) fp16 zeros).

    Per-row, per-group min/max RTN at wbits=2. `groupsize` of -1 collapses
    to whole-row (num_groups = 1).
    """
    out_f, in_f = w.shape
    if groupsize <= 0:
        num_g = 1
        w_g = w.float().reshape(out_f, 1, in_f)
    else:
        if in_f % groupsize != 0:
            raise ValueError(
                f"in_features={in_f} not divisible by groupsize={groupsize}"
            )
        num_g = in_f // groupsize
        w_g = w.float().reshape(out_f, num_g, groupsize)

    q_max = 3
    w_max = w_g.amax(dim=-1, keepdim=True)
    w_min = w_g.amin(dim=-1, keepdim=True)
    dead = (w_max == w_min)
    w_max = torch.where(dead, w_max + 1.0, w_max)
    w_min = torch.where(dead, w_min - 1.0, w_min)
    scale = ((w_max - w_min) / q_max).clamp(min=1e-8)
    zero = torch.round(-w_min / scale).clamp(0, q_max)
    q = torch.clamp(torch.round(w_g / scale) + zero, 0, q_max).to(torch.int32)
    w_int = q.reshape(out_f, in_f)
    scales = scale.squeeze(-1).to(torch.float16)  # (out, num_g)
    zeros = zero.squeeze(-1).to(torch.float16)  # (out, num_g)
    return w_int, scales, zeros


def _dequant_grouped(
    w_int: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """(out, in) int + (out, num_g) scales/zeros → (out, in) fp dtype."""
    out_f, in_f = w_int.shape
    num_g = scales.shape[1]
    if num_g == 1:
        return (w_int.to(dtype) - zeros.to(dtype)) * scales.to(dtype)
    gs = in_f // num_g
    if in_f % num_g != 0:
        raise RuntimeError(f"in_features={in_f} not divisible by num_groups={num_g}")
    s = scales.to(dtype).unsqueeze(-1)  # (out, num_g, 1)
    z = zeros.to(dtype).unsqueeze(-1)
    w_g = w_int.to(dtype).reshape(out_f, num_g, gs)
    return ((w_g - z) * s).reshape(out_f, in_f)


class StreamingInt2Linear(nn.Module):
    """nn.Linear-compatible 2-bit weight Linear with per-call dequant."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight_u8: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight_u8", qweight_u8.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("zeros", zeros.contiguous())
        if bias is not None:
            self.register_parameter(
                "bias", nn.Parameter(bias.detach(), requires_grad=False)
            )
        else:
            self.bias = None

    def _materialize(self, dtype: torch.dtype) -> torch.Tensor:
        w_int = unpack_int2_u8(self.qweight_u8, self.in_features)
        return _dequant_grouped(w_int, self.scales, self.zeros, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._materialize(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def _apply(self, fn, recurse=True):
        # Packed weights are uint8 — never let .to(dtype) cast them.
        qw_before = self.qweight_u8
        result = super()._apply(fn, recurse=recurse)
        if self.qweight_u8.dtype != torch.uint8:
            self.qweight_u8 = qw_before.to(self.qweight_u8.device)
        if self.scales.dtype not in (torch.float16, torch.bfloat16):
            self.scales = self.scales.to(torch.float16)
        if self.zeros.dtype not in (torch.float16, torch.bfloat16):
            self.zeros = self.zeros.to(torch.float16)
        return result
