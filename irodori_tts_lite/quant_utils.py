"""Minimal 4-bit unpack + RTN-extras dequant helpers.

Vendored from OneCompression's onecomp.quantizer.gptq.gptq_layer (AutoGPTQ-v1
packing format) and the inverse of the encoders RTN post-pass packer. Only the
4-bit code paths are kept — Irodori-TTS-Lite ships int4 checkpoints exclusively.
"""
from __future__ import annotations

import torch


def _unpack_rows_int4(packed: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Unpack INT32-packed 4-bit values along dim-0 back to int values."""
    packed_rows, cols = packed.shape
    pack_factor = 8  # 32 // 4
    unpacked = torch.zeros(
        packed_rows, pack_factor, cols, dtype=torch.int32, device=packed.device
    )
    for i in range(pack_factor):
        unpacked[:, i, :] = (packed >> (i * 4)) & 0x0F
    return unpacked.reshape(packed_rows * pack_factor, cols)[:num_rows]


def unpack_int_weights(
    packed: torch.Tensor, wbits: int, original_shape: tuple[int, int]
) -> torch.Tensor:
    """Unpack AutoGPTQ-format packed weights, 4-bit only."""
    if wbits != 4:
        raise ValueError(f"only wbits=4 supported, got {wbits}")
    in_features = original_shape[1]
    unpacked = _unpack_rows_int4(packed, in_features)
    return unpacked.t().contiguous()


def unpack_zeros(packed_zeros: torch.Tensor, wbits: int, out_features: int) -> torch.Tensor:
    """Unpack AutoGPTQ-format packed zero points, 4-bit only."""
    if wbits != 4:
        raise ValueError(f"only wbits=4 supported, got {wbits}")
    return _unpack_rows_int4(packed_zeros.t().contiguous(), out_features).t().contiguous()


def dequant_gptq_to_fp(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    v1: bool = True,
) -> torch.Tensor:
    """Dequantize an AutoGPTQ-v1 packed 4-bit Linear back to (out, in) `dtype`."""
    weight_int = unpack_int_weights(qweight, 4, (out_features, in_features))
    zeros = unpack_zeros(qzeros, 4, out_features)
    if v1:
        zeros = (zeros + 1) & 0x0F
    scale_expanded = scales[g_idx, :].T
    zero_expanded = zeros[g_idx, :].T
    weight = scale_expanded.float() * (weight_int.float() - zero_expanded.float())
    return weight.to(dtype)


def dequant_extra_u8_to_weight(
    qweight_u8: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inverse of the encoders RTN post-pass uint8-nibble packer.

    qweight_u8: (out_features, in_padded // 2), uint8 — two 4-bit values per byte
    scales/zeros: (out_features, num_groups), fp/int
    """
    in_padded = qweight_u8.shape[1] * 2
    low = (qweight_u8 & 0x0F).to(torch.int16)
    high = ((qweight_u8 >> 4) & 0x0F).to(torch.int16)
    interleaved = torch.stack([low, high], dim=-1).reshape(out_features, in_padded)
    if in_padded != in_features:
        interleaved = interleaved[:, :in_features]
    w_int = interleaved.to(dtype)
    num_groups = scales.shape[1]
    if num_groups == 1:
        return (w_int - zeros.to(dtype)) * scales.to(dtype)
    if in_features % num_groups != 0:
        raise RuntimeError(
            f"in_features={in_features} not divisible by num_groups={num_groups}"
        )
    gs = in_features // num_groups
    scale_e = scales.to(dtype).unsqueeze(-1)
    zero_e = zeros.to(dtype).unsqueeze(-1)
    w_int_g = w_int.reshape(out_features, num_groups, gs)
    return ((w_int_g - zero_e) * scale_e).reshape(out_features, in_features)
