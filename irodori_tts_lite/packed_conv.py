"""Int4-packed Conv1d / ConvTranspose1d for DACVAE.

OneCompression の RTN 4-bit を Conv 重みに適用し、qweight (uint8 nibble pack) /
scales / zeros を buffer として保持。forward の中で 1 レイヤ分だけ dequant して
標準の F.conv1d / F.conv_transpose1d に渡すことで、

  * Disk: weight が 1/8 (fp32 比) になる
  * 実行時: モデル全体の重みは int4 のまま、瞬間的に fp16 化されるのは
    現在 forward 中のレイヤだけ → ピーク VRAM が大幅に下がる

NormConv1d / NormConvTranspose1d (dacvae.nn.layers) の pad/unpad ロジックも
そのまま再現してあるので、in-place 差し替えで動く。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_int4_nibbles(w_int: torch.Tensor) -> torch.Tensor:
    """(rows, cols) の 4-bit int を uint8 nibble pack に詰める."""
    rows, cols = w_int.shape
    if cols % 2 == 1:
        pad = torch.zeros(rows, 1, dtype=w_int.dtype, device=w_int.device)
        w_int = torch.cat([w_int, pad], dim=1)
        cols += 1
    a = w_int.to(torch.uint8)
    low = a[:, 0::2] & 0x0F
    high = (a[:, 1::2] & 0x0F) << 4
    return (low | high).contiguous()


def unpack_int4_nibbles(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """uint8 packed → int 値 (rows, cols)."""
    rows, half = packed.shape
    cols_padded = half * 2
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    interleaved = torch.stack([low, high], dim=-1).reshape(rows, cols_padded)
    if cols_padded != cols:
        interleaved = interleaved[:, :cols]
    return interleaved


def _dequant_weight(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                    cols: int, dtype: torch.dtype) -> torch.Tensor:
    w_int = unpack_int4_nibbles(qweight, cols).to(dtype)
    num_groups = scales.shape[1]
    if num_groups == 1:
        return (w_int - zeros.to(dtype)) * scales.to(dtype)
    gs = cols // num_groups
    if cols % num_groups != 0:
        raise RuntimeError(f"cols={cols} not divisible by num_groups={num_groups}")
    scale_e = scales.to(dtype).unsqueeze(-1)
    zero_e = zeros.to(dtype).unsqueeze(-1)
    w_int_g = w_int.reshape(w_int.shape[0], num_groups, gs)
    return ((w_int_g - zero_e) * scale_e).reshape(w_int.shape[0], cols)


def quantize_conv_weight(weight: torch.Tensor, groupsize: int = -1
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Conv 重み (..., in_dim_flat) を 4-bit に量子化."""
    if weight.dim() != 2:
        raise ValueError("flatten weight to 2-D before quantization")
    rows, cols = weight.shape
    if groupsize > 0 and cols % groupsize == 0:
        ng = cols // groupsize
        w = weight.float().reshape(rows, ng, groupsize)
    else:
        ng = 1
        w = weight.float().reshape(rows, 1, cols)

    q_max = 15
    w_max = w.amax(dim=-1, keepdim=True)
    w_min = w.amin(dim=-1, keepdim=True)
    dead = (w_max == 0) & (w_min == 0)
    w_max = torch.where(dead, torch.ones_like(w_max), w_max)
    w_min = torch.where(dead, -torch.ones_like(w_min), w_min)
    scale = ((w_max - w_min) / q_max).clamp(min=1e-8)
    zero = torch.round(-w_min / scale)
    q = torch.clamp(torch.round(w / scale) + zero, 0, q_max)

    q = q.to(torch.int32).reshape(rows, cols)
    scales = scale.squeeze(-1).to(torch.float16).reshape(rows, ng)
    zeros = zero.squeeze(-1).to(torch.float16).reshape(rows, ng)
    qweight = pack_int4_nibbles(q)
    return qweight, scales, zeros, cols


def _conv1d_pad(x: torch.Tensor, kernel_size: int, stride: int, dilation: int,
                pad_mode: str, causal: bool) -> torch.Tensor:
    if pad_mode == "none":
        return x
    length = x.shape[-1]
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    padding_total = effective_kernel_size - stride
    n_frames = (length - effective_kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    extra_padding = ideal_length - length
    if causal:
        return F.pad(x, (padding_total, extra_padding))
    padding_right = extra_padding // 2
    padding_left = padding_total - padding_right
    return F.pad(x, (padding_left, padding_right + extra_padding))


def _convtranspose1d_unpad(x: torch.Tensor, kernel_size: int, stride: int,
                            pad_mode: str, causal: bool) -> torch.Tensor:
    if pad_mode == "none":
        return x
    length = x.shape[-1]
    padding_total = kernel_size - stride
    if causal:
        return x[..., :length - padding_total]
    padding_right = padding_total // 2
    padding_left = padding_total - padding_right
    return x[..., padding_left:length - padding_right]


class PackedInt4Conv1d(nn.Module):
    """NormConv1d 互換の int4 packed 1D-Conv (pad/unpad 込み)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1,
                 qweight: torch.Tensor | None = None,
                 scales: torch.Tensor | None = None,
                 zeros: torch.Tensor | None = None,
                 cols: int | None = None,
                 bias: torch.Tensor | None = None,
                 causal: bool = False,
                 pad_mode: str = "auto"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        self.pad_mode = pad_mode
        if qweight is None:
            ck = (in_channels // groups) * kernel_size
            qweight = torch.zeros(out_channels, (ck + 1) // 2, dtype=torch.uint8)
            scales = torch.zeros(out_channels, 1, dtype=torch.float16)
            zeros = torch.zeros(out_channels, 1, dtype=torch.float16)
            cols = ck
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("zeros", zeros.contiguous())
        self.cols = int(cols)
        if bias is not None:
            self.register_parameter("bias", nn.Parameter(bias.detach(), requires_grad=False))
        else:
            self.bias = None

    def _materialize(self, dtype: torch.dtype) -> torch.Tensor:
        w = _dequant_weight(self.qweight, self.scales, self.zeros, self.cols, dtype)
        return w.reshape(self.out_channels, self.in_channels // self.groups, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _conv1d_pad(x, self.kernel_size, self.stride, self.dilation,
                        self.pad_mode, self.causal)
        w = self._materialize(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.conv1d(x, w, bias, self.stride, self.padding, self.dilation, self.groups)


class PackedInt4ConvTranspose1d(nn.Module):
    """NormConvTranspose1d 互換の int4 packed ConvTranspose1d (pad/unpad 込み)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1,
                 qweight: torch.Tensor | None = None,
                 scales: torch.Tensor | None = None,
                 zeros: torch.Tensor | None = None,
                 cols: int | None = None,
                 bias: torch.Tensor | None = None,
                 causal: bool = False,
                 pad_mode: str = "auto"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        self.pad_mode = pad_mode
        if qweight is None:
            ck = (out_channels // groups) * kernel_size
            qweight = torch.zeros(in_channels, (ck + 1) // 2, dtype=torch.uint8)
            scales = torch.zeros(in_channels, 1, dtype=torch.float16)
            zeros = torch.zeros(in_channels, 1, dtype=torch.float16)
            cols = ck
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("zeros", zeros.contiguous())
        self.cols = int(cols)
        if bias is not None:
            self.register_parameter("bias", nn.Parameter(bias.detach(), requires_grad=False))
        else:
            self.bias = None

    def _materialize(self, dtype: torch.dtype) -> torch.Tensor:
        w = _dequant_weight(self.qweight, self.scales, self.zeros, self.cols, dtype)
        return w.reshape(self.in_channels, self.out_channels // self.groups, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._materialize(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        y = F.conv_transpose1d(x, w, bias, self.stride, self.padding,
                                self.output_padding, self.groups, self.dilation)
        return _convtranspose1d_unpad(y, self.kernel_size, self.stride,
                                       self.pad_mode, self.causal)


def quantize_conv_module(mod: nn.Module, groupsize: int = 32) -> PackedInt4Conv1d | PackedInt4ConvTranspose1d:
    """NormConv1d / NormConvTranspose1d を Packed バージョンに変換して返す.

    呼び出し前に `nn.utils.remove_weight_norm(mod)` で weight_norm を外しておくこと.
    """
    from torch.nn import Conv1d, ConvTranspose1d

    weight = mod.weight.detach()
    bias = mod.bias.detach() if mod.bias is not None else None
    kernel_size = mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size
    stride = mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride
    padding = mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
    dilation = mod.dilation[0] if isinstance(mod.dilation, tuple) else mod.dilation
    groups = mod.groups
    causal = getattr(mod, "causal", False)
    pad_mode = getattr(mod, "pad_mode", "none")

    if isinstance(mod, ConvTranspose1d):
        # weight shape: (in_channels, out_channels // groups, kernel_size)
        out_padding = mod.output_padding[0] if isinstance(mod.output_padding, tuple) else mod.output_padding
        w2d = weight.reshape(weight.shape[0], -1)
        qweight, scales, zeros, cols = quantize_conv_weight(w2d, groupsize=groupsize)
        return PackedInt4ConvTranspose1d(
            in_channels=mod.in_channels, out_channels=mod.out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            output_padding=out_padding, dilation=dilation, groups=groups,
            qweight=qweight, scales=scales, zeros=zeros, cols=cols,
            bias=bias, causal=causal, pad_mode=pad_mode,
        )
    else:
        # Conv1d: weight shape (out_channels, in_channels // groups, kernel_size)
        w2d = weight.reshape(weight.shape[0], -1)
        qweight, scales, zeros, cols = quantize_conv_weight(w2d, groupsize=groupsize)
        return PackedInt4Conv1d(
            in_channels=mod.in_channels, out_channels=mod.out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups,
            qweight=qweight, scales=scales, zeros=zeros, cols=cols,
            bias=bias, causal=causal, pad_mode=pad_mode,
        )


def replace_conv_with_packed(model: nn.Module, *, groupsize: int = 32,
                              cast_remaining_to: torch.dtype | None = torch.float16) -> dict:
    """DACVAE モデルの NormConv1d / NormConvTranspose1d をすべて Packed に置き換える.

    Returns:
        statistics dict (replaced count, params before/after, bytes before/after)
    """
    target_classes = ("NormConv1d", "NormConvTranspose1d", "Conv1d", "ConvTranspose1d")
    replaced = 0
    bytes_before = 0
    bytes_after = 0

    # Two-pass: first collect, then replace (so we don't mutate during iteration)
    to_replace = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls in target_classes and hasattr(mod, "kernel_size"):
            to_replace.append((name, mod))

    for name, mod in to_replace:
        # Remove weight_norm parametrization if present
        try:
            from torch.nn.utils import remove_weight_norm
            remove_weight_norm(mod)
        except (ValueError, AttributeError):
            pass

        # Count bytes before
        for p in mod.parameters(recurse=False):
            bytes_before += p.numel() * p.element_size()

        packed = quantize_conv_module(mod, groupsize=groupsize)

        # Count bytes after
        for b in packed.buffers(recurse=False):
            bytes_after += b.numel() * b.element_size()
        for p in packed.parameters(recurse=False):
            bytes_after += p.numel() * p.element_size()

        # Find parent and swap
        parent_name, _, child_name = name.rpartition(".")
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        setattr(parent, child_name, packed)
        replaced += 1

    # Cast remaining float params/buffers (LSTM, Snake1d alpha, embedding) to target dtype
    if cast_remaining_to is not None:
        from .packed_conv import PackedInt4Conv1d, PackedInt4ConvTranspose1d
        for mod in model.modules():
            if isinstance(mod, (PackedInt4Conv1d, PackedInt4ConvTranspose1d)):
                continue
            for pname, p in list(mod._parameters.items()):
                if p is not None and p.dtype.is_floating_point and p.dtype != cast_remaining_to:
                    mod._parameters[pname] = nn.Parameter(
                        p.data.to(cast_remaining_to), requires_grad=p.requires_grad,
                    )
            for bname, b in list(mod._buffers.items()):
                if b is not None and b.dtype.is_floating_point and b.dtype != cast_remaining_to:
                    mod._buffers[bname] = b.to(cast_remaining_to)
        # LSTM flatten_parameters again after dtype change
        for mod in model.modules():
            if isinstance(mod, nn.LSTM):
                mod.flatten_parameters()

    return {
        "replaced": replaced,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "ratio": bytes_after / bytes_before if bytes_before else 0.0,
    }
