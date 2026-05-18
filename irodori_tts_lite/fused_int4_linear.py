"""Fused int4 dequantize + GEMM Triton kernel for OneCompression GPTQ packed
weights.

Vendored from the OneCompression project — copyright 2025-2026 Fujitsu Ltd.,
MIT License. See the top-level LICENSE file for terms. Imported into
Irodori-TTS-Lite verbatim so the runtime has no compile-time dep on
OneCompression.

The kernel reads OneCompression's AutoGPTQ-v1 packed format directly:
    qweight: shape (in_features // 8, out_features), int32
             — 8 nibbles per int32, packed along the K (in_features) axis
    qzeros : shape (num_groups, out_features // 8), int32
             — 8 nibbles per int32, packed along the N (out_features) axis,
               stored with v1 -1 offset (modular restore via (z + 1) & 0xF)
    scales : shape (num_groups, out_features), fp16
    g_idx  : ignored when actorder=False (implicit g = k // groupsize)

The kernel is designed for the DiT inference shape regime: M ~ 64-128
(batch * latent_seq), K in {512, 768, 1280, 3680}, N in {1280, 3680}.

Currently supports only:
    wbits = 4
    groupsize = 32
    actorder = False
    fp16 input/output

These cover 100% of the shipped quantized-DiT layers.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_int4_gemm_kernel(
    a_ptr,
    qw_ptr,
    s_ptr,
    qz_ptr,
    bias_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_qwk,
    stride_qwn,
    stride_sg,
    stride_sn,
    stride_qzg,
    stride_qzn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    GROUPSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_K_GROUPS: tl.constexpr,
    K_LOGICAL: tl.constexpr,
):
    """
    Computes  C = A @ dequant(qW) + bias

    A : [M, K_LOGICAL]      fp16   (K_LOGICAL <= K)
    qW: [K // 8, N]         int32   (8 nibbles per int32, packed along K)
    s : [K // GS, N]        fp16
    qz: [K // GS, N // 8]   int32   (8 nibbles per int32, packed along N, v1 offset)
    C : [M, N]              fp16

    Each program handles a [BLOCK_M, BLOCK_N] tile of the output. The K loop
    advances ``NUM_K_GROUPS`` groups (= NUM_K_GROUPS * GROUPSIZE K rows) per
    iteration so each tl.dot operates on K_BLK = NUM_K_GROUPS * GROUPSIZE,
    giving the tensor cores a fatter K dim.

    K_LOGICAL <= K supports K-padding: when the physical K (constexpr) is
    rounded up so K_BLK divides K, the activation is masked at offs_k <
    K_LOGICAL and the padded weight rows are stored as zero scales (so they
    contribute zero to the output regardless of the masked activation).
    Triton constant-folds the mask away when K_LOGICAL == K.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    K_BLK: tl.constexpr = NUM_K_GROUPS * GROUPSIZE  # K rows processed per outer iter
    N_WORDS: tl.constexpr = NUM_K_GROUPS * 4        # packed int32 rows per outer iter

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Per-K-row constants for unpacking and dequantising K_BLK elements at once.
    k_in_blk = tl.arange(0, K_BLK)              # [K_BLK]
    bit_off = (k_in_blk % 8) * 4                # [K_BLK] cycles 0,4,8,12,16,20,24,28
    # group_in_blk[k] = k // GROUPSIZE → which of the NUM_K_GROUPS subgroups owns row k

    # Per-N zero-extraction shifts (one int32 word covers 8 N-cols).
    qz_word_n = offs_n // 8                     # [BLOCK_N]
    qz_bit_n = (offs_n % 8) * 4                 # [BLOCK_N]

    a_row_ptrs = a_ptr + offs_m[:, None] * stride_am
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    g_arr = tl.arange(0, NUM_K_GROUPS)          # [NUM_K_GROUPS]
    word_arr = tl.arange(0, N_WORDS)            # [N_WORDS]

    # Skip padded K iterations: they would read zero-scales (producing zero contribution)
    # but still cost a full dequant+dot. ceil(K_LOGICAL/K_BLK) covers all real K rows.
    num_outer: tl.constexpr = (K_LOGICAL + K_BLK - 1) // K_BLK
    for outer in range(num_outer):
        g_base = outer * NUM_K_GROUPS
        # ---- Load scales [NUM_K_GROUPS, BLOCK_N] ----
        s_2d = tl.load(
            s_ptr + (g_base + g_arr)[:, None] * stride_sg + offs_n[None, :] * stride_sn,
            mask=n_mask[None, :],
            other=0.0,
        )

        # ---- Load packed zeros [NUM_K_GROUPS, BLOCK_N // 8] and unpack to [NUM_K_GROUPS, BLOCK_N] ----
        qz_packed = tl.load(
            qz_ptr + (g_base + g_arr)[:, None] * stride_qzg + qz_word_n[None, :] * stride_qzn,
            mask=n_mask[None, :],
            other=0,
        )
        qz_raw = (qz_packed >> qz_bit_n[None, :]) & 0xF
        qz_2d = (qz_raw + 1) & 0xF              # v1 modular restore, [NUM_K_GROUPS, BLOCK_N]

        # ---- Load N_WORDS packed weight rows as [N_WORDS, BLOCK_N] ----
        wp_base = g_base * 4
        w_words = tl.load(
            qw_ptr + (wp_base + word_arr)[:, None] * stride_qwk + offs_n[None, :] * stride_qwn,
            mask=n_mask[None, :],
            other=0,
        )

        # ---- Unpack to [K_BLK, BLOCK_N] int4 via broadcast+reshape ----
        # Each row of w_words is shared by 8 K rows differing only by bit_off.
        w_rep = tl.broadcast_to(w_words[:, None, :], (N_WORDS, 8, BLOCK_N))
        w_packed_per_k = tl.reshape(w_rep, (K_BLK, BLOCK_N))
        w_int = (w_packed_per_k >> bit_off[:, None]) & 0xF

        # ---- Expand per-group scales/zeros to [K_BLK, BLOCK_N] ----
        s_rep = tl.broadcast_to(s_2d[:, None, :], (NUM_K_GROUPS, GROUPSIZE, BLOCK_N))
        s_full = tl.reshape(s_rep, (K_BLK, BLOCK_N))
        qz_rep = tl.broadcast_to(qz_2d[:, None, :], (NUM_K_GROUPS, GROUPSIZE, BLOCK_N))
        qz_full = tl.reshape(qz_rep, (K_BLK, BLOCK_N))

        # ---- Dequant in A's dtype: w_fp = (w_int - qz) * s ----
        w_fp_h = (w_int.to(s_full.dtype) - qz_full.to(s_full.dtype)) * s_full

        # ---- Load A tile [BLOCK_M, K_BLK] and accumulate one fat dot ----
        offs_k = outer * K_BLK + k_in_blk
        if K_LOGICAL < K:
            # Padded layer: kernel sees K_PADDED >= K_LOGICAL; mask the K-dim
            # so OOB activation positions read 0 instead of trash.
            a_tile = tl.load(
                a_row_ptrs + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_LOGICAL),
                other=0.0,
            )
        else:
            # Aligned layer: K_LOGICAL == K, no K-mask needed; this branch
            # constant-folds at compile time so unpadded shapes pay no cost.
            a_tile = tl.load(
                a_row_ptrs + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M),
                other=0.0,
            )
        accumulator = tl.dot(a_tile, w_fp_h.to(a_tile.dtype), acc=accumulator)

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        accumulator += b[None, :].to(tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_tile = accumulator.to(c_ptrs.dtype.element_ty)
    tl.store(c_ptrs, c_tile, mask=(offs_m[:, None] < M) & n_mask[None, :])


def fused_int4_gemm(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    groupsize: int = 32,
) -> torch.Tensor:
    """Compute  ``c = a @ dequantize(qweight) + bias`` with a fused Triton kernel.

    Args:
        a:       [..., K] fp16, contiguous along K.
        qweight: [K // 8, N] int32 (OneCompression v1 GPTQ pack).
        scales:  [K // groupsize, N] fp16.
        qzeros:  [K // groupsize, N // 8] int32 (v1 -1 offset).
        bias:    optional [N] fp16/fp32.
        groupsize: per-group quantisation size; must be 32 in this kernel.

    Returns:
        [..., N] fp16 output (same leading shape as ``a``).
    """
    assert groupsize == 32, "fused_int4_gemm currently supports groupsize=32 only"
    assert qweight.dtype == torch.int32
    assert qzeros.dtype == torch.int32
    assert a.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"unsupported a.dtype={a.dtype}"
    )
    # If input is fp32, fall back to fp16 path (kernel requires 16-bit input).
    # We restore fp32 on the output below so callers see the dtype they passed.
    orig_dtype = a.dtype
    if a.dtype == torch.float32:
        a = a.to(torch.float16)
    # scales must match a's dtype for tl.dot type compatibility.
    if scales.dtype != a.dtype:
        scales = scales.to(a.dtype)

    leading_shape = a.shape[:-1]
    K = a.shape[-1]
    a_2d = a.reshape(-1, K).contiguous()
    M = a_2d.shape[0]
    N = scales.shape[1]

    assert qweight.shape == (K // 8, N), (
        f"qweight shape mismatch: expected ({K//8}, {N}), got {tuple(qweight.shape)}"
    )
    assert scales.shape == (K // groupsize, N), (
        f"scales shape mismatch: expected ({K//groupsize}, {N}), got {tuple(scales.shape)}"
    )
    assert qzeros.shape == (K // groupsize, N // 8), (
        f"qzeros shape mismatch: expected ({K//groupsize}, {N//8}), got {tuple(qzeros.shape)}"
    )
    assert N % 8 == 0, f"N must be a multiple of 8 (qzeros packing), got N={N}"
    assert K % 32 == 0, f"K must be a multiple of 32 (groupsize), got K={K}"

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # bias handling: kernel reads N-wide bias as same dtype as a
    if bias is not None:
        bias_h = bias.to(a.dtype).contiguous()
        has_bias = True
    else:
        bias_h = torch.empty(0, dtype=a.dtype, device=a.device)
        has_bias = False

    # Manual config heuristic — chosen by full sweep on RTX 3090 across all
    # production DiT shapes (M in {63, 76, 120, 189, 256, 360, 750, 768, 2250},
    # K in {512, 768, 1280, 3680}, N in {1280, 3680}).
    #
    # Small-M shapes need many small tiles to saturate the 82 SMs of a 3090;
    # large-M shapes prefer fatter M-tiles to amortise dequant overhead.
    if M <= 32:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 32, 32, 2, 4
    elif M <= 128:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 32, 32, 4, 3
    elif M <= 256:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 32, 4, 3
    elif M <= 1024:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 2
    elif N >= 2048 and K <= 1536:
        # (M=2250, K=1280, N=3680)-style: fat M-tile clears the win.
        BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 64, 4, 2
    else:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 2

    # NUM_K_GROUPS controls how many GPTQ groups (each GROUPSIZE K rows) are
    # consumed per outer loop iteration → tl.dot operates on K = NUM_K_GROUPS *
    # GROUPSIZE.  K_BLK must divide K.  Larger K_BLK amortises load/launch
    # overhead but consumes more shared memory (BM*K_BLK*2 + BN*K_BLK*2 per
    # stage); RTX 3090 caps SMEM at ~100KB.  nkg=8 (K_BLK=256) wins at
    # small-M tiles (BM*BN ≤ 1024) but exceeds SMEM at fat tiles.
    tile_area = BLOCK_M * BLOCK_N
    if tile_area <= 1024 and K % 256 == 0:
        NUM_K_GROUPS = 8
    elif tile_area <= 2048 and K % 128 == 0:
        NUM_K_GROUPS = 4
    elif K % 128 == 0:
        NUM_K_GROUPS = 4
    elif K % 64 == 0:
        NUM_K_GROUPS = 2
    else:
        NUM_K_GROUPS = 1

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _fused_int4_gemm_kernel[grid](
        a_2d,
        qweight,
        scales,
        qzeros,
        bias_h,
        c,
        M,
        N,
        K,
        a_2d.stride(0),
        a_2d.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS=has_bias,
        GROUPSIZE=groupsize,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_K_GROUPS=NUM_K_GROUPS,
        K_LOGICAL=K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out = c.reshape(*leading_shape, N)
    if orig_dtype == torch.float32:
        out = out.to(torch.float32)
    return out


class FusedInt4Linear(torch.nn.Module):
    """nn.Module wrapper around the fused int4 GEMM kernel.

    Drop-in replacement for a 4-bit / groupsize=32 GPTQLinear loaded via
    ``GPTQLinear.from_saved_state``.  Holds packed buffers verbatim and
    dispatches to the Triton kernel in ``forward``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: torch.Tensor | None = None,
        groupsize: int = 32,
    ):
        super().__init__()
        assert groupsize == 32, "FusedInt4Linear supports groupsize=32 only"
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize

        # Pad K up to the next multiple of 256 so the kernel can use
        # NUM_K_GROUPS=8 (K_BLK=256) — its fastest path on Ampere — uniformly.
        # Without padding, K values like 3680 force NUM_K_GROUPS=1 (K_BLK=32),
        # which on the (M=63,K=3680,N=1280) shape is 2.4× slower than cuBLAS
        # bf16. Padding with zero scales/qweight makes the dequant in those
        # rows produce 0, and the kernel masks the activation load to OOB→0,
        # so the output is identical to the unpadded computation.
        K_PAD_TO = 256
        k_pad = ((in_features + K_PAD_TO - 1) // K_PAD_TO) * K_PAD_TO
        self._k_padded = k_pad
        if k_pad != in_features:
            qweight = self._pad_qweight(qweight, in_features, k_pad)
            scales = self._pad_scales(scales, in_features, k_pad, groupsize)
            qzeros = self._pad_qzeros(qzeros, in_features, k_pad, groupsize)

        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("qzeros", qzeros.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
            self._has_bias = True
        else:
            self.bias = None
            self._has_bias = False
        # Empty placeholder used when HAS_BIAS=False; rebuilt lazily in forward
        # to match input dtype (bf16 vs fp16). Cached as attribute (not buffer)
        # so _apply doesn't recurse.
        self._empty_bias: torch.Tensor | None = None
        # Per-M-bucket kernel launch config cache. Diffusion sampling visits
        # only ~10 unique M values, so this dict stays tiny and lookup is ~0.1us
        # vs the ~5-10us re-derivation of (BLOCK_M, BLOCK_N, num_warps,
        # num_stages, NUM_K_GROUPS, grid) on every call.
        self._cfg_cache: dict[int, tuple] = {}

    @staticmethod
    def _pad_qweight(qw: torch.Tensor, k_orig: int, k_pad: int) -> torch.Tensor:
        # qweight shape (K // 8, N), int32. Padded rows are 0 → unpacked nibbles
        # are 0; combined with zero scales below, contribute 0 to the output.
        rows_orig = k_orig // 8
        rows_pad = k_pad // 8
        if qw.shape[0] != rows_orig:
            raise ValueError(f"qweight K-dim mismatch: got {qw.shape[0]}, expected {rows_orig}")
        out = torch.zeros((rows_pad, qw.shape[1]), dtype=qw.dtype, device=qw.device)
        out[:rows_orig].copy_(qw)
        return out

    @staticmethod
    def _pad_scales(s: torch.Tensor, k_orig: int, k_pad: int, gs: int) -> torch.Tensor:
        # scales shape (K // gs, N). Pad with 0.0 → padded dequant rows are 0.
        g_orig = k_orig // gs
        g_pad = k_pad // gs
        out = torch.zeros((g_pad, s.shape[1]), dtype=s.dtype, device=s.device)
        out[:g_orig].copy_(s)
        return out

    @staticmethod
    def _pad_qzeros(qz: torch.Tensor, k_orig: int, k_pad: int, gs: int) -> torch.Tensor:
        # qzeros shape (K // gs, N // 8). Pad with 0; doesn't matter what the
        # zero-value is in padded rows because scale=0 zeroes the dequant.
        g_orig = k_orig // gs
        g_pad = k_pad // gs
        out = torch.zeros((g_pad, qz.shape[1]), dtype=qz.dtype, device=qz.device)
        out[:g_orig].copy_(qz)
        return out

    @classmethod
    def from_gptq_linear(cls, layer) -> "FusedInt4Linear":
        """Build a FusedInt4Linear from a GPTQLinear (4-bit, groupsize=32).

        Reuses the saved packed buffers verbatim — no re-quantisation.
        """
        assert layer.wbits == 4, f"FusedInt4Linear requires wbits=4, got {layer.wbits}"
        assert (
            layer.groupsize == 32
        ), f"FusedInt4Linear requires groupsize=32, got {layer.groupsize}"
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            qweight=layer.qweight,
            scales=layer.scales.to(torch.float16),
            qzeros=layer.qzeros,
            bias=layer.bias if layer.bias is not None else None,
            groupsize=32,
        )

    def _build_cfg(self, M: int) -> tuple:
        """Compute and cache the (BLOCK_M, BLOCK_N, num_warps, num_stages,
        NUM_K_GROUPS, grid) tuple for this layer at activation-shape M.

        Heuristic from RTX 3090 sweep across production DiT shapes; small-M
        memory-bound regime favours nkg=8 (K_BLK=256) which beats cuBLAS bf16
        per-shape (11.7us vs 13us at M=63 K=N=1280). K is padded to a multiple
        of 256 in __init__, so nkg=8 is always available regardless of the
        layer's logical K.
        """
        K = self._k_padded
        N = self.out_features
        if M <= 32:
            BLOCK_M, BLOCK_N, num_warps, num_stages = 32, 32, 2, 4
        elif M <= 128:
            if N >= 2048:
                # Wide-N at M=63: nkg=8 (K_BLK=256) is wasteful; nkg=4 ns=2
                # ties cuBLAS (26us vs 25us) where nkg=8 ran 36us.
                BLOCK_M, BLOCK_N, num_warps, num_stages = 32, 32, 4, 2
            else:
                BLOCK_M, BLOCK_N, num_warps, num_stages = 32, 32, 4, 3
        elif M <= 256:
            if N >= 2048:
                # Wide-N at M=256 (e.g., 256×1280×3680): tall tile boosts
                # arithmetic intensity along K and lets nkg=1 pack more CTAs/SM.
                # Sweep-best on RTX 3090: BM=128 BN=32 nkg=1 → 54us, beats
                # cuBLAS 61us by 12%.
                BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 32, 4, 3
            else:
                BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 32, 4, 3
        elif M <= 1024:
            BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 2
        elif N >= 2048 and K <= 1536:
            BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 64, 4, 2
        else:
            BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 2

        tile_area = BLOCK_M * BLOCK_N
        if tile_area <= 1024 and K % 256 == 0:
            NUM_K_GROUPS = 8
        elif K % 128 == 0:
            NUM_K_GROUPS = 4
        elif K % 64 == 0:
            NUM_K_GROUPS = 2
        else:
            NUM_K_GROUPS = 1

        # Occupancy cap for large M: each stage's SMEM scales with K_BLK, so a
        # fatter K_BLK forces fewer CTAs/SM. For large M the kernel becomes
        # occupancy-bound and lower nkg wins (stable bench on RTX 3090, M=2250):
        #   K=N=1280:   nkg=1 134us beats nkg=2 141us
        #   K=3680 N=1280: nkg=1 416us beats nkg=2 424us
        # Wide-N (N>=2048) hits the BM=128 BN=64 branch and prefers nkg=2 there:
        #   K=1280 N=3680: nkg=2 ≈ 387us beats nkg=1.
        if M > 1024:
            if N >= 2048 and K <= 1536:
                NUM_K_GROUPS = min(NUM_K_GROUPS, 2)
            else:
                NUM_K_GROUPS = 1
        elif 128 < M <= 256:
            # Sweep on RTX 3090: nkg=4 (current default at this BM/BN) loses to
            # nkg=2 by 4-6us across (256,3680,1280) and (256,1280,1280); for
            # wide-N (N>=2048) the tall tile picked above prefers nkg=1.
            if N >= 2048:
                NUM_K_GROUPS = 1
            else:
                NUM_K_GROUPS = min(NUM_K_GROUPS, 2)
        elif M <= 128 and N >= 2048:
            # Small-M wide-N: nkg=4 (K_BLK=128) ties cuBLAS where nkg=8 lags
            # 40%. With ns=2 (set above) SMEM fits well.
            NUM_K_GROUPS = min(NUM_K_GROUPS, 4)

        grid = ((M + BLOCK_M - 1) // BLOCK_M * ((N + BLOCK_N - 1) // BLOCK_N),)
        cfg = (BLOCK_M, BLOCK_N, num_warps, num_stages, NUM_K_GROUPS, grid)
        self._cfg_cache[M] = cfg
        return cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path: skip the public ``fused_int4_gemm`` wrapper's asserts,
        # shape checks and dtype dispatch — they fire on every step of
        # diffusion sampling and add measurable Python overhead.  All
        # invariants are guaranteed by ``__init__`` + the model's dtype cast.
        if x.dtype == torch.float32:
            return fused_int4_gemm(
                x, self.qweight, self.scales, self.qzeros, self.bias, 32
            )

        x_dim = x.dim()
        if x_dim == 2:
            a_2d = x
            M = x.shape[0]
        else:
            a_2d = x.reshape(-1, self.in_features)
            M = a_2d.shape[0]

        cfg = self._cfg_cache.get(M)
        if cfg is None:
            cfg = self._build_cfg(M)
        BLOCK_M, BLOCK_N, num_warps, num_stages, NUM_K_GROUPS, grid = cfg

        if self._has_bias:
            bias_h = self.bias
        else:
            cached = self._empty_bias
            if cached is None or cached.dtype != x.dtype or cached.device != x.device:
                cached = torch.empty(0, dtype=x.dtype, device=x.device)
                self._empty_bias = cached
            bias_h = cached

        N = self.out_features
        c = torch.empty((M, N), dtype=x.dtype, device=x.device)
        _fused_int4_gemm_kernel[grid](
            a_2d, self.qweight, self.scales, self.qzeros, bias_h, c,
            M, N, self._k_padded,
            a_2d.stride(0), a_2d.stride(1),
            self.qweight.stride(0), self.qweight.stride(1),
            self.scales.stride(0), self.scales.stride(1),
            self.qzeros.stride(0), self.qzeros.stride(1),
            c.stride(0), c.stride(1),
            HAS_BIAS=self._has_bias,
            GROUPSIZE=32,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            NUM_K_GROUPS=NUM_K_GROUPS,
            K_LOGICAL=self.in_features,
            num_warps=num_warps, num_stages=num_stages,
        )
        if x_dim == 2:
            return c
        return c.reshape(*x.shape[:-1], N)

    @torch.no_grad()
    def warmup(self, m_values=(32, 128, 256, 1024, 2250)) -> None:
        """Trigger Triton JIT compilation for each (M-bucket, K, N) signature
        the heuristic will pick at inference time. Without this, the first
        forward at a never-seen M pays ~150-200 ms compilation cost.

        Skips M values that exceed in_features cap or are otherwise invalid.
        """
        device = self.qweight.device
        dtype = self.scales.dtype if self.scales.dtype in (torch.float16, torch.bfloat16) else torch.float16
        for m in m_values:
            if m <= 0:
                continue
            x = torch.zeros((int(m), self.in_features), dtype=dtype, device=device)
            _ = self.forward(x)
        torch.cuda.synchronize() if device.type == "cuda" else None

    def _apply(self, fn, recurse=True):
        # Override _apply so .to(dtype=...) on the parent module doesn't cast our
        # packed integer buffers (qweight, qzeros) — those must remain int32 —
        # and keeps scales in a 16-bit float dtype matching the model.
        qweight_before = self.qweight
        qzeros_before = self.qzeros
        result = super()._apply(fn, recurse=recurse)
        if self.qweight.dtype != torch.int32:
            self.qweight = qweight_before.to(self.qweight.device)
        if self.qzeros.dtype != torch.int32:
            self.qzeros = qzeros_before.to(self.qzeros.device)
        # If scales got cast to fp32 (default model dtype), bring back to fp16
        # so the kernel input dtype is at most upcast in one place. bf16 is
        # left untouched so the kernel runs natively without conversion.
        if self.scales.dtype not in (torch.float16, torch.bfloat16):
            self.scales = self.scales.to(torch.float16)
        return result
