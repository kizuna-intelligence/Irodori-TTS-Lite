"""Patched checkpoint loader for int4-quantized TTS DiT.

Monkey-patches `irodori_tts.inference_runtime` so the standard
`InferenceRuntime.from_key` flow loads a quantized safetensors checkpoint
with:

  * DiT-block Linears swapped for `FusedInt4Linear` (Triton kernel)
  * AdaLN projections eager-dequantized to fp16 nn.Linear at load time
    (hot path; per-launch Triton overhead would dominate)
  * Encoder / cond_module / text_embedding extras dequantized to fp16
    nn.Linear / nn.Embedding (cold path; called once per inference)

Call `patch()` once before `inference_runtime.InferenceRuntime.from_key`
is used. Re-importing the standard `infer` CLI after patching gives you a
working int4 inference pipeline with no other code changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .fused_int4_linear import FusedInt4Linear
from .quant_utils import (
    dequant_extra_u8_to_weight,
    dequant_gptq_to_fp,
)


# AdaLN: 12 small projections × 12 DiT blocks × N RF steps ⇒ thousands of
# tiny Triton launches per inference. Pre-dequantize to fp16 nn.Linear.
_EAGER_DEQUANT_KEYWORDS = ("adaln",)


class _Options:
    use_fused: bool = True
    force_fp16: bool = False
    disable_eager: bool = False
    codec_int4: bool = False
    codec_int4_groupsize: int = 32


_opts = _Options()


def configure(
    *,
    use_fused: bool | None = None,
    force_fp16: bool | None = None,
    disable_eager: bool | None = None,
    codec_int4: bool | None = None,
    codec_int4_groupsize: int | None = None,
) -> None:
    """Adjust runtime knobs before the first `from_key` call."""
    if use_fused is not None:
        _opts.use_fused = use_fused
    if force_fp16 is not None:
        _opts.force_fp16 = force_fp16
    if disable_eager is not None:
        _opts.disable_eager = disable_eager
    if codec_int4 is not None:
        _opts.codec_int4 = codec_int4
    if codec_int4_groupsize is not None:
        _opts.codec_int4_groupsize = codec_int4_groupsize


def _should_eager_dequant(name: str) -> bool:
    if _opts.disable_eager:
        return False
    return any(kw in name for kw in _EAGER_DEQUANT_KEYWORDS)


def _can_use_fused(in_features: int, out_features: int, wbits: int, groupsize: int,
                   actorder: bool) -> bool:
    return (
        not actorder
        and wbits == 4
        and groupsize == 32
        and in_features % 32 == 0
        and out_features % 8 == 0
    )


_PENDING_SWAPS: dict[str, dict] = {}
_PENDING_EXTRA: dict[str, dict] = {}


def _patched_load(path):
    from irodori_tts import inference_runtime
    state, cfg, train_cfg = _orig_load(path)
    quant_layers = None
    extra_layers = None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            md = f.metadata() or {}
            if md.get("quant_method") == "autobit" and "quant_layers_json" in md:
                quant_layers = json.loads(md["quant_layers_json"])
            if "extra_quant_layers_json" in md:
                extra_layers = json.loads(md["extra_quant_layers_json"])
    except Exception:
        quant_layers = None
        extra_layers = None

    if not quant_layers and not extra_layers:
        return state, cfg, train_cfg

    if quant_layers:
        print(
            f"[irodori_tts_lite] detected packed AutoBit checkpoint with "
            f"{len(quant_layers)} quantized Linears"
        )
        _PENDING_SWAPS.clear()
        for entry in quant_layers:
            name = entry["name"]
            layer_state = {}
            for s in ("qweight", "scales", "qzeros", "g_idx", "bias"):
                k = f"{name}.{s}"
                if k in state:
                    layer_state[s] = state.pop(k)
            _PENDING_SWAPS[name] = {"entry": entry, "state": layer_state}

    if extra_layers:
        print(
            f"[irodori_tts_lite] detected {len(extra_layers)} extra-quant "
            f"(encoder/embedding) layers"
        )
        _PENDING_EXTRA.clear()
        for entry in extra_layers:
            name = entry["name"]
            tensors = {}
            for s in ("qweight_u8", "scales", "zeros"):
                k = f"_extra.{name}.{s}"
                if k in state:
                    tensors[s] = state.pop(k)
            bkey = f"{name}.bias"
            if bkey in state:
                tensors["bias"] = state.pop(bkey)
            _PENDING_EXTRA[name] = {"entry": entry, "tensors": tensors}

    return state, cfg, train_cfg


def _build_fused_from_entry(entry: dict, layer_state: dict, device: torch.device) -> FusedInt4Linear:
    return FusedInt4Linear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=layer_state["qweight"].to(device),
        scales=layer_state["scales"].to(device=device, dtype=torch.float16),
        qzeros=layer_state["qzeros"].to(device),
        bias=(layer_state["bias"].to(device) if "bias" in layer_state else None),
        groupsize=int(entry["groupsize"]),
    )


def _build_eager_linear(entry: dict, layer_state: dict, dtype: torch.dtype,
                        device: torch.device) -> torch.nn.Linear:
    in_f = int(entry["in_features"])
    out_f = int(entry["out_features"])
    has_bias = "bias" in layer_state and layer_state["bias"] is not None
    weight = dequant_gptq_to_fp(
        qweight=layer_state["qweight"],
        scales=layer_state["scales"],
        qzeros=layer_state["qzeros"],
        g_idx=layer_state["g_idx"],
        in_features=in_f,
        out_features=out_f,
        dtype=dtype,
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
    ).to(device)
    lin = torch.nn.Linear(in_f, out_f, bias=has_bias, device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if has_bias:
            lin.bias.copy_(layer_state["bias"].to(dtype=dtype, device=device))
    return lin


def _patched_from_key(cls, key):
    from irodori_tts import inference_runtime
    from irodori_tts.config import ModelConfig as _DiTModelConfig
    from irodori_tts.model import TextToLatentRFDiT
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from irodori_tts.codec import DACVAECodec

    model_device = inference_runtime.resolve_runtime_device(key.model_device)
    codec_device = inference_runtime.resolve_runtime_device(key.codec_device)
    model_dtype = inference_runtime.resolve_runtime_dtype(
        precision=key.model_precision, device=model_device
    )
    codec_dtype = inference_runtime.resolve_runtime_dtype(
        precision=key.codec_precision, device=codec_device
    )

    model_state, model_cfg_dict, train_cfg = inference_runtime._load_checkpoint_for_inference(
        Path(key.checkpoint)
    )
    model_cfg = _DiTModelConfig(**model_cfg_dict)

    # Build fp32 model on CPU first — at wide-scope quant the freshly-instantiated
    # fp32 model is ~2GB; swap quantized layers on CPU then move the assembled
    # (much smaller) model to GPU at the end.
    model = TextToLatentRFDiT(model_cfg)
    target_device = torch.device("cpu") if (_PENDING_SWAPS or _PENDING_EXTRA) else model_device
    model = model.to(target_device)

    eager_dtype = torch.float16 if _opts.force_fp16 else model_dtype
    fused_count = eager_count = fallback_count = 0
    if _PENDING_SWAPS:
        modules = dict(model.named_modules())
        for name, info in _PENDING_SWAPS.items():
            entry = info["entry"]
            layer_state = info["state"]
            parent_name, _, child_name = name.rpartition(".")
            parent = modules.get(parent_name) if parent_name else model

            if _should_eager_dequant(name):
                lin = _build_eager_linear(entry, layer_state, eager_dtype, target_device)
                setattr(parent, child_name, lin)
                eager_count += 1
            elif _opts.use_fused and _can_use_fused(
                int(entry["in_features"]), int(entry["out_features"]),
                int(entry["wbits"]), int(entry["groupsize"]),
                bool(entry.get("actorder", False)),
            ):
                fused = _build_fused_from_entry(entry, layer_state, target_device)
                setattr(parent, child_name, fused)
                fused_count += 1
            else:
                lin = _build_eager_linear(entry, layer_state, eager_dtype, target_device)
                setattr(parent, child_name, lin)
                fallback_count += 1
        _PENDING_SWAPS.clear()
        print(
            f"[irodori_tts_lite] fused={fused_count} eager_dequant={eager_count} "
            f"fallback={fallback_count}"
        )

    if _PENDING_EXTRA:
        modules_now = dict(model.named_modules())
        extra_count = 0
        for name, info in _PENDING_EXTRA.items():
            entry = info["entry"]
            tensors = info["tensors"]
            in_f = int(entry["in_features"])
            out_f = int(entry["out_features"])
            weight = dequant_extra_u8_to_weight(
                tensors["qweight_u8"], tensors["scales"], tensors["zeros"],
                in_f, out_f, eager_dtype,
            ).to(target_device)
            target_mod = modules_now.get(name)
            if target_mod is None:
                raise KeyError(f"extra-quant target not found in model: {name}")
            target_w = target_mod.weight
            with torch.no_grad():
                # Some "extras" are non-Linear (RMSNorm q_norm/k_norm stored
                # with shape (heads, head_dim) but packed as if (in=head_dim,
                # out=heads)). Reshape into the target module's parameter shape
                # in those cases so we don't corrupt the module class.
                if target_w.shape == weight.shape:
                    target_mod.weight = torch.nn.Parameter(weight, requires_grad=False)
                elif target_w.numel() == weight.numel():
                    target_mod.weight = torch.nn.Parameter(
                        weight.reshape(target_w.shape).contiguous(), requires_grad=False,
                    )
                else:
                    raise RuntimeError(
                        f"extra-quant shape mismatch at {name}: target {tuple(target_w.shape)} "
                        f"vs decoded {tuple(weight.shape)}"
                    )
            if "bias" in tensors and getattr(target_mod, "bias", None) is not None:
                target_mod.bias = torch.nn.Parameter(
                    tensors["bias"].to(dtype=eager_dtype, device=target_device),
                    requires_grad=False,
                )
            extra_count += 1
        _PENDING_EXTRA.clear()
        print(f"[irodori_tts_lite] extra_quant_dequanted={extra_count}")

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected[:8]}")

    # Cast remaining fp32 chunks to model_dtype on CPU, then move to GPU. This
    # avoids allocating fp32 transients in VRAM during the .to(device) hop.
    if target_device.type == "cpu" and model_device.type != "cpu":
        for mod in model.modules():
            if isinstance(mod, FusedInt4Linear):
                continue
            for pname, p in list(mod._parameters.items()):
                if p is not None and p.dtype.is_floating_point and p.dtype != model_dtype:
                    mod._parameters[pname] = torch.nn.Parameter(
                        p.data.to(model_dtype), requires_grad=p.requires_grad
                    )
            for bname, b in list(mod._buffers.items()):
                if b is not None and b.dtype.is_floating_point and b.dtype != model_dtype:
                    mod._buffers[bname] = b.to(model_dtype)
        model = model.to(model_device)

    if _opts.force_fp16:
        for _, mod in model.named_modules():
            if isinstance(mod, FusedInt4Linear):
                continue
            for pname, p in list(mod._parameters.items()):
                if p is not None and p.dtype.is_floating_point:
                    mod._parameters[pname] = torch.nn.Parameter(
                        p.data.to(torch.float16), requires_grad=p.requires_grad
                    )
            for bname, b in list(mod._buffers.items()):
                if b is not None and b.dtype.is_floating_point:
                    mod._buffers[bname] = b.to(torch.float16)
    else:
        model = model.to(dtype=model_dtype)
    model.eval()

    # Warm up Triton autotuner. DiT inference visits ~10 distinct M values;
    # the warmup cost (~1s) is paid once and amortises across every request.
    if _opts.use_fused and fused_count > 0:
        import time as _t
        t0 = _t.perf_counter()
        seen_sig: dict[tuple, FusedInt4Linear] = {}
        for mod in model.modules():
            if isinstance(mod, FusedInt4Linear):
                sig = (mod.in_features, mod.out_features, mod.bias is not None)
                seen_sig.setdefault(sig, mod)
        for layer in seen_sig.values():
            layer.warmup(m_values=(16, 63, 64, 120, 189, 192, 256, 360, 512, 768, 2250))
        print(
            f"[irodori_tts_lite] warmup done in {(_t.perf_counter() - t0) * 1000:.0f} ms "
            f"({len(seen_sig)} unique (K,N,has_bias) signatures)"
        )

    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=model_cfg.text_tokenizer_repo,
        add_bos=bool(model_cfg.text_add_bos),
        local_files_only=False,
    )
    caption_tokenizer = None
    if model_cfg.use_caption_condition:
        caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=model_cfg.caption_tokenizer_repo_resolved,
            add_bos=model_cfg.caption_add_bos_resolved,
            local_files_only=False,
        )

    default_text_max_len = 256
    default_caption_max_len = default_text_max_len
    if isinstance(train_cfg, dict):
        ckpt_text_max_len = train_cfg.get("max_text_len")
        if isinstance(ckpt_text_max_len, int) and ckpt_text_max_len > 0:
            default_text_max_len = int(ckpt_text_max_len)
        ckpt_caption_max_len = train_cfg.get("max_caption_len")
        if isinstance(ckpt_caption_max_len, int) and ckpt_caption_max_len > 0:
            default_caption_max_len = int(ckpt_caption_max_len)
        else:
            default_caption_max_len = default_text_max_len

    if _opts.codec_int4:
        # Load codec on CPU as fp32, swap Conv layers to packed int4, then move
        # to target device. Keeps peak VRAM low (no fp32 transient on GPU).
        codec = DACVAECodec.load(
            repo_id=key.codec_repo,
            device="cpu",
            dtype=torch.float32,
            deterministic_encode=bool(key.codec_deterministic_encode),
            deterministic_decode=bool(key.codec_deterministic_decode),
            enable_watermark=bool(key.enable_watermark),
        )
        from .packed_conv import replace_conv_with_packed
        cast_dtype = torch.float16 if _opts.force_fp16 else codec_dtype
        stats = replace_conv_with_packed(
            codec.model, groupsize=_opts.codec_int4_groupsize,
            cast_remaining_to=cast_dtype,
        )
        print(
            f"[irodori_tts_lite] codec int4: replaced {stats['replaced']} Conv layers, "
            f"weight bytes {stats['bytes_before']/1024**2:.0f} → "
            f"{stats['bytes_after']/1024**2:.0f} MB"
        )
        codec.model.to(codec_device)
        try:
            codec.device = torch.device(codec_device)
            codec.dtype = cast_dtype
        except Exception:
            pass
    else:
        codec = DACVAECodec.load(
            repo_id=key.codec_repo,
            device=str(codec_device),
            dtype=codec_dtype,
            deterministic_encode=bool(key.codec_deterministic_encode),
            deterministic_decode=bool(key.codec_deterministic_decode),
            enable_watermark=bool(key.enable_watermark),
        )

    return cls(
        key=key,
        model_cfg=model_cfg,
        train_cfg=train_cfg if isinstance(train_cfg, dict) else None,
        model=model,
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        codec=codec,
        default_text_max_len=default_text_max_len,
        default_caption_max_len=default_caption_max_len,
    )


_orig_load = None
_patched = False


def patch() -> None:
    """Install the quantized-checkpoint hooks into `irodori_tts.inference_runtime`.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _orig_load, _patched
    if _patched:
        return
    from irodori_tts import inference_runtime
    _orig_load = inference_runtime._load_checkpoint_for_inference
    inference_runtime._load_checkpoint_for_inference = _patched_load
    inference_runtime.InferenceRuntime.from_key = classmethod(_patched_from_key)
    _patched = True
