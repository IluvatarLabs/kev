# Modified implementation for Kev; see LICENSE and NOTICE.
"""Qwen3 backbone adapter: the parent's Qwen3Model (all layers, final norm, tied embedding matrix as `embed_tokens`; no
lm_head, no vocabulary projection), the attention backend, and prefix-cache replication for batched question branches.

Attention backend (verified against transformers 5.17.0 source): Qwen3Model hands its attention_mask to
masking_utils.create_causal_mask, whose _preprocess_mask_arguments returns any 4D tensor unchanged ("If the mask is
already 4D, simply return as-is"); integrations/sdpa_attention.py then passes it to
torch.nn.functional.scaled_dot_product_attention as `attn_mask` (is_causal is forced off when a mask is given), and the
eager path adds it to the attention scores. So the packed form's additive float block-causal mask is honoured by both:
SDPA on CUDA, eager elsewhere (the default). Row and cached passes give a 2D padding mask and let transformers build
the causal mask; passing it (or a cache) also disables the position_ids "packed sequence" detection, which only runs
when both are None.
"""
from pathlib import Path

import torch
from transformers import AutoModel, DynamicCache


def default_attn(device) -> str:
    return "sdpa" if str(device).startswith("cuda") else "eager"


def load_backbone(name_or_path, *, revision=None, dtype=torch.float32, attn=None, device="cpu", local_files_only=False):
    """Qwen3Model from a parent (Hub id resolved through the HF cache) or from a local artifact's backbone/ directory.
    Raises if the checkpoint is not Qwen3 or if any backbone weight was not in the checkpoint (a silent random init)."""
    lm, info = AutoModel.from_pretrained(str(name_or_path), revision=revision, dtype=dtype, attn_implementation=attn or default_attn(device),
                                         local_files_only=local_files_only, output_loading_info=True)
    if lm.config.model_type != "qwen3" or type(lm).__name__ != "Qwen3Model":
        raise ValueError(f"{name_or_path}: expected a Qwen3 decoder, got {type(lm).__name__} ({lm.config.model_type})")
    if info["missing_keys"]:
        raise ValueError(f"{name_or_path}: {len(info['missing_keys'])} backbone weights missing from the checkpoint, e.g. {sorted(info['missing_keys'])[:3]}")
    return lm.to(device)


def save_backbone(lm, out_dir, dtype):
    """Cast the live (merged, plain) Qwen3Model's parameters to `dtype` and write it as an HF directory."""
    cast_parameters(lm, dtype)
    lm.save_pretrained(Path(out_dir))


def cast_parameters(module, dtype):
    """Cast parameters only. nn.Module.to(dtype) would also cast the rotary `inv_freq` buffer, which from_pretrained keeps
    in fp32 at any load dtype; a bf16 inv_freq moves bf16 probabilities by ~2e-2 on identical weights (measured)."""
    for p in module.parameters():
        p.data = p.data.to(dtype)
    return module


def replicate_cache(cache: DynamicCache, n: int, config) -> DynamicCache:
    """A new DynamicCache holding `n` batch copies of a batch-1 prefix cache. DynamicLayer.update concatenates into new
    tensors, so running branches on the replica never writes to `cache` (the caller's prefix stays reusable)."""
    return DynamicCache(ddp_cache_data=[(k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)) for k, v, *_ in cache], config=config)
