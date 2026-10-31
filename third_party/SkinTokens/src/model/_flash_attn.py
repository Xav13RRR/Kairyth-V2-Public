"""flash_attn helper with PyTorch SDPA fallback (Kimodo / Windows)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func  # type: ignore

    def flash_attn_func(q, k, v, **kwargs):
        res = _flash_attn_func(q, k, v, **kwargs)
        if isinstance(res, tuple):
            return res
        return res, None

except Exception:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # type: ignore

        def flash_attn_func(q, k, v, **kwargs):
            res = _flash_attn_func(q, k, v, **kwargs)
            if isinstance(res, tuple):
                return res
            return res, None

    except Exception:

        def flash_attn_func(q, k, v, **kwargs):
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            if q.shape[1] != k.shape[1]:
                repeat_factor = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            out = F.scaled_dot_product_attention(q, k, v)
            return out.permute(0, 2, 1, 3), None
