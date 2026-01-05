# -*- coding: utf-8 -*-
"""
Wrapper module:
- Re-export everything from modeling_llama_jt_orig
- Monkeypatch LlamaAttention.execute/forward to fixed-shape KV cache attention
"""

from .modeling_llama_jt_orig import *  # noqa: F401,F403

try:
    # Import patched execute
    from .attention_fixed import fixed_llama_attention_execute

    # Grab LlamaAttention class from orig exports
    _LlamaAttention = globals().get("LlamaAttention", None)
    if _LlamaAttention is None:
        # Some repos name it differently
        for _k in list(globals().keys()):
            if _k.lower() == "llamaattention" or _k.endswith("LlamaAttention"):
                _LlamaAttention = globals()[_k]
                break

    if _LlamaAttention is None:
        raise RuntimeError("Cannot find LlamaAttention in modeling_llama_jt_orig exports.")

    # Patch both execute and forward if present
    if hasattr(_LlamaAttention, "execute"):
        _LlamaAttention.execute = fixed_llama_attention_execute
    if hasattr(_LlamaAttention, "forward"):
        _LlamaAttention.forward = fixed_llama_attention_execute

    print("✅ [fixed-attn] Patched LlamaAttention (fixed MAX KV cache + valid/causal mask).", flush=True)
except Exception as _e:
    print(f"[WARN] [fixed-attn] patch failed: {_e}", flush=True)