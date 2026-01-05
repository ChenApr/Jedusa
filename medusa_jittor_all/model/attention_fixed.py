# -*- coding: utf-8 -*-
"""
Fixed-shape attention execute for LLaMA-style attention in Jittor.

Key idea:
- KV cache is preallocated to max_cache_len (axis=2).
- Each step updates cache via setitem() at dynamic indices (no concat).
- Attention always attends over full max_cache_len, but uses valid+causal masks,
  so shapes stay constant during decode (q_len=1) -> avoids recompilation storms.
"""

from typing import Optional, Tuple, Any
import math
import jittor as jt


def _as_bool_mask(x: jt.Var) -> jt.Var:
    # x: {0/1} or bool
    if str(x.dtype).startswith("bool"):
        return x
    return x != 0


def repeat_kv(x: jt.Var, n_rep: int) -> jt.Var:
    """
    x: [B, n_kv, T, D] -> [B, n_kv*n_rep, T, D]
    """
    if n_rep == 1:
        return x
    B, n_kv, T, D = x.shape
    x = x.unsqueeze(2)                # [B, n_kv, 1, T, D]
    x = x.expand(B, n_kv, n_rep, T, D)
    x = x.reshape(B, n_kv * n_rep, T, D)
    return x


def build_valid_causal_mask(
    max_kv_len: int,
    cache_len: Any,   # [1] int32 (shared) OR int
    q_len: int,
) -> jt.Var:
    """
    Returns additive mask [B=1, 1, q_len, max_kv_len] with 0 for allowed, -1e4 for blocked.
    Uses:
      - valid: k_pos < cache_len
      - causal: k_pos <= q_pos (q_pos are absolute positions in cache)
    """
    # positions
    k_pos = jt.arange(max_kv_len).cast(jt.int32).reshape(1, 1, 1, max_kv_len)  # [1,1,1,K]
    q_pos = (cache_len - q_len + jt.arange(q_len).cast(jt.int32)).reshape(1, 1, q_len, 1)  # [1,1,Q,1]

    if isinstance(cache_len, int):
        valid = k_pos < cache_len
    else:
        valid = k_pos < cache_len.reshape(1, 1, 1, 1)  # [1,1,1,K]

    causal = k_pos <= q_pos                        # [1,1,Q,K]
    allow = jt.logical_and(valid, causal)          # [1,1,Q,K]

    zero = jt.zeros_like(allow).cast(jt.float32)
    neg = (jt.zeros_like(allow).cast(jt.float32) - 1.0) * 1e4
    return jt.where(allow, zero, neg)              # additive mask


def merge_attention_mask(
    base_additive: jt.Var,
    attention_mask: Optional[jt.Var],
    max_kv_len: int,
) -> jt.Var:
    """
    Merge model-provided attention_mask into base_additive.

    Supports common formats:
    - additive mask: [B, 1, Q, K] (float, with -inf/0)
    - key padding mask: [B, S] with 1 keep / 0 pad (we'll map to [B,1,1,K])
    """
    if attention_mask is None:
        return base_additive

    am = attention_mask
    if len(am.shape) == 4:
        # Assume already additive and aligned (maybe K is dynamic; pad/truncate to max_kv_len)
        B, _, Q, K = am.shape
        if K == max_kv_len:
            return base_additive + am
        # pad right if smaller
        if K < max_kv_len:
            pad = jt.zeros((B, 1, Q, max_kv_len - K), dtype=am.dtype)
            am2 = jt.concat([am, pad], dim=3)
            return base_additive + am2
        # truncate if larger
        return base_additive + am[:, :, :, :max_kv_len]

    if len(am.shape) == 2:
        # [B, S] key padding
        B, S = am.shape
        keep = _as_bool_mask(am)
        if S < max_kv_len:
            pad = jt.zeros((B, max_kv_len - S), dtype=keep.dtype)
            keep = jt.concat([keep, pad], dim=1)
        else:
            keep = keep[:, :max_kv_len]
        keep = keep.reshape(B, 1, 1, max_kv_len)  # broadcast over Q
        zero = jt.zeros((B, 1, 1, max_kv_len), dtype=jt.float32)
        neg = (jt.zeros((B, 1, 1, max_kv_len), dtype=jt.float32) - 1.0) * 1e4
        add = jt.where(keep, zero, neg)
        return base_additive + add

    # Unknown format: ignore to avoid crash
    return base_additive


def update_kv_cache_fixed(
    k_cache: jt.Var, v_cache: jt.Var, cache_len: Any,
    k_new: jt.Var, v_new: jt.Var
) -> Tuple[jt.Var, jt.Var, Any]:
    """
    k_cache/v_cache: [B, n_kv, K, D]
    k_new/v_new:     [B, n_kv, Q, D]
    cache_len: [1] int32 (Var) OR int (CPU)
    """
    Q = int(k_new.shape[2])
    
    # ✅ Fast path: if cache_len is int, use slicing + assign (in-place, 0-copy)
    if isinstance(cache_len, int):
        start = cache_len
        end = start + Q
        # Slicing returns a view, assign updates the original tensor
        k_cache[:, :, start:end, :].assign(k_new)
        v_cache[:, :, start:end, :].assign(v_new)
        return k_cache, v_cache, (cache_len + Q)

    # Slow path: cache_len is Var (requires setitem copy)
    idx = (cache_len + jt.arange(Q).cast(jt.int32))  # [Q]
    k_cache = k_cache.setitem((slice(None), slice(None), idx, slice(None)), k_new)
    v_cache = v_cache.setitem((slice(None), slice(None), idx, slice(None)), v_new)
    cache_len = cache_len + Q
    return k_cache, v_cache, cache_len


def fixed_llama_attention_execute(self: Any,
                                 hidden_states: jt.Var,
                                 attention_mask: Optional[jt.Var] = None,
                                 position_ids: Optional[jt.Var] = None,
                                 past_key_value: Optional[Tuple[jt.Var, jt.Var, jt.Var]] = None,
                                 output_attentions: bool = False,
                                 use_cache: bool = False,
                                 **kwargs):
    """
    A drop-in replacement for LlamaAttention.execute/forward in Jittor.

    Expected self attributes:
      q_proj, k_proj, v_proj, o_proj
      num_heads, num_key_value_heads (or num_kv_heads), head_dim
      rotary_emb + apply_rotary_pos_emb (we try to import from origin module if present)
    """
    B, Q, _ = hidden_states.shape

    num_heads = int(getattr(self, "num_heads"))
    n_kv = getattr(self, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = getattr(self, "num_kv_heads", None)
    if n_kv is None:
        n_kv = num_heads
    n_kv = int(n_kv)

    head_dim = int(getattr(self, "head_dim"))
    scale = 1.0 / math.sqrt(head_dim)

    # Projections
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # [B, Q, H*D] -> [B, H, Q, D]
    q = q.reshape(B, Q, num_heads, head_dim).permute(0, 2, 1, 3)
    # [B, Q, Hkv*D] -> [B, Hkv, Q, D]
    k = k.reshape(B, Q, n_kv, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(B, Q, n_kv, head_dim).permute(0, 2, 1, 3)

    # Rotary embedding (best effort: follow common llama ports)
    if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
        # Try to reuse origin apply_rotary_pos_emb if exists (most ports define it in module)
        apply_fn = None
        try:
            mod = __import__(self.__module__, fromlist=["apply_rotary_pos_emb"])
            apply_fn = getattr(mod, "apply_rotary_pos_emb", None)
        except Exception:
            apply_fn = None

        # Build cos/sin
        try:
            # common: cos,sin = rotary_emb(v, seq_len=...)
            # kv_seq_len should be cache_len+Q if cache exists, else Q
            if past_key_value is not None:
                k_cache, _, cache_len = past_key_value
                kv_seq_len = int(k_cache.shape[2])
                cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            else:
                cos, sin = self.rotary_emb(v, seq_len=Q)
        except TypeError:
            # fallback signature
            cos, sin = self.rotary_emb(v)

        if apply_fn is not None:
            # Some apply_rotary_pos_emb expect (q,k,cos,sin,position_ids)
            try:
                q, k = apply_fn(q, k, cos, sin, position_ids)
            except Exception:
                # fallback: ignore rotary if mismatch
                pass

    # Cache update (fixed-shape)
    present = None
    if past_key_value is not None:
        k_cache, v_cache, cache_len = past_key_value
        k_cache, v_cache, cache_len = update_kv_cache_fixed(k_cache, v_cache, cache_len, k, v)

        max_kv_len = int(k_cache.shape[2])
        
        # ✅ Optimization: Determine effective K length for attention
        # If attention_mask is provided and has specific shape [B, 1, Q, K_bucket], use K_bucket.
        # This avoids computing attention over full max_kv_len when we only need bucket_len.
        target_k_len = max_kv_len
        if attention_mask is not None and len(attention_mask.shape) == 4:
             target_k_len = int(attention_mask.shape[-1])
        
        # Expand kv heads to num_heads (GQA)
        n_rep = num_heads // n_kv
        
        # Slice K/V if needed
        if target_k_len < max_kv_len:
             k_use = repeat_kv(k_cache[:, :, :target_k_len, :], n_rep)
             v_use = repeat_kv(v_cache[:, :, :target_k_len, :], n_rep)
        else:
             k_use = repeat_kv(k_cache, n_rep)
             v_use = repeat_kv(v_cache, n_rep)

        # fixed additive mask (using target_k_len)
        base_mask = build_valid_causal_mask(max_kv_len=target_k_len, cache_len=cache_len, q_len=Q)
        
        # ✅ Medusa Mask Injection
        medusa_mask = getattr(self, "medusa_mask", None)
        if medusa_mask is not None:
            # medusa_mask: [1, 1, M, M]
            # Q is current query length (e.g. M)
            M = int(medusa_mask.shape[-1])
            
            # If mask is larger than Q (e.g. padded), slice it
            if M > Q:
                medusa_mask = medusa_mask[:, :, :Q, :Q]
            
            # medusa_mask: [1, 1, Q, Q] with 1=allow, 0=block
            # convert to additive: 0=allow, -1e4=block
            m_add = (medusa_mask - 1.0) * 1e4
            
            # Pad to [1, 1, Q, K]
            # We need to place m_add at indices [start:end] where start = cache_len (before update)
            # If cache_len is int, we know exactly where.
            
            start_idx = 0
            if isinstance(cache_len, int):
                # cache_len here is AFTER update, so start was cache_len - Q
                start_idx = cache_len - Q
            elif isinstance(cache_len, jt.Var):
                 pass

            if isinstance(cache_len, int):
                # Construct mask: [Prefix(0) | Tree(m_add) | Future(-inf)]
                # Prefix len = start_idx
                # Tree len = Q
                # Future len = target_k_len - start_idx - Q
                
                parts = []
                if start_idx > 0:
                    parts.append(jt.zeros((1, 1, Q, start_idx), dtype=m_add.dtype))
                
                parts.append(m_add)
                
                future_len = target_k_len - (start_idx + Q)
                if future_len > 0:
                    parts.append(jt.zeros((1, 1, Q, future_len), dtype=m_add.dtype))
                
                m_full = jt.concat(parts, dim=3)
            else:
                # Old fallback
                if target_k_len > Q:
                    pad_hist = jt.zeros((1, 1, Q, target_k_len - Q), dtype=m_add.dtype)
                    m_full = jt.concat([pad_hist, m_add], dim=3)
                else:
                    m_full = m_add
            
            # Combine with base_mask (intersection of constraints)
            base_mask = base_mask + m_full

        attn_mask = merge_attention_mask(base_mask, attention_mask, max_kv_len=target_k_len)
        present = (k_cache, v_cache, cache_len)
    else:
        # No cache mode (training/debug). Use dynamic K=Q.
        target_k_len = int(k.shape[2])
        n_rep = num_heads // n_kv
        k_use = repeat_kv(k, n_rep)
        v_use = repeat_kv(v, n_rep)
        # causal within Q
        cache_len = jt.array([Q]).cast(jt.int32)
        base_mask = build_valid_causal_mask(max_kv_len=target_k_len, cache_len=cache_len, q_len=Q)
        attn_mask = merge_attention_mask(base_mask, attention_mask, max_kv_len=target_k_len)

    # Attention: [B,H,Q,D] x [B,H,D,K] -> [B,H,Q,K]
    attn_weights = jt.matmul(q, k_use.transpose(0, 1, 3, 2)) * scale
    attn_weights = attn_weights + attn_mask.cast(attn_weights.dtype)

    attn_probs = jt.nn.softmax(attn_weights, dim=-1)
    attn_out = jt.matmul(attn_probs, v_use)  # [B,H,Q,D]

    # [B,H,Q,D] -> [B,Q,H*D]
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, Q, num_heads * head_dim)
    attn_out = self.o_proj(attn_out)

    if output_attentions:
        return attn_out, attn_probs, present if use_cache else None
    return attn_out, present if use_cache else None