# medusa_jittor_all/model/modeling_llama_jt.py
# FULL PATCH: fixed max_cache KV + internal stable-shape masks in attention
# + KV tuple/legacy compat (fix: no .current_length on jt.Var)

import math
from typing import Optional, Tuple
import jittor as jt

# ===================== KV COMPAT PATCH =====================
# Accept both:
#   tuple KV: past_key_values[layer] = (k_var, v_var, len_var)
#   legacy:   past_key_values[layer][0].current_length (cache object with .cat/.data)
from medusa_jittor_all.model.kv_compat import pkv_len_var as _pkv_len_var
# =================== END KV COMPAT PATCH ====================

from jittor import nn
from transformers import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


def linear_matmul(linear, x):
    w = getattr(linear, "weight", None)
    b = getattr(linear, "bias", None)
    if w is None:
        return linear(x)

    if hasattr(x, "dtype") and hasattr(w, "dtype") and x.dtype != w.dtype:
        x = x.astype(w.dtype)

    if len(x.shape) == 3:
        B, S, H = x.shape
        y2 = jt.matmul(x.reshape((-1, H)), w.transpose())
        y = y2.reshape((B, S, y2.shape[-1]))
    else:
        y = jt.matmul(x, w.transpose())

    if b is not None:
        y = y + b
    return y


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = jt.ones(hidden_size)
        self.variance_epsilon = eps

    def execute(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float32()
        variance = hidden_states.sqr().mean(-1, keepdims=True)
        hidden_states = hidden_states * jt.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (jt.arange(0, self.dim, 2).float() / self.dim))
        self.inv_freq = inv_freq.stop_grad()
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=jt.float32)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = int(seq_len)
        t = jt.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = jt.einsum("i,j->ij", t, self.inv_freq)
        emb = jt.concat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].astype(dtype).stop_grad()
        self.sin_cached = emb.sin()[None, None, :, :].astype(dtype).stop_grad()

    def execute(self, x, seq_len=None):
        seq_len = int(seq_len)
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].astype(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].astype(x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jt.concat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(0).squeeze(0)  # [seq, dim]
    sin = sin.squeeze(0).squeeze(0)

    cos = cos[position_ids].unsqueeze(1)  # [B,1,q,dim]
    sin = sin[position_ids].unsqueeze(1)

    if cos.dtype != q.dtype:
        cos = cos.astype(q.dtype)
    if sin.dtype != q.dtype:
        sin = sin.astype(q.dtype)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def execute(self, x):
        gate = linear_matmul(self.gate_proj, x)
        up = linear_matmul(self.up_proj, x)
        intermediate = self.act_fn(gate) * up
        down = linear_matmul(self.down_proj, intermediate)
        return down


def repeat_kv(hidden_states: jt.Var, n_rep: int) -> jt.Var:
    B, KVH, K, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(B, KVH, n_rep, K, D)
    return hidden_states.reshape(B, KVH * n_rep, K, D)


def _as_scalar(v):
    """
    Convert v (jt.Var / python scalar / 1-element tensor) into a scalar-like jt.Var.
    Jittor does NOT allow reshape(()).
    """
    if isinstance(v, jt.Var):
        x = v
    else:
        x = jt.array(v)

    try:
        if len(x.shape) == 0:
            return x
    except Exception:
        pass

    try:
        x1 = x.reshape((-1,))
        return x1[0]
    except Exception:
        try:
            xs = jt.squeeze(x)
            if len(xs.shape) == 0:
                return xs
            xs1 = xs.reshape((-1,))
            return xs1[0]
        except Exception:
            return x


def _clamp_int(x: jt.Var, lo: int, hi: int) -> jt.Var:
    lo_v = jt.array(lo).cast(jt.int32)
    hi_v = jt.array(hi).cast(jt.int32)
    return jt.minimum(jt.maximum(x, lo_v), hi_v)


def _is_tuple_kv(past_key_value):
    # tuple KV: (k_var, v_var, len_var)
    if past_key_value is None:
        return False
    if not isinstance(past_key_value, (tuple, list)) or len(past_key_value) != 3:
        return False
    k, v, ln = past_key_value
    return isinstance(k, jt.Var) and isinstance(v, jt.Var) and isinstance(ln, jt.Var)


def _is_legacy_cache_obj(past_key_value):
    # legacy cache object: past_key_value[0].cat + .current_length + .data
    if past_key_value is None:
        return False
    try:
        return (
            isinstance(past_key_value, (tuple, list))
            and len(past_key_value) >= 2
            and hasattr(past_key_value[0], "cat")
            and hasattr(past_key_value[0], "current_length")
            and hasattr(past_key_value[0], "data")
        )
    except Exception:
        return False


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # injected per-call (medusa)
        self.medusa_mask = None  # expected [1,1,q,q] or None

    def _build_cache_mask(
        self,
        position_ids: jt.Var,   # [B,q]
        start_pos: jt.Var,      # scalar (past_len before append)
        total_len: jt.Var,      # scalar (past_len after append)
        max_cache: int,
        q_len: int,
        bsz: int,
    ) -> jt.Var:
        """
        Return float32 mask [B,1,q,max_cache], 0 allowed, -1e9 disallowed.
        Shape stable: key_len == max_cache always.
        """
        total_len = _as_scalar(total_len).cast(jt.int32)
        start_pos = _as_scalar(start_pos).cast(jt.int32)

        key_pos = jt.arange(max_cache).cast(jt.int32)  # [K]

        # Medusa intra-block graph (prefix always visible)
        if self.medusa_mask is not None and int(self.medusa_mask.shape[-1]) == int(q_len):
            local_k = key_pos - start_pos               # [K]
            in_block = (local_k >= 0) & (local_k < jt.array(q_len).cast(jt.int32))  # [K]
            local_k_clip = _clamp_int(local_k, 0, q_len - 1)  # [K] for gather

            medusa_allow = (self.medusa_mask[0, 0] > 0.0)      # [q,q] bool
            cols_allow = medusa_allow[:, local_k_clip]         # [q,K] bool

            prefix_allow = (local_k < 0).unsqueeze(0)          # [1,K]
            block_allow = cols_allow & in_block.unsqueeze(0)   # [q,K]

            allow = prefix_allow | block_allow                 # [q,K]
            valid = (key_pos < total_len).unsqueeze(0)         # [1,K]
            allow = allow & valid

            mask2d = (1.0 - allow.cast(jt.float32)) * (-1e9)   # [q,K]
            mask = mask2d.unsqueeze(0).unsqueeze(1)            # [1,1,q,K]
            if bsz > 1:
                mask = mask + jt.zeros((bsz, 1, 1, 1), dtype=mask.dtype)
            return mask

        # Normal causal+valid
        query_pos = position_ids.cast(jt.int32).reshape((bsz, 1, q_len, 1))  # [B,1,q,1]
        key_pos4 = key_pos.reshape((1, 1, 1, max_cache))                     # [1,1,1,K]
        valid = key_pos4 < total_len
        causal = key_pos4 <= query_pos
        allow = valid & causal
        return (1.0 - allow.cast(jt.float32)) * (-1e9)

    def _build_nocache_mask(self, position_ids: jt.Var, q_len: int, bsz: int) -> jt.Var:
        key_pos = jt.arange(q_len).cast(jt.int32).reshape((1, 1, 1, q_len))
        query_pos = position_ids.cast(jt.int32).reshape((bsz, 1, q_len, 1))
        allow = key_pos <= query_pos
        return (1.0 - allow.cast(jt.float32)) * (-1e9)

    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,  # ignored in cache mode
        position_ids: Optional[jt.Var] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.shape

        # Proj
        query_states = linear_matmul(self.q_proj, hidden_states)
        key_states = linear_matmul(self.k_proj, hidden_states)
        value_states = linear_matmul(self.v_proj, hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ----------------------------
        # cache mode: tuple KV (fixed shape) OR legacy cache object
        # ----------------------------
        if _is_tuple_kv(past_key_value):
            k_cache, v_cache, len_var = past_key_value
            max_cache = int(k_cache.shape[2])

            # Rotary: use fixed seq_len=max_cache for stable indexing
            cos, sin = self.rotary_emb(value_states, seq_len=max_cache)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # read current length (host sync once per layer; ok for correctness)
            try:
                before_i = int(_as_scalar(len_var).numpy())
            except Exception:
                before_i = int(_as_scalar(len_var).reshape((-1,)).numpy()[0])

            end_i = before_i + int(q_len)
            if end_i > max_cache:
                # clamp (avoid OOB). This should not happen if your max_ctx budgeting is correct.
                end_i = max_cache
            write_len = end_i - before_i
            if write_len > 0:
                # slice-write into fixed cache
                k_cache[:, :, before_i:end_i, :].assign(key_states[:, :, :write_len, :])
                v_cache[:, :, before_i:end_i, :].assign(value_states[:, :, :write_len, :])

            # update shared length var
            try:
                len_var.assign(jt.array([end_i]).cast(jt.int32))
            except Exception:
                len_var.assign(jt.array(end_i).cast(jt.int32).reshape((1,)))

            before = jt.array(before_i).cast(jt.int32)
            after = jt.array(end_i).cast(jt.int32)

            key_full = repeat_kv(k_cache, self.num_key_value_groups)   # [B,H,K,D]
            val_full = repeat_kv(v_cache, self.num_key_value_groups)

            attn_weights = jt.matmul(query_states, key_full.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = attn_weights.float32()

            if attention_mask is not None:
                # attention_mask 期望形状 [B,1,q,max_cache] 或 [1,1,q,max_cache]
                mask = attention_mask
                if int(mask.shape[0]) == 1 and bsz > 1:
                    mask = mask + jt.zeros((bsz,1,1,1), dtype=mask.dtype)
                attn_weights = attn_weights + mask.astype(attn_weights.dtype)
            else:
                mask = self._build_cache_mask(...)
                attn_weights = attn_weights + mask

            attn_probs = nn.softmax(attn_weights, dim=-1).astype(query_states.dtype)
            attn_output = jt.matmul(attn_probs, val_full)

            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
            attn_output = linear_matmul(self.o_proj, attn_output)

            if not output_attentions:
                attn_probs = None
            present = past_key_value if use_cache else None
            return attn_output, attn_probs, present

        if _is_legacy_cache_obj(past_key_value):
            k_cache = past_key_value[0]
            v_cache = past_key_value[1]
            max_cache = int(k_cache.data.shape[2])  # fixed K

            cos, sin = self.rotary_emb(value_states, seq_len=max_cache)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            before = _as_scalar(k_cache.current_length).cast(jt.int32)
            k_cache.cat(key_states, dim=2)
            v_cache.cat(value_states, dim=2)
            after = _as_scalar(k_cache.current_length).cast(jt.int32)

            key_full = repeat_kv(k_cache.data, self.num_key_value_groups)  # [B,H,K,D]
            val_full = repeat_kv(v_cache.data, self.num_key_value_groups)

            attn_weights = jt.matmul(query_states, key_full.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = attn_weights.float32()

            mask = self._build_cache_mask(
                position_ids=position_ids,
                start_pos=before,
                total_len=after,
                max_cache=max_cache,
                q_len=q_len,
                bsz=bsz,
            )
            attn_weights = attn_weights + mask

            attn_probs = nn.softmax(attn_weights, dim=-1).astype(query_states.dtype)
            attn_output = jt.matmul(attn_probs, val_full)

            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
            attn_output = linear_matmul(self.o_proj, attn_output)

            if not output_attentions:
                attn_probs = None
            present = past_key_value if use_cache else None
            return attn_output, attn_probs, present

        # ----------------------------
        # no-cache prefill
        # ----------------------------
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = jt.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.float32()

        mask = self._build_nocache_mask(position_ids=position_ids, q_len=q_len, bsz=bsz)
        attn_weights = attn_weights + mask

        attn_probs = nn.softmax(attn_weights, dim=-1).astype(query_states.dtype)
        attn_output = jt.matmul(attn_probs, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = linear_matmul(self.o_proj, attn_output)

        if not output_attentions:
            attn_probs = None
        present = (key_states, value_states) if use_cache else None
        return attn_output, attn_probs, present


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # ---- compat: attention may return 2 or 3 items ----
        # Common 2-item pattern (most likely your patched version):
        #   output_attentions=False: (attn_output, present)
        #   output_attentions=True : (attn_output, attn_weights)
        if not isinstance(attn_out, (tuple, list)):
            raise TypeError(f"self_attn returned non-tuple: {type(attn_out)}")

        if len(attn_out) == 3:
            hidden_states, attn_weights, present = attn_out
        elif len(attn_out) == 2:
            hidden_states = attn_out[0]
            if output_attentions:
                attn_weights = attn_out[1]
                present = past_key_value if use_cache else None
            else:
                attn_weights = None
                present = attn_out[1] if use_cache else None
        else:
            raise ValueError(f"self_attn returned {len(attn_out)} items, expected 2 or 3")
        # ---- end compat ----
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present,)
        return outputs


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # injected by medusa_model
        self.medusa_mask = None

    def execute(
        self,
        input_ids: jt.Var = None,
        attention_mask: Optional[jt.Var] = None,  # ignored
        position_ids: Optional[jt.Var] = None,
        past_key_values=None,
        inputs_embeds: Optional[jt.Var] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # ✅ Accept extra args like output_orig
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length, _ = inputs_embeds.shape

        past_len = jt.array(0).cast(jt.int32)
        if past_key_values is not None:
            past_len = _as_scalar(_pkv_len_var(past_key_values)).cast(jt.int32)

        if position_ids is None:
            position_ids = (jt.arange(seq_length).cast(jt.int32) + past_len).unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).cast(jt.int32)

        if inputs_embeds is None:
            if input_ids.dtype != jt.int32:
                input_ids = input_ids.cast(jt.int32)
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # propagate medusa mask into each attention layer
        m_mask = getattr(self, "medusa_mask", None)
        for layer in self.layers:
            layer.self_attn.medusa_mask = m_mask

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            pkv = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def execute(
        self,
        input_ids: jt.Var = None,
        attention_mask: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        past_key_values=None,
        inputs_embeds: Optional[jt.Var] = None,
        labels: Optional[jt.Var] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )