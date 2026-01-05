import jittor as jt
from jittor import nn
import numpy as np

TOPK = 5


def _use_cuda() -> bool:
    try:
        return bool(getattr(jt.flags, "use_cuda", 0))
    except Exception:
        return False


def _to_cuda(x: jt.Var) -> jt.Var:
    if x is None:
        return None
    if not isinstance(x, jt.Var):
        return x
    if not _use_cuda():
        return x
    try:
        return x.cuda()
    except Exception:
        return x


def _is_int_dtype(dtype) -> bool:
    return dtype in (jt.int8, jt.int16, jt.int32, jt.int64, jt.uint8)


def argmax_indices(x: jt.Var, dim: int = -1) -> jt.Var:
    out = jt.argmax(x, dim=dim)
    if isinstance(out, tuple) and len(out) == 2:
        a, b = out
        try:
            a_is_int = _is_int_dtype(a.dtype)
            b_is_int = _is_int_dtype(b.dtype)
            if a_is_int and not b_is_int:
                return a
            if b_is_int and not a_is_int:
                return b
        except Exception:
            pass
        return a
    return out


def _ensure_int(x):
    if isinstance(x, jt.Var):
        try:
            return int(x.item())
        except Exception:
            return int(x.numpy().reshape(-1)[0])
    return int(x)


def pad_path(path, length, pad_value=-2):
    return path + [pad_value] * (length - len(path))


def generate_medusa_buffers(medusa_choices, device="cuda", mask_dtype=jt.float16):
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    medusa_attn_mask_np = np.eye(medusa_len, dtype=np.float16)
    medusa_attn_mask_np[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur = sorted_medusa_choices[start + j]
            if len(cur) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur[: c + 1]) + 1)
            medusa_attn_mask_np[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    medusa_attn_mask = jt.array(medusa_attn_mask_np).cast(mask_dtype)

    medusa_tree_indices_np = np.zeros(medusa_len, dtype=np.int32)
    medusa_tree_indices_np[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur = sorted_medusa_choices[start + j]
            medusa_tree_indices_np[start + j + 1] = int(cur[-1] + TOPK * i + 1)
        start += depth_counts[i]
    medusa_tree_indices = jt.array(medusa_tree_indices_np).cast(jt.int32)

    medusa_position_ids_np = np.zeros(medusa_len, dtype=np.int32)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids_np[start + 1 : start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]
    medusa_position_ids = jt.array(medusa_position_ids_np).cast(jt.int32)

    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur = sorted_medusa_choices[-i - 1]
        if cur in retrieve_paths:
            continue
        idxs = []
        for c in range(len(cur)):
            idxs.append(sorted_medusa_choices.index(cur[: c + 1]))
            retrieve_paths.append(cur[: c + 1])
        retrieve_indices_nest.append(idxs)

    max_length = max(len(x) for x in retrieve_indices_nest)
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices_np = np.array(retrieve_indices, dtype=np.int32)
    retrieve_indices_np = retrieve_indices_np + 1
    retrieve_indices_np = np.concatenate(
        [np.zeros((retrieve_indices_np.shape[0], 1), dtype=np.int32), retrieve_indices_np],
        axis=1,
    )
    retrieve_indices = jt.array(retrieve_indices_np).cast(jt.int32)

    # ✅ ensure on cuda
    if device == "cuda" and _use_cuda():
        medusa_attn_mask = _to_cuda(medusa_attn_mask)
        medusa_tree_indices = _to_cuda(medusa_tree_indices)
        medusa_position_ids = _to_cuda(medusa_position_ids)
        retrieve_indices = _to_cuda(retrieve_indices)

    return {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),  # [1,1,L,L]
        "tree_indices": medusa_tree_indices,                             # [L]
        "medusa_position_ids": medusa_position_ids,                      # [L]
        "retrieve_indices": retrieve_indices,                            # [P,K]
    }


def reset_medusa_mode(model):
    model.base_model.model.medusa_mask = None
    model.base_model.model.medusa_mode = None


def initialize_medusa(input_ids, model, medusa_attn_mask, past_key_values):
    # ✅ Prefill phase: use standard causal attention (no tree mask)
    model.base_model.model.medusa_mask = None
    medusa_logits, outputs, logits = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        output_orig=True,
        medusa_forward=True,
    )
    return medusa_logits, logits


def generate_candidates(
    medusa_logits,
    logits,
    tree_indices,
    retrieve_indices,
    temperature=0.0,
    posterior_threshold=0.3,
    posterior_alpha=0.09,
    top_p=0.8,
    sampling="typical",
    fast=False,
):
    # base next token
    if temperature == 0 or fast:
        base_next = argmax_indices(logits[:, -1], dim=-1).cast(jt.int32)
    else:
        # for stability, prefer greedy in jittor path (avoid numpy sampling sync)
        base_next = argmax_indices(logits[:, -1], dim=-1).cast(jt.int32)

    topk_idx = jt.topk(medusa_logits[:, 0, -1], TOPK, dim=-1)[1].cast(jt.int32)  # [H,K]
    candidates = jt.concat([base_next.reshape((-1,)), topk_idx.view(-1)], dim=0).cast(jt.int32)
    tree_candidates = candidates[tree_indices]
    tree_candidates_ext = jt.concat([tree_candidates, jt.zeros((1,), dtype=jt.int32)], dim=0)
    cart_candidates = tree_candidates_ext[retrieve_indices]
    return cart_candidates.cast(jt.int32), tree_candidates.unsqueeze(0).cast(jt.int32)


def prepare_medusa_attention_mask(medusa_mask, cache_len: int, max_kv_len: int, q_len: int):
    """
    Pre-compute the additive attention mask for Medusa tree decoding.
    medusa_mask: [1, 1, Q, Q] (1=allow, 0=block)
    cache_len: int (current length before update, i.e. start index of tree)
    max_kv_len: int (capacity)
    q_len: int (Q)
    
    Returns: [1, 1, Q, max_kv_len] additive mask (0=allow, -1e4=block)
    """
    # 1. Convert to additive
    m_add = (medusa_mask - 1.0) * 1e4
    
    # 2. Construct parts
    # [Prefix (0) | Tree (m_add) | Future (0)]
    
    start_idx = cache_len
    parts = []
    
    if start_idx > 0:
        parts.append(jt.zeros((1, 1, q_len, start_idx), dtype=m_add.dtype))
        
    parts.append(m_add)
    
    future_len = max_kv_len - (start_idx + q_len)
    if future_len > 0:
        parts.append(jt.zeros((1, 1, q_len, future_len), dtype=m_add.dtype))
        
    return jt.concat(parts, dim=3)


def tree_decoding(model, tree_candidates, past_key_values, medusa_position_ids, input_ids, retrieve_indices, current_len_cpu=None, attention_mask=None):
    pos = (medusa_position_ids + int(input_ids.shape[1])).cast(jt.int32)
    if len(pos.shape) == 1:
        pos = pos.unsqueeze(0)

    # ✅ Construct temporary PKV list with int length if available
    pkv_arg = past_key_values
    if current_len_cpu is not None:
        pkv_arg = []
        for k, v, _ in past_key_values:
            pkv_arg.append((k, v, int(current_len_cpu)))

    tree_medusa_logits, outputs, tree_logits = model(
        input_ids=tree_candidates.cast(jt.int32),
        output_orig=True,
        past_key_values=pkv_arg,
        position_ids=pos,
        medusa_forward=True,
        attention_mask=attention_mask,
    )

    logits = tree_logits[0, retrieve_indices]                # [P,K,V]
    medusa_logits = tree_medusa_logits[:, 0, retrieve_indices]  # [H,P,K,V]
    return medusa_logits, logits, outputs


def evaluate_posterior(
    logits,
    candidates,
    temperature,
    posterior_threshold=0.3,
    posterior_alpha=0.09,
    top_p=0.8,
    sampling="typical",
    fast=True,
):
    # keep greedy accept (stable + fast)
    pred = argmax_indices(logits[:, :-1], dim=-1).cast(jt.int32)  # [P,K-1]
    tgt = candidates[:, 1:].cast(jt.int32)                        # [P,K-1]
    match = (tgt == pred).cast(jt.int32)
    prefix = jt.cumprod(match, dim=1).sum(dim=1)                  # [P]

    accept_len = prefix.max()
    accept_len_int = _ensure_int(accept_len)

    if accept_len_int == 0:
        best = jt.array(0).cast(jt.int32)
    else:
        best = argmax_indices(prefix, dim=0).cast(jt.int32)

    return best, accept_len_int


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length: int,
    retrieve_indices,
    outputs,
    logits,
    medusa_logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Fast KV compaction:
      - prefer packed tensor: [2L, B, KVH, MAX, HD]
      - do fixed-length (Amax) gather+assign to avoid shape-specialization each step
    """
    prev_len = int(input_ids.shape[1])
    new_len = prev_len + accept_length + 1

    # append accepted tokens
    add_tokens = candidates[best_candidate, : accept_length + 1].cast(jt.int32)  # [a+1]
    if len(add_tokens.shape) == 2:
        add_tokens = add_tokens.reshape((-1,))
    input_ids = jt.concat([input_ids, add_tokens.unsqueeze(0)], dim=1).cast(jt.int32)

    # Amax is FIXED by retrieve_indices width
    Amax = int(retrieve_indices.shape[1])

    # sel_full: [Amax] absolute positions in tree-cache segment
    sel_full = retrieve_indices[best_candidate, :Amax].cast(jt.int32) + prev_len

    # pad positions might be negative -> clamp to prev_len (safe junk copy)
    sel_full = jt.maximum(sel_full, jt.array(prev_len).cast(jt.int32))

    # ---- get max_cache and do KV copy ----
    # ✅ Fix: Use `outputs` (fresh KV from TreeFwd) as source, because `past_key_values_data` is stale.
    # outputs[1] is past_key_values list[(k,v,len)]
    new_pkv_list = outputs[1]
    
    if isinstance(past_key_values_data, dict):
        # slow path (still works)
        if "max_cache_len" in past_key_values_data:
            max_cache = int(past_key_values_data["max_cache_len"])
        else:
            max_cache = int(past_key_values_data["k"][0].shape[2])
        k_list = past_key_values_data["k"]
        v_list = past_key_values_data["v"]
        end_full = prev_len + Amax
        if end_full > max_cache:
            raise RuntimeError(f"KV cache overflow: prev_len+Amax={end_full} > max_cache={max_cache}.")
        
        for i, (k_cache, v_cache) in enumerate(zip(k_list, v_list)):
            # Source from new_pkv_list
            k_new, v_new, _ = new_pkv_list[i]
            k_cache[:, :, prev_len:end_full, :].assign(k_new[:, :, sel_full, :])
            v_cache[:, :, prev_len:end_full, :].assign(v_new[:, :, sel_full, :])
    else:
        # fast packed path: [2L,B,KVH,MAX,HD]
        max_cache = int(past_key_values_data.shape[3])
        end_full = prev_len + Amax
        if end_full > max_cache:
            raise RuntimeError(f"KV cache overflow: prev_len+Amax={end_full} > max_cache={max_cache}.")
        
        # Loop over layers to update packed tensor from detached new_pkv_list
        for i in range(len(new_pkv_list)):
            k_new, v_new, _ = new_pkv_list[i]
            # pkv_data[2*i] is K, [2*i+1] is V
            past_key_values_data[2*i, :, :, prev_len:end_full, :].assign(k_new[:, :, sel_full, :])
            past_key_values_data[2*i+1, :, :, prev_len:end_full, :].assign(v_new[:, :, sel_full, :])

    # rollback shared length to *new_len* (only accepted tokens become valid)
    # keep shape [1]
    current_length_data.assign(jt.array([new_len]).cast(current_length_data.dtype))

    logits_next = logits[best_candidate].unsqueeze(0)[:, : accept_length + 1, :]  # [1,a+1,V]
    medusa_logits_next = medusa_logits[:, best_candidate].unsqueeze(1)[:, :, : accept_length + 1, :]  # [H,1,a+1,V]

    new_token += accept_length + 1
    return input_ids, logits_next, medusa_logits_next, new_token