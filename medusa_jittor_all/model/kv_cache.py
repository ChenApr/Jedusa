# -*- coding: utf-8 -*-
"""
Packed fixed-shape KV cache for Jittor.

Returns:
  past_key_values: list[(k_var, v_var, len_var)] length = num_layers
  pkv_data: jt.Var shape [2*L, B, KVH, MAX, HD]   (packed)
  len_var : jt.Var int32 shape [1] (shared length)

Robust inference order:
  1) Prefer model.config (num_hidden_layers / num_key_value_heads / head_dim)
  2) Fallback to locating layers + attention module by scanning common paths / BFS

This avoids errors when `model` is a wrapper (e.g., MedusaModel / LlamaForCausalLM).
"""

from typing import Any, List, Tuple, Optional
import jittor as jt


# ----------------- helpers -----------------

def _get_by_path(obj, path: str):
    cur = obj
    for seg in path.split("."):
        if not hasattr(cur, seg):
            return None
        cur = getattr(cur, seg)
    return cur


def _is_layer_list(x) -> bool:
    # Accept list/tuple/ModuleList-like containers
    if x is None:
        return False
    if isinstance(x, (list, tuple)):
        return len(x) > 0
    # Jittor nn.ModuleList is usually list-like but to be safe:
    if hasattr(x, "__len__") and hasattr(x, "__getitem__"):
        try:
            return len(x) > 0
        except Exception:
            return False
    return False


def _looks_like_decoder_layer(layer) -> bool:
    if layer is None:
        return False
    # typical names
    return any(hasattr(layer, n) for n in ("self_attn", "attn", "attention"))


def _locate_layers(model) -> Optional[Any]:
    # 1) common known paths
    candidates = [
        "layers",
        "model.layers",
        "base_model.model.layers",
        "model.model.layers",
        "base_model.layers",
        "base_model.model.model.layers",
        "transformer.layers",
        "transformer.h",          # some repos use transformer.h
        "model.decoder.layers",
        "decoder.layers",
    ]
    for p in candidates:
        obj = _get_by_path(model, p)
        if _is_layer_list(obj) and _looks_like_decoder_layer(obj[0]):
            return obj

    # 2) BFS scan through attributes (limited depth)
    #    Try to find a container named "layers"/"h" or a container whose elements look like decoder layers.
    queue = [(model, 0)]
    seen = set()
    while queue:
        obj, d = queue.pop(0)
        if obj is None or d > 4:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        # quick check: has attr "layers"
        if hasattr(obj, "layers"):
            cand = getattr(obj, "layers")
            if _is_layer_list(cand) and _looks_like_decoder_layer(cand[0]):
                return cand

        # scan attributes
        try:
            names = list(vars(obj).keys())
        except Exception:
            names = [n for n in dir(obj) if not n.startswith("_")]

        for name in names:
            if name.startswith("_"):
                continue
            try:
                val = getattr(obj, name)
            except Exception:
                continue

            # container candidate
            if _is_layer_list(val):
                try:
                    if _looks_like_decoder_layer(val[0]):
                        return val
                except Exception:
                    pass

            # descend into nested modules/wrappers
            if d < 4 and (hasattr(val, "__dict__") or hasattr(val, "modules")):
                # filter out jittor Var / tensors
                if isinstance(val, jt.Var):
                    continue
                queue.append((val, d + 1))

    return None


def _infer_from_config(model) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None, None, None

    num_layers = None
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        v = getattr(cfg, k, None)
        if v is not None:
            num_layers = int(v)
            break

    # KV heads
    n_kv = getattr(cfg, "num_key_value_heads", None)
    if n_kv is None:
        # if no GQA
        n_kv = getattr(cfg, "num_attention_heads", None)

    # head_dim
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        hs = getattr(cfg, "hidden_size", None)
        nh = getattr(cfg, "num_attention_heads", None)
        if hs is not None and nh is not None:
            head_dim = int(hs) // int(nh)

    return num_layers, (int(n_kv) if n_kv is not None else None), (int(head_dim) if head_dim is not None else None)


def _infer_layers(model) -> int:
    num_layers, _, _ = _infer_from_config(model)
    if num_layers is not None:
        return int(num_layers)

    layers = _locate_layers(model)
    if layers is not None:
        return int(len(layers))

    raise RuntimeError(
        "Cannot infer num_layers. "
        "Tried model.config.{num_hidden_layers|n_layer|num_layers} and scanning for *.layers."
    )


def _infer_kv_heads_and_dim(model) -> Tuple[int, int]:
    # Prefer config (works even if model is wrapper)
    _, n_kv, head_dim = _infer_from_config(model)
    if n_kv is not None and head_dim is not None:
        return int(n_kv), int(head_dim)

    # Fallback: locate layers -> layer0.self_attn -> read fields
    layers = _locate_layers(model)
    if layers is None:
        raise RuntimeError("Cannot locate model layers to infer KV heads/dim (and config missing required fields).")

    layer0 = layers[0]
    attn = None
    for name in ("self_attn", "attn", "attention"):
        if hasattr(layer0, name):
            attn = getattr(layer0, name)
            break
    if attn is None:
        raise RuntimeError("Cannot locate attention module (self_attn/attn/attention) on layer0.")

    n_kv = getattr(attn, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = getattr(attn, "num_kv_heads", None)
    if n_kv is None:
        n_kv = getattr(attn, "num_heads", None)
    if n_kv is None:
        raise RuntimeError("Cannot infer num_key_value_heads/num_heads from attention (and config missing).")

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        hs = getattr(attn, "hidden_size", None)
        nh = getattr(attn, "num_heads", None)
        if hs is not None and nh is not None:
            head_dim = int(hs) // int(nh)
    if head_dim is None:
        raise RuntimeError("Cannot infer head_dim from attention (and config missing).")

    return int(n_kv), int(head_dim)


# ----------------- public API -----------------

def initialize_past_key_values(
    model: Any,
    max_cache_len: int,
    dtype: Any = jt.float16,
    batch_size: int = 1,
):
    """
    Returns:
      past_key_values: list[(k_cache, v_cache, len_var)] length = num_layers
      pkv_data: packed tensor [2*L, B, KVH, MAX, HD]
      len_var: shared length jt.Var int32 [1]
    """
    num_layers = _infer_layers(model)
    n_kv, head_dim = _infer_kv_heads_and_dim(model)

    len_var = jt.zeros((1,), dtype=jt.int32)  # shared length

    # packed: [2L, B, KVH, MAX, HD]
    pkv = jt.zeros((2 * num_layers, batch_size, n_kv, int(max_cache_len), head_dim), dtype=dtype)
    
    # Check device
    try:
        if jt.flags.use_cuda and not pkv.is_cuda:
             print(f"[KV Cache] Warning: pkv allocated on CPU! Moving to CUDA...", flush=True)
             pkv = pkv.cuda()
        else:
             print(f"[KV Cache] Allocated on {'CUDA' if pkv.is_cuda else 'CPU'}", flush=True)
    except Exception:
        pass

    past_key_values: List[Tuple[jt.Var, jt.Var, jt.Var]] = []
    for i in range(num_layers):
        k = pkv[2 * i + 0]
        v = pkv[2 * i + 1]
        past_key_values.append((k, v, len_var))

    return past_key_values, pkv, len_var


def reset_current_length_data(cur_len_data):
    if cur_len_data is None:
        return
    if isinstance(cur_len_data, jt.Var):
        try:
            cur_len_data.assign(jt.zeros(cur_len_data.shape, dtype=cur_len_data.dtype))
        except Exception:
            pass
        return
    if isinstance(cur_len_data, (list, tuple)):
        for x in cur_len_data:
            reset_current_length_data(x)