# -*- coding: utf-8 -*-
import jittor as jt

def pkv_len_var(past_key_values):
    """
    Return shared length jt.Var(int32,[1]) from:
      - list[layer] of (k,v,len)
      - list[layer] of objects with .current_length
      - single layer tuple (k,v,len)
      - single cache object with .current_length
    """
    z = jt.zeros((1,), dtype=jt.int32)
    if past_key_values is None:
        return z

    # list-of-layers
    if isinstance(past_key_values, (list, tuple)):
        if len(past_key_values) == 0:
            return z
        first = past_key_values[0]
        # first layer is (k,v,len)
        if isinstance(first, (list, tuple)):
            if len(first) >= 3 and isinstance(first[2], jt.Var):
                return first[2]
            if len(first) >= 1 and hasattr(first[0], "current_length"):
                return first[0].current_length
        # first layer is object with .current_length
        if hasattr(first, "current_length"):
            return first.current_length
        # (k,v,len) but passed directly
        if len(past_key_values) >= 3 and isinstance(past_key_values[2], jt.Var):
            return past_key_values[2]

    # single object
    if hasattr(past_key_values, "current_length"):
        return past_key_values.current_length

    return z

def unpack_layer_pkv(layer_pkv):
    """
    Normalize a per-layer past_key_value into (k_var, v_var, len_var).
    Supports:
      - (k,v,len)
      - (k,v) where k.current_length exists
      - object with .key/.value/.current_length (best effort)
    """
    z = jt.zeros((1,), dtype=jt.int32)
    if layer_pkv is None:
        return None, None, z

    if isinstance(layer_pkv, (list, tuple)):
        if len(layer_pkv) >= 3 and isinstance(layer_pkv[2], jt.Var):
            return layer_pkv[0], layer_pkv[1], layer_pkv[2]
        if len(layer_pkv) >= 2 and hasattr(layer_pkv[0], "current_length"):
            return layer_pkv[0], layer_pkv[1], layer_pkv[0].current_length
        if len(layer_pkv) >= 2:
            # no len info
            return layer_pkv[0], layer_pkv[1], z

    # object style
    if hasattr(layer_pkv, "current_length"):
        k = getattr(layer_pkv, "k", None) or getattr(layer_pkv, "key", None) or getattr(layer_pkv, "key_cache", None) or getattr(layer_pkv, "data", None)
        v = getattr(layer_pkv, "v", None) or getattr(layer_pkv, "value", None) or getattr(layer_pkv, "value_cache", None)
        return k, v, layer_pkv.current_length

    return None, None, z