# -*- coding: utf-8 -*-
"""
Wrapper module:
- Ensure fixed attention patch is loaded (by importing modeling_llama_jt wrapper)
- Re-export MedusaModel from medusa_model_orig so external benchmark can do:
    --medusa_impl medusa_jittor_all.model.medusa_model:MedusaModel
"""

# Import modeling_llama_jt to trigger monkeypatch
try:
    from . import modeling_llama_jt  # noqa: F401
except Exception as _e:
    print(f"[WARN] failed to import modeling_llama_jt for patch trigger: {_e}", flush=True)

# Re-export everything from orig
from .medusa_model_orig import *  # noqa: F401,F403

# Optional: harden that MedusaModel exists for your benchmark resolver
if "MedusaModel" not in globals():
    # Try best effort: find a class name containing MedusaModel
    for _k, _v in list(globals().items()):
        if isinstance(_v, type) and (_k.lower() == "medusamodel" or _k.endswith("MedusaModel")):
            globals()["MedusaModel"] = _v
            break