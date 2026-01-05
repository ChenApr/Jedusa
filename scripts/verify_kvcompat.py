#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import jittor as jt
from medusa_jittor_all.model.kv_compat import pkv_len_var

def main():
    # fabricate a pkv: list of layers, each (k,v,len)
    lenv = jt.array([7], dtype=jt.int32)
    k = jt.zeros((1, 8, 2048, 128), dtype=jt.float16)
    v = jt.zeros((1, 8, 2048, 128), dtype=jt.float16)
    pkv = [(k, v, lenv)]

    got = pkv_len_var(pkv)
    print("pkv_len_var:", got, "value=", int(got.item()))
    assert int(got.item()) == 7
    print("✅ kv_compat works for (k,v,len)")

if __name__ == "__main__":
    main()