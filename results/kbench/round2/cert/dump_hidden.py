#!/usr/bin/env python3
"""Teacher-forced CPU-oracle forward of the 16 speedbench greedy transcripts; saves the LM-head input rows
(final_hidden, rounded to BF16 as SGLang materialises it) for every position from the last prompt token on.

Output: <out>/<id>.npz with hidden (uint16 BF16 bits [T, 2304]), ids, start.
"""
import json
import os
import sys
import time

import numpy as np

REF = os.environ.get("INKLING_REF", "ref")
sys.path.insert(0, REF)
from inkling2_ref import Inkling2Ref  # noqa: E402
from validate_sglang_greedy import build  # noqa: E402


def bf16_bits(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def main():
    out_dir = sys.argv[1]
    ids_sel = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    compute = os.environ.get("COMPUTE", "fp32")
    os.makedirs(out_dir, exist_ok=True)
    from tokenizers import Tokenizer
    w = os.path.join(REF, "..", ".weights")
    tok = Tokenizer.from_file(os.path.join(w, "tokenizer.json"))
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(REF, "data", "sglang_B0_greedy.jsonl"))}
    prompts = {p["id"]: p for p in json.load(open(os.path.join(REF, "data", "speedbench_v1_greedy_prompts.json")))["prompts"]}
    model = Inkling2Ref(w, compute, expert_threads=4)
    for pid in ids_sel or list(recs):
        f = os.path.join(out_dir, f"{pid}.npz")
        if os.path.exists(f):
            continue
        p_ids, comp, checks, ok = build(tok, prompts[pid], recs[pid])
        ids = p_ids + comp
        t0 = time.time()
        out = model.forward(ids, return_logits=False) if "return_logits" in model.forward.__code__.co_varnames else model.forward(ids)
        h = out["final_hidden"][len(p_ids) - 1:]
        np.savez(f, hidden=bf16_bits(h), ids=np.array(ids), start=len(p_ids) - 1, ok=ok)
        print(f"{pid}: {h.shape[0]} rows ok={ok} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
