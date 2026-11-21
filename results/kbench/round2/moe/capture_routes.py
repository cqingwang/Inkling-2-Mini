#!/usr/bin/env python3
"""Real Inkling-2-Mini routing for the MoE tile bench, from the engine's CPU oracle.

Teacher-forces SGLang's greedy outputs (engine/ref/data/sglang_B0_greedy.jsonl, rebuilt exactly as
validate_sglang_greedy.py does) through engine/ref/inkling2_ref.py and keeps the top-8 expert ids of
every MoE visit and position. Verify rows of DSpark are draft tokens; when drafts are accepted they are
the model's own continuation, so consecutive positions of the greedy text are the routing of a verify
block. Output: routes.npz with ids[prompt] = int16 [48 visits, T, 8] plus the visit -> layer map.

Usage: uv run --with numpy --with tokenizers python capture_routes.py --max-pos 160 --out routes.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REF = os.environ.get("INKLING_REF", "ref")
WEIGHTS = os.environ.get("INKLING_WEIGHTS", "weights")
sys.path.insert(0, REF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pos", type=int, default=160, help="positions per prompt (prompt + completion)")
    ap.add_argument("--ids", default="", help="comma list of prompt ids (default: all 16)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="routes.npz")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    from tokenizers import Tokenizer
    from inkling2_ref import Inkling2Ref
    from validate_sglang_greedy import build

    tok = Tokenizer.from_file(os.path.join(WEIGHTS, "tokenizer.json"))
    recs = [json.loads(l) for l in open(os.path.join(REF, "data", "sglang_B0_greedy.jsonl"))]
    prompts = {p["id"]: p for p in json.load(open(os.path.join(REF, "data", "speedbench_v1_greedy_prompts.json")))["prompts"]}
    want = [s for s in a.ids.split(",") if s] or [r["id"] for r in recs]
    model = Inkling2Ref(WEIGHTS, "fp32", expert_threads=a.threads)
    layer_of_visit = [v.layer for v in model.visits]
    out = {"layer_of_visit": np.array(layer_of_visit, np.int16)}
    if os.path.exists(a.out):
        old = np.load(a.out)
        out.update({k: old[k] for k in old.files})
    for rec in recs:
        pid = rec["id"]
        if pid not in want or ("ids_" + pid) in out:
            continue
        p_ids, comp, checks, ok = build(tok, prompts[pid], rec)
        ids = (p_ids + comp)[: a.max_pos]
        t0 = time.time()
        res = model.forward(ids, capture={"topk_ids"})
        caps = res["captures"]
        arr = np.stack([caps[f"{v:02d}/topk_ids"] for v in range(len(model.visits))]).astype(np.int16)
        out["ids_" + pid] = arr
        out["plen_" + pid] = np.array(len(p_ids), np.int32)
        np.savez(a.out, **out)
        print(f"{pid}: T={len(ids)} prompt={len(p_ids)} ok={ok} {time.time() - t0:.0f}s shape={arr.shape}", flush=True)


if __name__ == "__main__":
    main()
