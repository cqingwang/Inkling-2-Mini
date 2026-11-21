#!/usr/bin/env python3
"""Download the public prompt sources for build_prompts.py (Mac side; needs huggingface_hub, pyarrow).

  python3 -I fetch_sources.py DOWNLOAD_DIR

Fetches single files from each HF dataset repo (about 1.6 GB in all), plus one parquet row group of Wikipedia
(20231101) per language through HfFileSystem range reads (a few MB each) into DOWNLOAD_DIR/wiki/wiki.jsonl.
Treat the downloads as untrusted data: keep them in their own directory and read them with python -I.
"""
import json
import os
import random
import sys
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

D = sys.argv[1]
jobs = [
 ("OpenAssistant/oasst2", "data/train-00000-of-00001-88ba0162028a73fc.parquet"),
 ("OpenAssistant/oasst2", "data/validation-00000-of-00001-1deeef95c3248fe0.parquet"),
 ("HuggingFaceH4/ultrachat_200k", "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"),
 ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
 ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
 ("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl"),
 ("abisee/cnn_dailymail", "3.0.0/test-00000-of-00001.parquet"),
 ("NousResearch/hermes-function-calling-v1", "func-calling-singleturn.json"),
 ("NousResearch/hermes-function-calling-v1", "func-calling.json"),
 ("NousResearch/hermes-function-calling-v1", "glaive-function-calling-5k.json"),
 ("NousResearch/hermes-function-calling-v1", "json-mode-singleturn.json"),
 ("NousResearch/hermes-function-calling-v1", "json-mode-agentic.json"),
 ("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet"),
 ("llm-jp/oasst2-33k-ja", "oasst2-33k-ja.jsonl"),
 ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "data/train-00000-of-00001.parquet"),
 ("ise-uiuc/Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl"),
 ("allenai/WildChat-1M", "data/train-00000-of-00014.parquet"),
 ("allenai/WildChat-1M", "data/train-00009-of-00014.parquet"),
]
for sub in ["algebra","counting_and_probability","geometry","intermediate_algebra","number_theory","prealgebra","precalculus"]:
    jobs.append(("EleutherAI/hendrycks_math", f"{sub}/train-00000-of-00001.parquet"))
for l in ["en","es","fr","de","ru","zh","ja","th","sw","bn","te"]:
    jobs.append(("juletxara/mgsm", f"{l}/test-00000-of-00001.parquet"))
for repo, f in jobs:
    t = time.time()
    try:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=f"{D}/{repo.replace('/', '__')}")
        print("ok", repo, f, f"{time.time() - t:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print("FAIL", repo, f, repr(e)[:200], flush=True)

fs = HfFileSystem()
os.makedirs(os.path.join(D, "wiki"), exist_ok=True)
want = {"pl": 500, "de": 350, "es": 350, "zh": 300, "ja": 300, "fr": 250, "en": 600, "ru": 250}
with open(os.path.join(D, "wiki", "wiki.jsonl"), "w") as f:
    for lang, n in want.items():
        t = time.time()
        files = fs.ls(f"datasets/wikimedia/wikipedia/20231101.{lang}", detail=False)
        files = sorted(x for x in files if x.endswith(".parquet"))
        p = files[1]
        with fs.open(p, "rb", block_size=4 << 20) as fh:
            pf = pq.ParquetFile(fh)
            rg = pf.num_row_groups // 2
            tab = pf.read_row_group(rg, columns=["id", "url", "title", "text"])
        rows = tab.to_pylist()
        rng = random.Random(lang)
        # keep medium/long articles (summaries of 1.5K-24K chars)
        rows = [r for r in rows if 1500 <= len(r["text"]) <= 40000]
        rng.shuffle(rows)
        for r in rows[:n]:
            r["lang"] = lang
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(lang, p, "rg", rg, "of", pf.num_row_groups, "kept", min(n, len(rows)), f"{time.time()-t:.0f}s", flush=True)
