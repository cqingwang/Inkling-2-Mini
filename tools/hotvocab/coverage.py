#!/usr/bin/env python3
"""Held-out coverage of a frequency-ranked draft hot-vocabulary list, and what the misses cost.

  python3 coverage.py --tokenizer TOKDIR traces.jsonl [more.jsonl ...] \
      [--eval-only speedbench-greedy.jsonl ...] [--folds 2] [--json report.json] [--write-hot hot.json]

Needs the `tokenizers` package (or `transformers`) only when a record has no token ids.

What is counted: every token the target emits after the prompt, i.e. the positions the draft has to propose.
  * gen_corpus.py records with `response_token_ids` (SGLang `return_token_ids`) are used as is: exact.
  * Text-only records are rebuilt the way the server emits them (checked against usage.completion_tokens on the
    round-1 traces: exact on 271/273 records):
      <|message_model|><|content_thinking|> reasoning <|end_message|>
      <|message_model|><|content_text|> content <|end_message|><|content_model_end_sampling|>
    with the closing tokens left out when the record stopped on max_tokens. Reasoning and content are tokenized
    separately (a special token sits between them, so BPE never merges across).
Accepted inputs per JSONL line: gen_corpus.py output; the round-1 trace format {prompt, effort, reasoning,
content}; speedbench greedy dumps {id, text = reasoning NUL content, usage}; a bare list of ids; SGLang
/generate output {output_ids}; OpenAI chat responses.

Method (same ranking rule as overlay/tools/build_hotvocab.py: EOS ids first, then ids by count):
  * Split BY PROMPT (sha1 of the prompt id or prompt text, --folds K). For each fold, rank on the other K-1
    folds and score the held-out fold; hits and totals are pooled over the folds.
  * Coverage(N) = held-out target tokens whose id is in the first N of the ranking. 1 - coverage = the share of
    positions where the draft cannot propose the target token (a forced rejection).
  * Learning curve: the same, ranking on 1/8, 1/4, 1/2 and all of the training prompts, to see whether more
    traces still move the curve.
  * --eval-only files are never used for ranking; they are scored against the list built from ALL traces.

Acceptance model [estimate]: the draft proposes k tokens per step (DSpark block 8 -> k = 7). With per-position
conditional acceptance a, the accepted length from a start position t is sum_{i<=k} a^i, and a hot-set miss at
t+j caps it at j (no draft can be accepted at or after the miss in that block). a is solved from the measured
tokens/step (--tokens-per-step, default 2.47 = the recipe README's 2.45-2.49) and the cap is applied with the
real miss positions of each held-out sequence, so clustered misses (a Polish paragraph) are priced as clustered.
It assumes a miss is independent of whether the draft would have been right; misses are rare tokens that the
draft gets right less often, so the true loss is lower. Speed factor at c1 = accept ratio x step / (step -
saved draft bytes / bandwidth).
"""

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys

SIZES = (8192, 16384, 24576, 32768, 40960, 49152, 65536, 81920, 98304)
MSG_MODEL, THINK, TEXT, END, EOS, TOOL_JSON, TOOL_TEXT, XML = (200001, 200008, 200004, 200010, 200006, 200049,
                                                               200050, 200047)
SEGMENT_OPEN = {THINK: "reasoning", TEXT: "content", TOOL_JSON: "tool", TOOL_TEXT: "tool", XML: "tool"}
SPECIAL_MIN = 199998  # ids >= this are template/special tokens in Inkling-2-Mini's tokenizer
VOCAB_ROWS = 201024


# ------------------------------------------------------------------ tokenizer

def load_encoder(path):
    tj = os.path.join(path, "tokenizer.json")
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tj)
        ids = {s: tok.token_to_id(s) for s in ("<|message_model|>", "<|content_thinking|>", "<|content_text|>",
                                                 "<|end_message|>", "<|content_model_end_sampling|>",
                                                 "<|content_invoke_tool_json|>")}
        return (lambda s: tok.encode(s, add_special_tokens=False).ids), ids
    except ImportError:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        ids = {s: tok.convert_tokens_to_ids(s) for s in ("<|message_model|>", "<|content_thinking|>",
                                                          "<|content_text|>", "<|end_message|>",
                                                          "<|content_model_end_sampling|>",
                                                          "<|content_invoke_tool_json|>")}
        return (lambda s: tok.encode(s, add_special_tokens=False)), ids


def eos_ids(path):
    out = []
    for name in ("generation_config.json", "config.json"):
        try:
            with open(os.path.join(path, name)) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        for c in (cfg, cfg.get("text_config", {})):
            v = c.get("eos_token_id")
            if isinstance(v, int):
                out.append(v)
            elif isinstance(v, list):
                out.extend(int(x) for x in v)
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ record metadata guesses

CODE_HINT = re.compile(r"```|\bdef \w+\(|#include|\b(python|rust|typescript|javascript|java|sql|bash|c\+\+|go|react|"
                       r"function|class|module|script|unit tests?|schema|queries)\b", re.I)
STRUCT_HINT = re.compile(r"\b(json|yaml|csv|markdown table|table)\b", re.I)
MATH_HINT = re.compile(r"\b(solve|calculate|how much|how many|compute|prove)\b|\$[^$]+\$", re.I)
LANG_NAMES = {"polish": "pl", "german": "de", "spanish": "es", "french": "fr", "italian": "it", "portuguese": "pt",
              "ukrainian": "uk", "chinese": "zh", "japanese": "ja", "russian": "ru", "korean": "ko"}


def guess_category(prompt):
    if not prompt:
        return "unknown"
    if MATH_HINT.search(prompt):
        return "math"
    if CODE_HINT.search(prompt):
        return "code"
    if STRUCT_HINT.search(prompt):
        return "structured"
    return "chat"


def guess_lang(text):
    t = text[:4000]
    if not t:
        return "unknown"
    n = len(t)
    kana = len(re.findall(r"[぀-ヿ]", t))
    han = len(re.findall(r"[一-鿿]", t))
    hang = len(re.findall(r"[가-힯]", t))
    cyr = len(re.findall(r"[Ѐ-ӿ]", t))
    if kana > 0.02 * n:
        return "ja"
    if han > 0.1 * n:
        return "zh"
    if hang > 0.1 * n:
        return "ko"
    if cyr > 0.2 * n:
        return "uk" if re.search(r"[іїєґ]", t) else "ru"
    low = t.lower()
    score = {"pl": len(re.findall(r"[ąęłńśźżć]", low)) + 3 * len(re.findall(r"\b(się|jest|nie|że|oraz)\b", low)),
             "de": len(re.findall(r"[äöüß]", low)) + 3 * len(re.findall(r"\b(und|der|die|das|nicht|ist)\b", low)),
             "es": len(re.findall(r"[ñ¿¡]", low)) + 3 * len(re.findall(r"\b(el|los|las|que|para|una)\b", low)),
             "fr": len(re.findall(r"[àâçèêëîïôœùû]", low)) + 3 * len(re.findall(r"\b(les|des|est|une|pour|avec)\b", low)),
             "it": 3 * len(re.findall(r"\b(il|gli|della|che|sono|per)\b", low)),
             "pt": len(re.findall(r"[ãõ]", low)) + 3 * len(re.findall(r"\b(não|uma|são|com|para)\b", low))}
    en = 3 * len(re.findall(r"\b(the|and|is|of|to|with)\b", low))
    best = max(score, key=score.get)
    return best if score[best] > max(en, 6) else "en"


# ------------------------------------------------------------------ record parsing

def text_record_ids(enc, reasoning, content, finished, tool_calls=None):
    """Rebuild the emitted ids from text (see module doc); returns (ids, segment labels)."""
    ids, seg = [], []

    def add(xs, label):
        ids.extend(xs)
        seg.extend([label] * len(xs))

    if reasoning or not content:
        add([MSG_MODEL, THINK], "special")
        add(enc(reasoning) if reasoning else [], "reasoning")
        if content or tool_calls or finished:
            add([END], "special")
    if content:
        add([MSG_MODEL, TEXT], "special")
        add(enc(content), "content")
        if finished or tool_calls:
            add([END], "special")
    for tc in tool_calls or []:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        add([MSG_MODEL], "special")
        add(enc(name), "tool")
        add([TOOL_JSON], "special")
        add(enc(json.dumps({"name": name, "args": args}, separators=(",", ":"), sort_keys=True)), "tool")
        add([END], "special")
    if finished:
        add([EOS], "special")
    return ids, seg


def segments_from_ids(ids):
    seg, cur = [], "reasoning"
    for t in ids:
        if t >= SPECIAL_MIN:
            seg.append("special")
            cur = SEGMENT_OPEN.get(t, cur)
        else:
            seg.append(cur)
    return seg


def parse_line(obj, enc):
    """-> dict(key, meta, ids, seg, mode, usage_ct) or None."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, int) for x in obj):
            return dict(key=hashlib.sha1(json.dumps(obj[:64]).encode()).hexdigest(), meta={}, ids=obj,
                        seg=segments_from_ids(obj), mode="ids", usage_ct=None)
        return None
    if not isinstance(obj, dict) or obj.get("error"):
        return None
    usage = obj.get("usage") or {}
    meta = {k: obj.get(k) for k in ("category", "language", "effort", "temperature", "source") if obj.get(k) is not None}
    finish = obj.get("finish_reason")
    for key in ("response_token_ids", "output_ids", "token_ids", "output_token_ids", "completion_token_ids"):
        v = obj.get(key)
        if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
            k = obj.get("id") or obj.get("prompt") or hashlib.sha1(json.dumps(v[:64]).encode()).hexdigest()
            txt = (obj.get("reasoning_content") or "") + (obj.get("content") or "") + (obj.get("text") or "")
            meta.setdefault("language", guess_lang(obj.get("content") or txt))
            return dict(key=str(k), meta=meta, ids=v, seg=segments_from_ids(v), mode="ids",
                        usage_ct=usage.get("completion_tokens"))
    if "choices" in obj:  # OpenAI response
        ch = (obj.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        v = ch.get("response_token_ids")
        if isinstance(v, list) and v:
            return dict(key=str(obj.get("id")), meta=meta, ids=v, seg=segments_from_ids(v), mode="ids",
                        usage_ct=usage.get("completion_tokens"))
        reasoning, content = m.get("reasoning_content") or "", m.get("content") or ""
        finished = ch.get("finish_reason") != "length"
        ids, seg = text_record_ids(enc, reasoning, content, finished, m.get("tool_calls"))
        meta.setdefault("language", guess_lang(content or reasoning))
        return dict(key=str(obj.get("id")), meta=meta, ids=ids, seg=seg, mode="text",
                    usage_ct=usage.get("completion_tokens"))
    reasoning = obj.get("reasoning_content") or obj.get("reasoning") or ""
    content = obj.get("content") or ""
    if not reasoning and not content and isinstance(obj.get("text"), str):
        t = obj["text"]
        reasoning, content = t.split("\u0000", 1) if "\u0000" in t else ("", t)
    if not reasoning and not content and not obj.get("tool_calls"):
        return None
    # finished: explicit finish_reason, else infer from usage (completion < max_tokens), else assume finished
    if finish is not None:
        finished = finish != "length"
    elif obj.get("max_tokens") and usage.get("completion_tokens"):
        finished = usage["completion_tokens"] < obj["max_tokens"]
    else:
        finished = None
    ids, seg = text_record_ids(enc, reasoning, content, True if finished is None else finished, obj.get("tool_calls"))
    if finished is None and usage.get("completion_tokens"):
        # round-1 traces have no finish_reason: when usage says the closing tokens were not emitted (the record
        # stopped on max_tokens), drop them
        extra = len(ids) - usage["completion_tokens"]
        if 0 < extra <= 2 and all(t in (END, EOS) for t in ids[-extra:]):
            ids, seg = ids[:-extra], seg[:-extra]
    prompt = obj.get("prompt")
    key = obj.get("id") or prompt or hashlib.sha1((reasoning + content)[:2000].encode()).hexdigest()
    if "category" not in meta:
        cid = str(obj.get("id") or "")
        m = re.match(r"(prose|code|structured|agentic|longctx|multilingual)-", cid)
        meta["category"] = m.group(1) if m else guess_category(prompt or "")
    if "language" not in meta:
        m = re.search(r"Answer in (\w+)", prompt or "")
        meta["language"] = LANG_NAMES.get(m.group(1).lower(), "other") if m else guess_lang(content or reasoning)
    return dict(key=str(key), meta=meta, ids=ids, seg=seg, mode="text", usage_ct=usage.get("completion_tokens"))


def read_records(paths, enc, dedupe=True):
    seen = set()
    for p in paths:
        with open(p, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(b"data:"):
                    line = line[5:].strip()
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                r = parse_line(obj, enc)
                if r is None or not r["ids"]:
                    continue
                sig = hashlib.sha1(json.dumps(r["ids"]).encode()).hexdigest()
                if dedupe and sig in seen:  # identical outputs (e.g. the same greedy dump from several arms)
                    continue
                seen.add(sig)
                r["file"] = os.path.basename(p)
                yield r


# ------------------------------------------------------------------ ranking and scoring

PRIOR = collections.Counter()  # back-fill for ids never seen in the training traces (--prior)


def rank(counter, front):
    fs = set(front)
    out = list(front) + [t for t, _ in counter.most_common() if t not in fs]
    if PRIOR:
        seen = fs | set(counter)
        out += [t for t, _ in PRIOR.most_common() if t not in seen]
    return out


def load_prior(paths, enc, max_chars):
    """Token counts of prompt text (gen_corpus prompt files or any JSONL with messages/content/text)."""
    cnt, used = collections.Counter(), 0
    for p in paths:
        with open(p, "rb") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                texts = [m.get("content") for m in obj.get("messages", []) if isinstance(m.get("content"), str)]
                texts += [obj[k] for k in ("text", "content") if isinstance(obj.get(k), str)]
                for t in texts:
                    cnt.update(i for i in enc(t) if i < SPECIAL_MIN)
                    used += len(t)
                if used >= max_chars:
                    return cnt
    return cnt


def fold_of(key, k):
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % k


def solve_a(tokens_per_step, k):
    target = tokens_per_step - 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(mid ** i for i in range(1, k + 1)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class Scorer:
    """Accumulates hits/totals per group and the acceptance-model sums for a set of N."""

    def __init__(self, sizes, a_list, k):
        self.sizes = sizes
        self.k = k
        self.a_list = a_list
        self.geo = {a: [sum(a ** i for i in range(1, j + 1)) for j in range(k + 1)] for a in a_list}
        self.tot = collections.Counter()           # (group, value) -> tokens
        self.hit = collections.Counter()           # (group, value, N) -> hits
        self.acc_full = collections.Counter()      # a -> sum over start positions
        self.acc_hot = collections.Counter()       # (a, N) -> sum
        self.rec_miss = {n: [] for n in sizes}     # per-record miss rate (for the spread)

    def add(self, r, ranks):
        ids, seg, meta = r["ids"], r["seg"], r["meta"]
        groups = [("all", "all"), ("category", meta.get("category", "unknown")),
                  ("language", meta.get("language", "unknown")), ("effort", str(meta.get("effort", "unknown"))),
                  ("temperature", str(meta.get("temperature", "unknown"))), ("source", meta.get("source", r["file"]))]
        L = len(ids)
        for g in groups:
            self.tot[g] += L
        segc = collections.Counter(seg)
        for s, c in segc.items():
            self.tot[("segment", s)] += c
        for n in self.sizes:
            top = ranks[n]
            hits = [t in top for t in ids]
            nh = sum(hits)
            for g in groups:
                self.hit[g + (n,)] += nh
            for s in segc:
                self.hit[("segment", s, n)] += sum(h for h, sg in zip(hits, seg) if sg == s)
            self.rec_miss[n].append(1 - nh / L)
            # run[t] = consecutive hits starting at t, capped at k
            run = [0] * (L + 1)
            for t in range(L - 1, -1, -1):
                run[t] = min(self.k, run[t + 1] + 1) if hits[t] else 0
            for a in self.a_list:
                g = self.geo[a]
                self.acc_hot[(a, n)] += sum(g[min(run[t], L - t)] for t in range(L))
        for a in self.a_list:
            g = self.geo[a]
            self.acc_full[a] += sum(g[min(self.k, L - t)] for t in range(L))

    def cov(self, g, n):
        return self.hit[g + (n,)] / self.tot[g] if self.tot[g] else float("nan")


def breakeven_miss(a, k, keep):
    """Independent-miss model: miss rate m at which tokens/step falls to `keep` x the full-vocab value."""
    full = 1 + sum(a ** i for i in range(1, k + 1))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        if (1 + sum((a * (1 - m)) ** i for i in range(1, k + 1))) / full > keep:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def draft_bytes(n, head, fold, steps):
    head_row = 2304 * 2 if head == "bf16" else 2304 + 4
    fold_row = 256 * 2 if fold == "bf16" else 256 + 4
    return n * (head_row + steps * fold_row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--tokenizer", required=True, help="dir with tokenizer.json (+ generation_config.json for EOS)")
    ap.add_argument("--eval-only", nargs="*", default=[], help="scored against the list from ALL traces, never ranked on")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--extra-ids", default="", help="ids always put first (as build_hotvocab --extra-ids)")
    ap.add_argument("--tokens-per-step", type=float, default=2.47, help="measured mean tokens per verify step")
    ap.add_argument("--draft-k", type=int, default=7, help="draft tokens per step (DSpark block 8 -> 7)")
    ap.add_argument("--step-ms", type=float, default=43.1,
                    help="c1 step time with the full-vocab draft (recipe README full stack: 43.1 ms)")
    ap.add_argument("--gbps", type=float, default=220.0)
    ap.add_argument("--draft-head", choices=("bf16", "fp8"), default="fp8",
                    help="draft LM head format on both sides of the comparison (shipped DRAFT_BYTES has head8 = fp8)")
    ap.add_argument("--draft-fold", choices=("bf16", "fp8"), default="fp8", help="fold format (shipped: fold8 = fp8)")
    ap.add_argument("--prior", nargs="*", default=[],
                    help="text JSONL (e.g. prompts.jsonl) whose token counts rank ids never seen in the traces, after "
                         "all seen ids (the traces always decide the order of seen ids)")
    ap.add_argument("--prior-max-chars", type=int, default=60_000_000)
    ap.add_argument("--no-learning-curve", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--write-hot", default=None, help="also write the hot list (JSON ids, most frequent first) "
                                                     "ranked on ALL traces, like build_hotvocab.py")
    ap.add_argument("--top", type=int, default=98304, help="ids to write with --write-hot")
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    enc, sp = load_encoder(a.tokenizer)
    expect = {"<|message_model|>": MSG_MODEL, "<|content_thinking|>": THINK, "<|content_text|>": TEXT,
              "<|end_message|>": END, "<|content_model_end_sampling|>": EOS, "<|content_invoke_tool_json|>": TOOL_JSON}
    if any(sp.get(k) != v for k, v in expect.items()):
        print(f"tokenizer special ids differ from Inkling-2-Mini's: {sp}", file=sys.stderr)
        return 2
    front = [int(x) for x in a.extra_ids.split(",") if x.strip()]
    front += [t for t in eos_ids(a.tokenizer) if t not in front]

    if a.prior:
        PRIOR.update(load_prior(a.prior, enc, a.prior_max_chars))
        print(f"prior: {sum(PRIOR.values()):,} tokens, {len(PRIOR):,} distinct ids from {a.prior}")
    recs = list(read_records(a.traces, enc))
    if not recs:
        print("no records", file=sys.stderr)
        return 1
    evals = list(read_records(a.eval_only, enc)) if a.eval_only else []
    a_val = solve_a(a.tokens_per_step, a.draft_k)
    a_list = sorted({round(a_val, 4), 0.7, 0.8})

    # sanity: rebuilt length vs usage.completion_tokens on text records
    chk = [(len(r["ids"]), r["usage_ct"]) for r in recs if r["mode"] == "text" and r["usage_ct"]]
    exact = sum(1 for x, y in chk if x == y)
    modes = collections.Counter(r["mode"] for r in recs)
    total_tok = sum(len(r["ids"]) for r in recs)
    prompts = {r["key"] for r in recs}
    print(f"records {len(recs)} ({dict(modes)}), prompts {len(prompts)}, target tokens {total_tok:,}, "
          f"distinct ids {len({t for r in recs for t in r['ids']}):,}")
    if chk:
        print(f"text rebuild vs usage.completion_tokens: exact on {exact}/{len(chk)} records, "
              f"sum {sum(x for x, _ in chk):,} vs {sum(y for _, y in chk):,}")
    print(f"acceptance model: k={a.draft_k}, tokens/step {a.tokens_per_step} -> a = {a_val:.4f}")

    # ---------------- K-fold by prompt
    folds = [[] for _ in range(a.folds)]
    for r in recs:
        folds[fold_of(r["key"], a.folds)].append(r)
    sc = Scorer(sizes, a_list, a.draft_k)
    for f in range(a.folds):
        cnt = collections.Counter()
        for g in range(a.folds):
            if g != f:
                for r in folds[g]:
                    cnt.update(r["ids"])
        ranked = rank(cnt, front)
        ranks = {n: set(ranked[:n]) for n in sizes}
        for r in folds[f]:
            sc.add(r, ranks)

    def table(title, group, values, min_tok=0):
        print(f"\n{title}  (held-out, split by prompt, {a.folds} folds pooled; coverage %, miss % = 100 - coverage)")
        print(f"{'':<14}{'tokens':>10} " + " ".join(f"{n // 1024:>5}K" for n in sizes))
        rows = {}
        for v in values:
            g = (group, v)
            if sc.tot[g] < min_tok:
                continue
            cv = {n: sc.cov(g, n) for n in sizes}
            rows[v] = {"tokens": sc.tot[g], "coverage": {str(n): cv[n] for n in sizes}}
            print(f"{str(v):<14}{sc.tot[g]:>10,} " + " ".join(f"{100 * cv[n]:6.2f}" for n in sizes))
        return rows

    rep = {"records": len(recs), "prompts": len(prompts), "tokens": total_tok, "modes": dict(modes),
           "text_rebuild_exact": [exact, len(chk)], "folds": a.folds, "sizes": sizes, "a": a_val}
    rep["all"] = table("ALL", "all", ["all"])
    for grp in ("category", "language", "segment", "effort", "temperature", "source"):
        vals = sorted({k[1] for k in sc.tot if k[0] == grp}, key=lambda v: -sc.tot[(grp, v)])
        rep[grp] = table(f"by {grp}", grp, vals)

    # per-record spread
    print("\nper-record miss rate (held-out): median / p90 / p99 / max, %")
    rep["record_miss_spread"] = {}
    for n in sizes:
        xs = sorted(sc.rec_miss[n])
        q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
        rep["record_miss_spread"][str(n)] = [q(0.5), q(0.9), q(0.99), xs[-1]]
        print(f"  {n // 1024:>3}K  {100 * q(0.5):6.2f} {100 * q(0.9):6.2f} {100 * q(0.99):6.2f} {100 * xs[-1]:6.2f}")

    # acceptance and speed model
    base = draft_bytes(VOCAB_ROWS, a.draft_head, a.draft_fold, a.draft_k)
    print(f"\nacceptance / speed model at c1 [estimate]: draft head {a.draft_head}, fold {a.draft_fold}; full-vocab "
          f"draft vocab bytes {base / 1e6:.0f} MB/step, step {a.step_ms} ms, {a.gbps} GB/s")
    print(f"{'N':>6} {'miss %':>7} {'saved MB':>9} {'saved ms':>9} {'b/e miss %':>10} " +
          " ".join(f"{'acc a=' + format(x, '.2f'):>11} {'tok/s':>7}" for x in a_list))
    rep["model"] = {}
    for n in sizes:
        hb = draft_bytes(n, a.draft_head, a.draft_fold, a.draft_k)
        saved_ms = (base - hb) / (a.gbps * 1e9) * 1e3
        be = breakeven_miss(a_val, a.draft_k, (a.step_ms - saved_ms) / a.step_ms)
        row = {"miss": 1 - sc.cov(("all", "all"), n), "saved_mb": (base - hb) / 1e6, "saved_ms": saved_ms,
               "breakeven_miss_independent": be}
        cells = []
        for x in a_list:
            acc_ratio = (sc.tot[("all", "all")] + sc.acc_hot[(x, n)]) / (sc.tot[("all", "all")] + sc.acc_full[x])
            speed = acc_ratio * a.step_ms / (a.step_ms - saved_ms)
            row[f"a={x:.4f}"] = {"tokens_per_step_ratio": acc_ratio, "speed_factor": speed}
            cells.append(f"{100 * (acc_ratio - 1):+10.2f}% {100 * (speed - 1):+6.2f}%")
        rep["model"][str(n)] = row
        print(f"{n:>6} {100 * row['miss']:7.2f} {(base - hb) / 1e6:9.0f} {saved_ms:9.2f} {100 * be:10.2f} " + " ".join(cells))
    print("(acc = change of tokens per step incl. the bonus token, from the real miss positions; tok/s = acc x step "
          "time saved; b/e miss = miss rate at which tok/s is unchanged if misses were independent, at a=%.3f, "
          "calibrated to %.2f tokens/step)" % (a_val, a.tokens_per_step))

    # ---------------- learning curve
    if not a.no_learning_curve and a.folds >= 2:
        print("\nlearning curve: rank on a fraction of the training prompts (fold 0 held out)")
        test = folds[0]
        train_recs = [r for g in range(1, a.folds) for r in folds[g]]
        keys = sorted({r["key"] for r in train_recs}, key=lambda k: hashlib.sha1(("lc" + k).encode()).hexdigest())
        ttot = sum(len(r["ids"]) for r in test)
        rep["learning_curve"] = []
        show = [n for n in sizes if n in (16384, 32768, 49152, 65536)] or sizes[:4]
        print(f"{'train frac':>10} {'train tok':>11} " + " ".join(f"{n // 1024:>5}K" for n in show))
        for frac in (0.125, 0.25, 0.5, 1.0):
            sub = set(keys[: max(1, int(len(keys) * frac))])
            cnt = collections.Counter()
            ntok = 0
            for r in train_recs:
                if r["key"] in sub:
                    cnt.update(r["ids"])
                    ntok += len(r["ids"])
            ranked = rank(cnt, front)
            cv = {}
            for n in show:
                top = set(ranked[:n])
                cv[n] = sum(sum(t in top for t in r["ids"]) for r in test) / ttot
            rep["learning_curve"].append({"frac": frac, "train_tokens": ntok, "coverage": {str(n): cv[n] for n in show}})
            print(f"{frac:>10} {ntok:>11,} " + " ".join(f"{100 * cv[n]:6.2f}" for n in show))

    # ---------------- list from ALL traces, eval-only sets, written list
    cnt_all = collections.Counter()
    for r in recs:
        cnt_all.update(r["ids"])
    ranked_all = rank(cnt_all, front)
    if evals:
        print(f"\neval-only sets, list ranked on ALL {len(recs)} trace records ({total_tok:,} tokens)")
        sc2 = Scorer(sizes, [a_val], a.draft_k)
        ranks = {n: set(ranked_all[:n]) for n in sizes}
        for r in evals:
            sc2.add(r, ranks)
        sc_save, sc = sc, sc2
        rep["eval_only"] = {"records": len(evals)}
        rep["eval_only"]["all"] = table("eval-only ALL", "all", ["all"])
        rep["eval_only"]["category"] = table("eval-only by category", "category",
                                             sorted({k[1] for k in sc.tot if k[0] == "category"}))
        rep["eval_only"]["segment"] = table("eval-only by segment", "segment",
                                            sorted({k[1] for k in sc.tot if k[0] == "segment"}))
        sc = sc_save
    if a.write_hot:
        out = ranked_all[: a.top]
        os.makedirs(os.path.dirname(os.path.abspath(a.write_hot)) or ".", exist_ok=True)
        with open(a.write_hot, "w") as f:
            json.dump(out, f)
        print(f"\nwrote {len(out)} ids ({len(cnt_all)} distinct seen; EOS/extra first: {front}) -> {a.write_hot}")
        rep["written"] = {"path": a.write_hot, "ids": len(out), "distinct_seen": len(cnt_all), "front": front}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
