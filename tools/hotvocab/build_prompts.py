#!/usr/bin/env python3
"""Build the prompt corpus for the hot-vocabulary trace run (Mac side; needs pyarrow).

  python3 -I build_prompts.py --dl DOWNLOAD_DIR --speedbench SPEEDBENCH_REPO --out prompts.jsonl

DOWNLOAD_DIR holds the public datasets fetched by fetch_sources.py (one sub-directory per HF repo,
"owner__name"), plus wiki/wiki.jsonl. Output: one JSON object per line
  {id, source, license, category, language, messages, tools?, reasoning_effort, max_tokens}
shuffled with a fixed seed, so any prefix of the file has the same mix.

category: chat | code | math | structured | agentic | summary  (task type)
language: ISO-639-1 code of the language the answer is expected in (multilingual share = language != en)

Exclusions:
  * near-duplicates of any speedbench prompt (prompts/*.json; word 6-gram / CJK char 8-gram containment
    >= 0.25 in either direction) and the bench/think_load.py arithmetic-puzzle shape;
  * exact duplicates after whitespace/case normalisation;
  * WildChat rows flagged toxic or redacted; Midjourney prompt-generator boilerplate (kept at 40).
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq

SEED = 20261121
LIC = {
    "wildchat": "ODC-BY-1.0 (allenai/WildChat-1M)",
    "ultrachat": "MIT (HuggingFaceH4/ultrachat_200k)",
    "oasst2": "Apache-2.0 (OpenAssistant/oasst2)",
    "oasst2ja": "Apache-2.0 (llm-jp/oasst2-33k-ja)",
    "dolly": "CC-BY-SA-3.0 (databricks/databricks-dolly-15k)",
    "magicoder": "MIT (ise-uiuc/Magicoder-OSS-Instruct-75K)",
    "selfoss": "ODC-BY (bigcode/self-oss-instruct-sc2-exec-filter-50k)",
    "gsm8k": "MIT (openai/gsm8k)",
    "math": "MIT (EleutherAI/hendrycks_math)",
    "mgsm": "CC-BY-SA-4.0 (juletxara/mgsm)",
    "hermes": "Apache-2.0 (NousResearch/hermes-function-calling-v1)",
    "aya": "Apache-2.0 (CohereLabs/aya_dataset)",
    "cnn": "Apache-2.0 (abisee/cnn_dailymail 3.0.0)",
    "wiki": "CC-BY-SA-4.0 (wikimedia/wikipedia 20231101)",
}

# Language of the answer -> instruction appended to ask for it (for "respond in X" variants).
RESPOND_IN = {
    "pl": "Odpowiedz po polsku.",
    "de": "Bitte antworte auf Deutsch.",
    "es": "Responde en español.",
    "fr": "Réponds en français.",
    "zh": "请用中文回答。",
    "ja": "日本語で答えてください。",
    "it": "Rispondi in italiano.",
    "ru": "Ответь на русском языке.",
}

WILDCHAT_LANG = {"English": "en", "Chinese": "zh", "Russian": "ru", "French": "fr", "Spanish": "es",
                 "German": "de", "Polish": "pl", "Japanese": "ja", "Portuguese": "pt", "Italian": "it",
                 "Korean": "ko", "Ukrainian": "uk", "Vietnamese": "vi", "Turkish": "tr"}
AYA_LANG = {"Polish": "pl", "German": "de", "Japanese": "ja", "Simplified Chinese": "zh", "Spanish": "es",
            "French": "fr", "Italian": "it", "Korean": "ko", "Ukrainian": "uk", "Portuguese": "pt",
            "Russian": "ru", "English": "en"}
OASST_LANG = {"pt-BR": "pt", "uk-UA": "uk"}

LANGS = r"(python|javascript|typescript|java|c\+\+|c#|rust|golang|bash|shell|sql|php|kotlin|swift|html|css|react|node\.?js|pandas|numpy|django|flask|docker|kubernetes|matlab|vba|excel formula|powershell|lua|haskell|scala|ruby|perl|unity|godot)"
CODE_RE = re.compile(r"```|\bdef \w+\(|#include|\bpublic (static|class)\b|^\s*(import|from) \w+|\bSELECT\b.+\bFROM\b|"
                     r"\bfunction \w*\(|Traceback \(most recent|\bconsole\.log\(|=>\s*\{|\b(npm|pip) install\b|"
                     r"\bgit (commit|push|clone|rebase|merge)\b|\bclass \w+\s*[:({]|;\s*\n\s*\}|"
                     r"\b(in|using|with|w|en|auf|mit|用|で)\s+" + LANGS + r"\b|\b" + LANGS + r"\s+(code|script|function|program|class|module|query|snippet|component|kod|código|代码)\b|"
                     r"\b(write|implement|create|fix|debug|refactor|optimi[sz]e)\s+(a|an|the|this|my)?\s*(function|script|program|class|method|query|regex|unit tests?|code)\b",
                     re.I | re.S | re.M)
STRUCT_RE = re.compile(r"\b(json|yaml|csv|xml|table|tabel|tabelle|tabla|表格|表)\b", re.I)
THINK_LOAD_RE = re.compile(r"reach the target|using each number at most once|Numbers:\s*\[|Target:\s*\d", re.I)
MJ_RE = re.compile(r"prompt generator for a generative AI called \"Midjourney\"|/imagine prompt", re.I)

CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def shingles(s):
    s = norm(s)
    if len(CJK_RE.findall(s)) > len(s) * 0.2:
        t = re.sub(r"\s", "", s)
        return {t[i:i + 8] for i in range(max(1, len(t) - 7))}
    w = re.findall(r"\w+", s)
    return {" ".join(w[i:i + 6]) for i in range(max(1, len(w) - 5))}


class BenchFilter:
    """Containment of word 6-gram (CJK: char 8-gram) shingles against every speedbench prompt."""

    def __init__(self, repo):
        self.index = collections.defaultdict(set)
        self.sizes = {}
        texts = []
        for f in sorted(glob.glob(os.path.join(repo, "prompts", "*.json"))):
            texts.extend(self._texts(json.load(open(f))))
        for i, t in enumerate(texts):
            sh = shingles(t)
            if len(sh) < 3:
                continue
            self.sizes[i] = len(sh)
            for x in sh:
                self.index[x].add(i)
        self.n_bench = len(self.sizes)
        self.dropped = 0

    def _texts(self, obj):
        out = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "content" and isinstance(v, str) and len(v) > 40:
                    out.append(v)
                else:
                    out.extend(self._texts(v))
        elif isinstance(obj, list):
            for v in obj:
                out.extend(self._texts(v))
        return out

    def hit(self, text, thr=0.25):
        sh = shingles(text)
        if not sh:
            return False
        cnt = collections.Counter()
        for x in sh:
            for i in self.index.get(x, ()):
                cnt[i] += 1
        for i, c in cnt.items():
            if c / len(sh) >= thr or c / self.sizes[i] >= thr:
                self.dropped += 1
                return True
        return False


def user(t):
    return {"role": "user", "content": t}


def pq_rows(path, cols=None):
    return pq.read_table(path, columns=cols).to_pylist()


def effort_for(cat, rng):
    w = {"math": (0.1, 0.3, 0.6), "code": (0.2, 0.45, 0.35), "agentic": (0.35, 0.45, 0.2),
         "structured": (0.45, 0.35, 0.2), "summary": (0.4, 0.4, 0.2), "chat": (0.3, 0.35, 0.35)}[cat]
    return rng.choices(["low", "medium", "high"], weights=w)[0]


def max_tokens_for(cat, effort):
    base = {"low": 1536, "medium": 2048, "high": 3072}[effort]
    return base


def code_or_chat(text):
    return "code" if CODE_RE.search(text[:4000]) else ("structured" if STRUCT_RE.search(text[:600]) else "chat")


# ---------------------------------------------------------------- sources

def src_wildchat(dl, rng, quota):
    rows = []
    for p in sorted(glob.glob(f"{dl}/allenai__WildChat-1M/data/*.parquet")):
        rows += pq_rows(p, ["conversation_hash", "conversation", "language", "toxic", "redacted", "turn"])
    rng.shuffle(rows)
    per = collections.Counter()
    mj = 0
    for r in rows:
        lang = WILDCHAT_LANG.get(r["language"])
        if lang is None or r["toxic"] or r["redacted"] or per[lang] >= quota.get(lang, 0):
            continue
        conv = [m for m in r["conversation"] if m["role"] in ("user", "assistant")]
        if not conv or conv[0]["role"] != "user":
            continue
        first = conv[0]["content"] or ""
        if not (12 <= len(first) <= 8000):
            continue
        if MJ_RE.search(first):
            mj += 1
            if mj > 40:
                continue
        # 20 %: continue a multi-turn conversation at a later user turn (history from the original model)
        msgs = [user(first)]
        users = [i for i, m in enumerate(conv) if m["role"] == "user"]
        if len(users) >= 2 and rng.random() < 0.2:
            k = rng.choice(users[1:])
            hist = conv[: k + 1]
            if sum(len(m["content"] or "") for m in hist) <= 16000 and all(m["content"] for m in hist):
                msgs = [{"role": m["role"], "content": m["content"]} for m in hist]
        per[lang] += 1
        yield dict(source="wildchat", key=r["conversation_hash"], category=code_or_chat(msgs[-1]["content"]),
                   language=lang, messages=msgs)


def src_ultrachat(dl, rng, n):
    rows = pq_rows(glob.glob(f"{dl}/HuggingFaceH4__ultrachat_200k/data/test_sft-*.parquet")[0])
    rng.shuffle(rows)
    for r in rows[:n]:
        conv = r["messages"]
        msgs = [user(conv[0]["content"])]
        if len(conv) >= 4 and rng.random() < 0.15:
            k = rng.choice([i for i in range(2, len(conv)) if conv[i]["role"] == "user"] or [0])
            if k:
                msgs = [{"role": m["role"], "content": m["content"]} for m in conv[: k + 1]]
        yield dict(source="ultrachat", key=r["prompt_id"], category=code_or_chat(msgs[-1]["content"]),
                   language="en", messages=msgs)


def src_oasst2(dl, rng, quota):
    rows = []
    for p in glob.glob(f"{dl}/OpenAssistant__oasst2/data/*.parquet"):
        rows += pq_rows(p, ["message_id", "parent_id", "text", "role", "lang", "review_result", "deleted", "rank"])
    by_parent = collections.defaultdict(list)
    for r in rows:
        if r["parent_id"]:
            by_parent[r["parent_id"]].append(r)
    roots = [r for r in rows if r["parent_id"] is None and r["role"] == "prompter" and r["review_result"]
             and not r["deleted"]]
    rng.shuffle(roots)
    per = collections.Counter()
    for r in roots:
        lang = OASST_LANG.get(r["lang"], r["lang"])
        if per[lang] >= quota.get(lang, quota.get("*", 0)):
            continue
        msgs = [user(r["text"])]
        if rng.random() < 0.15:  # second user turn after the best-ranked reply
            reps = sorted([c for c in by_parent[r["message_id"]] if c["role"] == "assistant" and not c["deleted"]],
                          key=lambda c: (c["rank"] is None, c["rank"]))
            if reps:
                follow = [c for c in by_parent[reps[0]["message_id"]] if c["role"] == "prompter" and not c["deleted"]]
                if follow:
                    msgs = [user(r["text"]), {"role": "assistant", "content": reps[0]["text"]}, user(follow[0]["text"])]
        per[lang] += 1
        yield dict(source="oasst2", key=r["message_id"], category=code_or_chat(msgs[-1]["content"]),
                   language=lang, messages=msgs)


def src_oasst2ja(dl, rng, n):
    rows = [json.loads(l) for l in open(f"{dl}/llm-jp__oasst2-33k-ja/oasst2-33k-ja.jsonl")]
    rng.shuffle(rows)
    for i, r in enumerate(rows[:n]):
        t = r["conversations"][0]["content"]
        yield dict(source="oasst2ja", key=hashlib.md5(t.encode()).hexdigest()[:12], category=code_or_chat(t), language="ja", messages=[user(t)])


def src_dolly(dl, rng, n):
    rows = [json.loads(l) for l in open(f"{dl}/databricks__databricks-dolly-15k/databricks-dolly-15k.jsonl")]
    rng.shuffle(rows)
    for i, r in enumerate(rows[:n]):
        t = r["instruction"]
        if r.get("context"):
            t = r["context"].strip() + "\n\n" + t if rng.random() < 0.5 else t + "\n\n" + r["context"].strip()
        cat = {"summarization": "summary", "information_extraction": "structured"}.get(r["category"], "chat")
        yield dict(source="dolly", key=f"{r['category']}-{hashlib.md5(t.encode()).hexdigest()[:10]}",
                   category=cat, language="en", messages=[user(t)])


def src_code(dl, rng, n_magic, n_oss):
    rows = [json.loads(l) for l in open(f"{dl}/ise-uiuc__Magicoder-OSS-Instruct-75K/data-oss_instruct-decontaminated.jsonl")]
    rng.shuffle(rows)
    for r in rows[:n_magic]:
        yield dict(source="magicoder", key=str(r["index"]), category="code", language="en",
                   messages=[user(r["problem"].strip())])
    rows = pq_rows(glob.glob(f"{dl}/bigcode__self-oss-instruct-sc2-exec-filter-50k/data/*.parquet")[0],
                   ["id", "instruction"])
    rng.shuffle(rows)
    for r in rows[:n_oss]:
        yield dict(source="selfoss", key=str(r["id"]), category="code", language="en",
                   messages=[user(r["instruction"].strip())])


MATH_SUFFIX = ["", "", "", " Show your work.", " Explain each step briefly.", " Give only the final answer."]


def src_math(dl, rng, n_gsm, n_math, mgsm_per_lang):
    rows = pq_rows(glob.glob(f"{dl}/openai__gsm8k/main/train-*.parquet")[0])
    rng.shuffle(rows)
    for r in rows[:n_gsm]:
        t = r["question"] + rng.choice(MATH_SUFFIX)
        yield dict(source="gsm8k", key=hashlib.md5(r["question"].encode()).hexdigest()[:12], category="math",
                   language="en", messages=[user(t)])
    rows = []
    for p in sorted(glob.glob(f"{dl}/EleutherAI__hendrycks_math/*/train-*.parquet")):
        rows += pq_rows(p, ["problem", "level", "type"])
    rng.shuffle(rows)
    for r in rows[:n_math]:
        yield dict(source="math", key=hashlib.md5(r["problem"].encode()).hexdigest()[:12], category="math",
                   language="en", messages=[user(r["problem"])])
    for lang in ("de", "es", "fr", "ru", "zh", "ja"):
        p = glob.glob(f"{dl}/juletxara__mgsm/{lang}/test-*.parquet")
        if not p:
            continue
        rows = pq_rows(p[0], ["question"])
        rng.shuffle(rows)
        for r in rows[:mgsm_per_lang]:
            yield dict(source="mgsm", key=f"{lang}-{hashlib.md5(r['question'].encode()).hexdigest()[:10]}",
                       category="math", language=lang, messages=[user(r["question"])])


TOOLS_RE = re.compile(r"<tools>\s*(\[.*\])\s*</tools>", re.S)
CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
RESP_RE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.S)


def hermes_conv(ex):
    """hermes function-calling conversation -> (tools, OpenAI messages ending at a user or tool turn)."""
    conv = ex["conversations"]
    m = TOOLS_RE.search(conv[0]["value"]) if conv and conv[0]["from"] == "system" else None
    if not m:
        return None
    try:
        tools = json.loads(m.group(1))
    except ValueError:
        return None
    tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
    for t in tools:
        f = t["function"]
        if not isinstance(f.get("name"), str):
            return None
    msgs, cut_points, ncall = [], [], 0
    for c in conv[1:]:
        if c["from"] == "human":
            msgs.append(user(c["value"]))
            cut_points.append(len(msgs))
        elif c["from"] == "gpt":
            calls = []
            for js in CALL_RE.findall(c["value"]):
                try:
                    d = json.loads(js)
                except ValueError:
                    try:
                        import ast
                        d = ast.literal_eval(js)
                    except Exception:  # noqa: BLE001
                        return None
                if not isinstance(d, dict) or not isinstance(d.get("arguments", {}), dict):
                    return None
                ncall += 1
                calls.append({"id": f"call_{ncall}", "type": "function",
                              "function": {"name": d.get("name"), "arguments": json.dumps(d.get("arguments", {}))}})
            if calls:
                msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
            else:
                msgs.append({"role": "assistant", "content": c["value"]})
        elif c["from"] == "tool":
            prev = msgs[-1] if msgs else None
            if not prev or not prev.get("tool_calls"):
                return None
            resps = RESP_RE.findall(c["value"]) or [c["value"]]
            for i, call in enumerate(prev["tool_calls"]):
                body = resps[i] if i < len(resps) else resps[-1]
                msgs.append({"role": "tool", "tool_call_id": call["id"], "content": body})
            cut_points.append(len(msgs))
    if not cut_points:
        return None
    return tools, msgs, cut_points


AGENT_SYSTEM = ("You are a coding agent working in the user's repository. Use the tools to inspect and change files "
                "and to run commands. Think before acting, keep changes minimal, and explain what you did at the end.")
AGENT_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                    "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command in the repository root and return stdout/stderr.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                    "timeout_s": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "search_code", "description": "Search the repository with a regular expression.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}},
                    "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web and return the top results.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def src_agentic(dl, rng, n_single, n_multi, n_glaive, n_coding_agent):
    D = f"{dl}/NousResearch__hermes-function-calling-v1"
    for fname, n, multi in (("func-calling-singleturn.json", n_single, False), ("func-calling.json", n_multi, True),
                            ("glaive-function-calling-5k.json", n_glaive, True)):
        rows = json.load(open(f"{D}/{fname}"))
        rng.shuffle(rows)
        k = 0
        for ex in rows:
            if k >= n:
                break
            got = hermes_conv(ex)
            if not got:
                continue
            tools, msgs, cuts = got
            cut = rng.choice(cuts) if multi else cuts[0]
            msgs = msgs[:cut]
            if not msgs or msgs[-1]["role"] not in ("user", "tool"):
                continue
            k += 1
            yield dict(source="hermes", key=f"{fname.split('.')[0]}-{ex['id']}", category="agentic", language="en",
                       messages=msgs, tools=tools)
    # coding-agent style: first turn of a code task with a generic repository tool set
    rows = pq_rows(glob.glob(f"{dl}/bigcode__self-oss-instruct-sc2-exec-filter-50k/data/*.parquet")[0],
                   ["id", "instruction"])
    rng2 = random.Random(SEED + 7)
    rng2.shuffle(rows)
    for r in rows[-n_coding_agent:]:  # the tail of the shuffled list; the code source uses the head
        task = r["instruction"].strip()
        wrap = rng.choice(["{t}", "In our repo, {t} Put it in a sensible module and add tests.",
                           "{t}\n\nCheck the existing code first, then implement it and run the tests.",
                           "Task: {t}\nWork in the repository; when finished, summarise the change."])
        yield dict(source="selfoss", key=f"agent-{r['id']}", category="agentic", language="en",
                   messages=[{"role": "system", "content": AGENT_SYSTEM}, user(wrap.format(t=task))],
                   tools=AGENT_TOOLS)


def src_json_mode(dl, rng, n_single, n_agentic):
    D = f"{dl}/NousResearch__hermes-function-calling-v1"
    for fname, n in (("json-mode-singleturn.json", n_single), ("json-mode-agentic.json", n_agentic)):
        rows = json.load(open(f"{D}/{fname}"))
        rng.shuffle(rows)
        for ex in rows[:n]:
            conv = ex["conversations"]
            if len(conv) < 2 or conv[0]["from"] != "system" or conv[1]["from"] != "human":
                continue
            yield dict(source="hermes", key=f"{fname.split('.')[0]}-{ex['id']}", category="structured",
                       language="en", messages=[{"role": "system", "content": conv[0]["value"]}, user(conv[1]["value"])])


SUMMARY_EN = [
    "Summarize the following article in one paragraph.",
    "Give me the five key points of this article as bullet points.",
    "Write a headline and a three-sentence summary for this news story.",
    "What are the main claims in this article, and what evidence supports each? Be concise.",
    "Explain this story to someone who has no background on the topic.",
    "Write a short newsletter blurb (about 120 words) based on this article.",
    "List the people and organizations mentioned and their role in the story.",
    "Turn this article into a short Q&A with 5 questions and answers.",
    "TL;DR this.",
    "Write a neutral summary, then note anything that seems one-sided or missing.",
]
STRUCT_EN = [
    "Extract every person, organization, place and date mentioned in the text below. Return JSON with keys "
    "\"people\", \"organizations\", \"places\", \"dates\" (arrays of strings). Return only the JSON.",
    "Convert the key facts of the text below into a markdown table with columns Fact, Value, Source sentence.",
    "Return a YAML document describing the text below with fields: title, topic, summary, key_entities (list), "
    "sentiment.",
    "Produce a JSON array of events in the text below, each with fields \"when\", \"who\", \"what\". JSON only.",
    "Write CSV (with a header row) listing every number in the text below with its unit and what it refers to.",
    "Create a JSON object with \"headline\", \"summary\" (max 50 words), \"tags\" (5 strings) and \"category\" "
    "for the text below.",
]
SUMMARY_L = {
    "pl": ["Streść poniższy artykuł w jednym akapicie.", "Wypisz pięć najważniejszych informacji z tego tekstu.",
           "Wyjaśnij ten temat uczniowi szkoły średniej na podstawie poniższego tekstu.",
           "Przygotuj 5 pytań quizowych z odpowiedziami na podstawie tego tekstu.",
           "Napisz krótką notatkę (ok. 150 słów) na podstawie tego artykułu."],
    "de": ["Fasse den folgenden Artikel in einem Absatz zusammen.", "Nenne die fünf wichtigsten Punkte als Liste.",
           "Erkläre das Thema einem Schüler anhand des folgenden Textes.",
           "Erstelle 5 Quizfragen mit Antworten zu diesem Text."],
    "es": ["Resume el siguiente artículo en un párrafo.", "Enumera los cinco puntos clave de este texto.",
           "Explica este tema a un estudiante usando el siguiente texto.",
           "Escribe 5 preguntas de examen con respuestas sobre este texto."],
    "fr": ["Résume l'article suivant en un paragraphe.", "Donne les cinq points clés de ce texte sous forme de liste.",
           "Explique ce sujet à un lycéen à partir du texte suivant."],
    "zh": ["请用一段话总结下面的文章。", "列出这篇文章的五个要点。", "根据下面的文本，向中学生解释这个主题。",
           "根据这段文字出5道问答题并给出答案。"],
    "ja": ["次の記事を一段落で要約してください。", "この文章の重要なポイントを5つ挙げてください。",
           "次の文章をもとに、このテーマを高校生に説明してください。", "この文章から5問のクイズと答えを作ってください。"],
    "ru": ["Кратко перескажи следующую статью в одном абзаце.", "Перечисли пять главных фактов из этого текста.",
           "Объясни эту тему школьнику на основе следующего текста."],
    "en": SUMMARY_EN[:5],
}


def src_long(dl, rng, n_cnn, n_cnn_struct, wiki_quota):
    rows = pq_rows(glob.glob(f"{dl}/abisee__cnn_dailymail/3.0.0/test-*.parquet")[0], ["id", "article"])
    rng.shuffle(rows)
    for i, r in enumerate(rows[: n_cnn + n_cnn_struct]):
        art = r["article"][:14000]
        if i < n_cnn:
            ins, cat = rng.choice(SUMMARY_EN), "summary"
        else:
            ins, cat = rng.choice(STRUCT_EN), "structured"
        t = f"{ins}\n\n{art}" if rng.random() < 0.7 else f"{art}\n\n{ins}"
        yield dict(source="cnn", key=r["id"], category=cat, language="en", messages=[user(t)])
    per = collections.Counter()
    for line in open(f"{dl}/wiki/wiki.jsonl"):
        r = json.loads(line)
        lang = r["lang"]
        if per[lang] >= wiki_quota.get(lang, 0):
            continue
        per[lang] += 1
        text = f"{r['title']}\n\n{r['text'][:12000]}"
        if lang == "en" and rng.random() < 0.4:
            ins, cat = rng.choice(STRUCT_EN), "structured"
        else:
            ins, cat = rng.choice(SUMMARY_L[lang]), "summary"
        t = f"{ins}\n\n{text}" if rng.random() < 0.7 else f"{text}\n\n{ins}"
        yield dict(source="wiki", key=f"{lang}-{r['id']}", category=cat, language=lang, messages=[user(t)])


def src_aya(dl, rng, quota):
    rows = pq_rows(glob.glob(f"{dl}/CohereLabs__aya_dataset/data/train-*.parquet")[0], ["inputs", "language"])
    rng.shuffle(rows)
    per = collections.Counter()
    for r in rows:
        lang = AYA_LANG.get(r["language"])
        if lang is None or lang == "en" or per[lang] >= quota.get(lang, 0) or len(r["inputs"]) < 15:
            continue
        per[lang] += 1
        yield dict(source="aya", key=f"{lang}-{hashlib.md5(r['inputs'].encode()).hexdigest()[:12]}",
                   category=code_or_chat(r["inputs"]), language=lang, messages=[user(r["inputs"])])


def src_respond_in(pool, rng, quota):
    """English prompts from other sources with an instruction (in the target language) to answer in it."""
    cands = [p for p in pool if p["language"] == "en" and len(p["messages"]) == 1 and not p.get("tools")
             and p["source"] in ("dolly", "ultrachat", "gsm8k", "oasst2", "magicoder", "wildchat")
             and len(p["messages"][0]["content"]) < 4000]
    rng.shuffle(cands)
    it = iter(cands)
    for lang, n in quota.items():
        for _ in range(n):
            p = next(it)
            t = p["messages"][0]["content"].rstrip() + "\n\n" + RESPOND_IN[lang]
            yield dict(source=p["source"], key=f"respond-{lang}-{p['key']}", category=p["category"], language=lang,
                       messages=[user(t)])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dl", required=True)
    ap.add_argument("--speedbench", required=True, help="speedbench repo (reads prompts/*.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    a = ap.parse_args()
    rng = random.Random(SEED)
    bench = BenchFilter(a.speedbench)

    gens = [
        src_wildchat(a.dl, rng, {"en": 6500, "zh": 500, "ru": 300, "fr": 220, "es": 220, "de": 220, "pl": 400,
                                 "ja": 180, "pt": 150, "it": 150, "ko": 80, "uk": 60}),
        src_ultrachat(a.dl, rng, 3800),
        src_oasst2(a.dl, rng, {"en": 2600, "es": 600, "ru": 300, "zh": 300, "de": 420, "fr": 300, "pt": 150,
                               "ca": 60, "it": 120, "uk": 80, "ja": 80, "pl": 54, "*": 0}),
        src_oasst2ja(a.dl, rng, 250),
        src_dolly(a.dl, rng, 2500),
        src_code(a.dl, rng, 3500, 2800),
        src_math(a.dl, rng, 2600, 2600, 90),
        src_agentic(a.dl, rng, 1500, 1500, 2000, 1500),
        src_json_mode(a.dl, rng, 1241, 1342),
        src_long(a.dl, rng, 1600, 900, {"en": 600, "pl": 354, "de": 200, "es": 200, "zh": 131, "ja": 200,
                                        "fr": 120, "ru": 87}),
        src_aya(a.dl, rng, {"pl": 900, "de": 241, "ja": 300, "zh": 300, "es": 200, "fr": 150, "it": 100,
                            "ko": 80, "uk": 80, "pt": 150, "ru": 100}),
    ]
    out, seen = [], set()
    drops = collections.Counter()
    dropped_examples = []

    def admit(rec):
        last = rec["messages"][-1]["content"] or ""
        allt = "\n".join((m.get("content") or "") for m in rec["messages"])
        k = norm(last)[:600]
        if k in seen:
            drops["dup"] += 1
            return
        if THINK_LOAD_RE.search(allt):
            drops["think_load_shape"] += 1
            return
        if bench.hit(last) or (len(rec["messages"]) > 1 and bench.hit(allt)):
            drops["speedbench_neardup"] += 1
            dropped_examples.append(last[:160])
            return
        seen.add(k)
        out.append(rec)

    for g in gens:
        for rec in g:
            admit(rec)
    for rec in src_respond_in(list(out), rng, {"pl": 600, "de": 200, "es": 150, "fr": 100, "zh": 120, "ja": 120,
                                               "it": 80, "ru": 80}):
        admit(rec)

    rng.shuffle(out)
    ids, used_ids = collections.Counter(), set()
    with open(a.out, "w") as f:
        for rec in out:
            eff = effort_for(rec["category"], rng)
            src = rec["source"]
            ids[src] += 1
            rid, salt = f"{src}-{hashlib.sha1(rec['key'].encode()).hexdigest()[:10]}", 0
            while rid in used_ids:
                salt += 1
                salted = "%s#%d" % (rec["key"], salt)
                rid = f"{src}-{hashlib.sha1(salted.encode()).hexdigest()[:10]}"
            used_ids.add(rid)
            row = {"id": rid,
                   "source": src, "license": LIC[src], "category": rec["category"], "language": rec["language"],
                   "messages": rec["messages"]}
            if rec.get("tools"):
                row["tools"] = rec["tools"]
            row["reasoning_effort"] = eff
            row["max_tokens"] = max_tokens_for(rec["category"], eff)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(out)
    st = {"prompts": n, "drops": dict(drops), "speedbench_neardup_examples": dropped_examples[:20], "speedbench_texts_indexed": bench.n_bench,
          "by_source": dict(collections.Counter(r["source"] for r in out).most_common()),
          "by_category": {k: f"{v} ({100 * v / n:.1f} %)" for k, v in collections.Counter(r["category"] for r in out).most_common()},
          "by_language": {k: f"{v} ({100 * v / n:.1f} %)" for k, v in collections.Counter(r["language"] for r in out).most_common()},
          "multilingual_share": sum(r["language"] != "en" for r in out) / n,
          "multi_turn": sum(len(r["messages"]) > 2 or (len(r["messages"]) == 2 and r["messages"][0]["role"] != "system") for r in out),
          "with_tools": sum(bool(r.get("tools")) for r in out)}
    print(json.dumps(st, indent=1, ensure_ascii=False))
    if a.stats:
        json.dump(st, open(a.stats, "w"), indent=1, ensure_ascii=False)
    # id uniqueness
    with open(a.out) as f:
        all_ids = [json.loads(l)["id"] for l in f]
    if len(set(all_ids)) != len(all_ids):
        print("ERROR: duplicate ids", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
