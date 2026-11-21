#!/usr/bin/env python3
"""Generate a trace corpus of the model's own outputs for the draft hot-vocabulary list.

Python 3 standard library only. Runs on the Spark against the local SGLang server (OpenAI chat API).

  python3 gen_corpus.py --prompts prompts.jsonl --out traces.jsonl [--concurrency 28] \
      [--target-tokens 4000000] [--url http://127.0.0.1:8000] [--model inkling-2-mini]

Resumable: ids already present in --out are skipped, so re-running the same command continues where it stopped
(also after a crash or a server restart). Prompts are taken in file order (the file is pre-shuffled, so any
prefix has the corpus mix). Stops issuing new requests once the cumulative completion tokens in --out reach
--target-tokens (in-flight requests still finish). Ctrl-C / SIGTERM: stop issuing, let in-flight finish;
a second Ctrl-C exits at once (a torn last line is ignored on the next start).

Per prompt (deterministic from the id, so a resumed run makes the same choices):
  * reasoning_effort: the prompt's suggestion with probability --follow-effort (0.7), else uniform low/medium/high;
  * temperature: 0 with probability --greedy-share (0.3), else 0.8 or 1.0 (half each); top_p 1.0;
  * max_tokens: the prompt's suggestion (1536/2048/3072 by effort), clamped to [--min-max-tokens, --max-max-tokens].

Requests are non-streaming with "return_token_ids": true, so each record carries the exact sampled ids
(response_token_ids) next to reasoning_content and content. If the server rejects that field, the script drops it
and falls back to text only (the coverage tool then re-tokenizes the text).

Output record (one JSON line): id, source, category, language, effort, temperature, max_tokens, finish_reason,
reasoning_content, content, tool_calls, response_token_ids, usage, elapsed_s, ts.
Failed requests go to <out>.errors.jsonl and are retried on the next run (at most --max-attempts per id).
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request


def h01(s, salt):
    """Deterministic uniform [0, 1) from a string."""
    return int(hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()[:12], 16) / float(1 << 48)


def choose(rec, a):
    pid = rec["id"]
    eff = rec.get("reasoning_effort") or "medium"
    if h01(pid, "follow") >= a.follow_effort:
        eff = ("low", "medium", "high")[int(h01(pid, "effort") * 3)]
    if h01(pid, "greedy") < a.greedy_share:
        temp = 0.0
    else:
        temp = 0.8 if h01(pid, "temp") < 0.5 else 1.0
    mt = int(rec.get("max_tokens") or {"low": 1536, "medium": 2048, "high": 3072}[eff])
    mt = max(a.min_max_tokens, min(a.max_max_tokens, mt))
    return eff, temp, mt


def load_done(path):
    """(done ids, cumulative completion tokens, records) from an existing output file; tolerates a torn line."""
    done, tok, n = set(), 0, 0
    if os.path.exists(path):
        with open(path, "rb") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                done.add(r["id"])
                tok += int((r.get("usage") or {}).get("completion_tokens") or 0)
                n += 1
    return done, tok, n


def load_attempts(path):
    att = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("counts", True):
                    att[r["id"]] = att.get(r["id"], 0) + 1
    return att


def http_json(url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def make_body(rec, eff, temp, mt, model, use_ids):
    b = {"model": model, "messages": rec["messages"], "max_tokens": mt, "temperature": temp,
         "top_p": 1.0, "stream": False, "chat_template_kwargs": {"reasoning_effort": eff}}
    if temp > 0:
        b["seed"] = int(h01(rec["id"], "seed") * 2**31)
    if rec.get("tools"):
        b["tools"] = rec["tools"]
    if use_ids:
        b["return_token_ids"] = True
    return b


class Runner:
    def __init__(self, a, prompts, done, tok0, n0, attempts):
        self.a = a
        self.queue = iter(prompts)
        self.lock = threading.Lock()
        self.out = open(a.out, "a", encoding="utf-8")
        self.err = open(a.out + ".errors.jsonl", "a", encoding="utf-8")
        self.done, self.attempts = done, attempts
        self.tok_total = tok0
        self.n_total = n0
        self.tok_run = 0
        self.n_run = 0
        self.n_err = 0
        self.n_length = 0
        self.reason_tok = 0
        self.inflight = 0
        self.stop = threading.Event()
        self.t0 = time.time()
        self.use_ids = not a.no_token_ids
        self.server_down_since = None
        self.consec_conn_err = 0

    # -------------------------------------------------------------- request
    def call(self, rec):
        eff, temp, mt = choose(rec, self.a)
        url = self.a.url.rstrip("/") + "/v1/chat/completions"
        t = time.time()
        sent_ids = self.use_ids
        try:
            d = http_json(url, make_body(rec, eff, temp, mt, self.a.model, sent_ids), timeout=self.a.timeout)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            if e.code == 400 and sent_ids and "token_ids" in msg:
                with self.lock:
                    if self.use_ids:
                        self.use_ids = False
                        print("server rejected return_token_ids; continuing with text only", flush=True)
                return self.call(rec)
            return None, {"http": e.code, "msg": msg}, True
        except (urllib.error.URLError, ConnectionError, OSError) as e:  # refused, reset, timeout
            is_timeout = "timed out" in repr(e)
            return None, {"conn": repr(e)[:300]}, is_timeout
        ch = (d.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        rec_out = {
            "id": rec["id"], "source": rec.get("source"), "category": rec.get("category"),
            "language": rec.get("language"), "effort": eff, "temperature": temp, "max_tokens": mt,
            "finish_reason": ch.get("finish_reason"),
            "reasoning_content": m.get("reasoning_content") or "",
            "content": m.get("content") or "",
            "tool_calls": m.get("tool_calls") or None,
            "response_token_ids": ch.get("response_token_ids"),
            "usage": d.get("usage"),
            "elapsed_s": round(time.time() - t, 2),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return rec_out, None, False

    # -------------------------------------------------------------- loop
    def next_prompt(self):
        with self.lock:
            if self.stop.is_set():
                return None
            if self.a.target_tokens and self.tok_total >= self.a.target_tokens:
                self.stop.set()
                return None
            if self.a.limit and self.n_run + self.n_err + self.inflight >= self.a.limit:
                return None
            for rec in self.queue:
                if rec["id"] in self.done or self.attempts.get(rec["id"], 0) >= self.a.max_attempts:
                    continue
                self.inflight += 1
                return rec
            return None

    def wait_server(self):
        """Block (without consuming prompts) until /v1/models answers again."""
        url = self.a.url.rstrip("/") + "/v1/models"
        while not self.stop.is_set():
            try:
                http_json(url, timeout=10)
                return
            except Exception:  # noqa: BLE001
                time.sleep(15)

    def worker(self):
        while True:
            rec = self.next_prompt()
            if rec is None:
                return
            while True:
                out, err, counts = self.call(rec)
                if out is None and err and "conn" in err and not counts:
                    # server down / restarting: do not burn the prompt, wait and retry it
                    with self.lock:
                        self.consec_conn_err += 1
                        if self.server_down_since is None:
                            self.server_down_since = time.time()
                            print(f"server unreachable ({err['conn'][:120]}); waiting", flush=True)
                    self.wait_server()
                    if self.stop.is_set():
                        err = None  # stopped while waiting: not an attempt, the prompt stays to do
                        break
                    continue
                break
            with self.lock:
                self.inflight -= 1
                if self.server_down_since is not None and out is not None:
                    print(f"server back after {time.time() - self.server_down_since:.0f} s", flush=True)
                    self.server_down_since = None
                    self.consec_conn_err = 0
                if out is not None:
                    self.out.write(json.dumps(out, ensure_ascii=False) + "\n")
                    self.out.flush()
                    ct = int((out.get("usage") or {}).get("completion_tokens") or 0)
                    rt = int(((out.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
                             or (out.get("usage") or {}).get("reasoning_tokens") or 0)
                    self.done.add(out["id"])
                    self.tok_total += ct
                    self.tok_run += ct
                    self.reason_tok += rt
                    self.n_run += 1
                    self.n_total += 1
                    self.n_length += out.get("finish_reason") == "length"
                elif err is not None:
                    self.n_err += 1
                    self.attempts[rec["id"]] = self.attempts.get(rec["id"], 0) + 1
                    self.err.write(json.dumps({"id": rec["id"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                               "counts": True, **err}) + "\n")
                    self.err.flush()

    def progress(self):
        last_t, last_tok = time.time(), self.tok_run
        while any(t.is_alive() for t in self.threads):
            time.sleep(1)
            now = time.time()
            if now - last_t < self.a.progress_s:
                continue
            with self.lock:
                el = now - self.t0
                rate_run = self.tok_run / el * 3600 if el > 0 else 0
                rate_win = (self.tok_run - last_tok) / (now - last_t) * 3600
                left = max(0, (self.a.target_tokens or 0) - self.tok_total)
                eta = left / rate_win * 60 if rate_win > 0 and self.a.target_tokens else float("nan")
                print(f"[{time.strftime('%H:%M:%S')}] done {self.n_total} (+{self.n_run} this run, {self.n_err} err, "
                      f"{self.inflight} in flight) | completion tok total {self.tok_total / 1e6:.3f} M "
                      f"(+{self.tok_run / 1e6:.3f} M) | {rate_win / 1e6:.2f} M tok/h now, {rate_run / 1e6:.2f} M tok/h run "
                      f"({rate_win / 3600:.0f} tok/s) | reasoning {100 * self.reason_tok / max(1, self.tok_run):.0f} % | "
                      f"finish=length {100 * self.n_length / max(1, self.n_run):.0f} % | ETA {eta:.0f} min",
                      flush=True)
                last_t, last_tok = now, self.tok_run

    def run(self):
        self.threads = [threading.Thread(target=self.worker, daemon=True) for _ in range(self.a.concurrency)]
        for t in self.threads:
            t.start()
        self.progress()
        for t in self.threads:
            t.join()
        el = time.time() - self.t0
        print(f"finished: {self.n_run} records, {self.tok_run} completion tokens in {el / 60:.1f} min "
              f"({self.tok_run / max(el, 1e-9) * 3600 / 1e6:.2f} M tok/h); total in {self.a.out}: {self.n_total} records, "
              f"{self.tok_total} tokens; errors this run {self.n_err}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--concurrency", type=int, default=28)
    ap.add_argument("--target-tokens", type=int, default=4_000_000, help="cumulative completion tokens; 0 = no limit")
    ap.add_argument("--limit", type=int, default=0, help="at most this many requests this run (smoke test)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per request, seconds")
    ap.add_argument("--follow-effort", type=float, default=0.7)
    ap.add_argument("--greedy-share", type=float, default=0.3)
    ap.add_argument("--min-max-tokens", type=int, default=1024)
    ap.add_argument("--max-max-tokens", type=int, default=3072)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--progress-s", type=float, default=60.0)
    ap.add_argument("--no-token-ids", action="store_true", help="do not ask for response_token_ids")
    ap.add_argument("--category", default="", help="comma list: only these categories")
    ap.add_argument("--language", default="", help="comma list: only these answer languages")
    ap.add_argument("--dry-run", action="store_true", help="print the first 3 request bodies and the mix, then exit")
    a = ap.parse_args()

    prompts = []
    with open(a.prompts, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    if a.category:
        keep = set(a.category.split(","))
        prompts = [p for p in prompts if p.get("category") in keep]
    if a.language:
        keep = set(a.language.split(","))
        prompts = [p for p in prompts if p.get("language") in keep]
    done, tok0, n0 = load_done(a.out)
    attempts = load_attempts(a.out + ".errors.jsonl")
    todo = [p for p in prompts if p["id"] not in done]
    print(f"prompts {len(prompts)}, already done {len(done)} ({tok0} completion tokens), to do {len(todo)}, "
          f"target {a.target_tokens} tokens, concurrency {a.concurrency}", flush=True)

    if a.dry_run:
        for p in todo[:3]:
            print(json.dumps(make_body(p, *choose(p, a), a.model, not a.no_token_ids), ensure_ascii=False)[:1500])
        mix = {}
        for p in todo[:2000]:
            eff, temp, mt = choose(p, a)
            mix[(eff, temp)] = mix.get((eff, temp), 0) + 1
        print("effort/temperature mix of the next 2000:", sorted(mix.items()))
        return 0

    try:
        models = http_json(a.url.rstrip("/") + "/v1/models", timeout=10)
        ids = [m.get("id") for m in models.get("data", [])]
        print(f"server models: {ids}", flush=True)
        if ids and a.model not in ids:
            print(f"warning: --model {a.model} not in {ids}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"warning: {a.url}/v1/models not reachable yet ({e!r}); workers will wait", flush=True)

    runner = Runner(a, todo, done, tok0, n0, attempts)
    hits = {"n": 0}

    def on_signal(signum, frame):  # noqa: ARG001
        hits["n"] += 1
        runner.stop.set()
        if hits["n"] == 1:
            print("stop requested: no new requests; waiting for in-flight ones (signal again to exit now)", flush=True)
        else:
            runner.out.flush()
            os._exit(130)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
