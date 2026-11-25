"""Label prepared messages with the teacher model (Gemini) and keep only valid outputs.

One message per request (mirrors production usage, so per-request cost/latency logged here
feeds the teacher's row of the results table).

Features: concurrent workers + global rate limit, retries with exponential backoff on API
errors AND on invalid outputs, schema/consistency validation, resumable (skips ids already
in the output file), per-call token + latency logging.

Examples (from repo root):
  python -m labeling.label --split train --n 50 --stratify --out data/labeled/pilot.jsonl
  python -m labeling.label --split train
  python -m labeling.label --split train --dry-run      # print prompt, no API call
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from labeling.splits import PREPARED, ROOT, load_jsonl
from labeling.validate import SCHEMA, TAXONOMY, parse_and_validate

HERE = Path(__file__).resolve().parent
PROMPT_VERSION = "v1"


def build_system_prompt() -> str:
    lines = [f"- {intent} [{v['category']}]: {v['description']}" for intent, v in TAXONOMY.items()]
    return (HERE / "prompt.md").read_text(encoding="utf-8").replace("<<INTENT_LIST>>", "\n".join(lines))


def format_user_message(text: str) -> str:
    return f"Message: {text}"


def api_schema() -> dict:
    """schema.json minus JSON-Schema meta keys the Gemini API does not need."""
    return {k: v for k, v in SCHEMA.items() if k not in ("$schema", "title")}


class RateLimiter:
    """Spaces request *starts* at least 60/rpm seconds apart across all threads."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)


def call_teacher(client, model: str, system_prompt: str, text: str, thinking_budget: int | None):
    """One API call. Returns (raw_text, usage_dict, latency_s). Raises on API error."""
    from google.genai import types

    cfg = dict(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_json_schema=api_schema(),
        temperature=0.0,
    )
    if thinking_budget is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model, contents=format_user_message(text), config=types.GenerateContentConfig(**cfg)
    )
    latency = time.perf_counter() - t0
    um = resp.usage_metadata
    usage = {
        "prompt_tokens": getattr(um, "prompt_token_count", None),
        "output_tokens": getattr(um, "candidates_token_count", None),
        "thoughts_tokens": getattr(um, "thoughts_token_count", None),
        "cached_tokens": getattr(um, "cached_content_token_count", None),
    }
    return resp.text, usage, latency


def label_one(row, *, client, model, system_prompt, limiter, max_attempts, thinking_budget):
    """Returns ("ok", record) or ("reject", record)."""
    errors_seen, usages = [], []
    for attempt in range(1, max_attempts + 1):
        limiter.wait()
        try:
            raw, usage, latency = call_teacher(client, model, system_prompt, row["text"], thinking_budget)
        except Exception as e:  # network / 429 / 5xx: back off and retry
            errors_seen.append(f"api: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(min(60, 2 ** attempt) + random.random())
            continue
        usages.append(usage)
        obj, errs = parse_and_validate(raw)
        if not errs:
            return "ok", {
                "id": row["id"],
                "text": row["text"],
                "label": obj,
                "teacher": model,
                "prompt_version": PROMPT_VERSION,
                "attempts": attempt,
                "latency_s": round(latency, 3),
                "usage": usage,          # of the successful call
                "usage_all_attempts": usages,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        errors_seen.append(f"invalid: {errs[:3]} raw={str(raw)[:200]!r}")
    return "reject", {"id": row["id"], "text": row["text"], "teacher": model,
                      "errors": errors_seen, "usage_all_attempts": usages}


def stratified_take(rows: list[dict], n: int) -> list[dict]:
    """Round-robin over bitext intents (rows are already shuffled) -> balanced pilot sample."""
    by_intent = defaultdict(list)
    for r in rows:
        by_intent[r["bitext_intent"]].append(r)
    queues = [by_intent[k] for k in sorted(by_intent)]
    out, i = [], 0
    while len(out) < n and any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
        i += 1
    return out


def p95(xs):
    return statistics.quantiles(xs, n=20)[-1] if len(xs) >= 20 else max(xs)


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--n", type=int, default=None, help="label only the first N rows")
    ap.add_argument("--stratify", action="store_true", help="with --n: balance across intents")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=float, default=float(os.getenv("TEACHER_RPM", "60")))
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="tokens of model 'thinking'; 0 disables (cheaper). Default: model default")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_jsonl(PREPARED / f"{args.split}.jsonl")
    if args.n:
        rows = stratified_take(rows, args.n) if args.stratify else rows[: args.n]
    out = args.out or ROOT / "data" / "labeled" / f"{args.split}.jsonl"
    rejects_path = out.with_suffix(".rejects.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = build_system_prompt()
    if args.dry_run:
        print(system_prompt, "\n\n---- user ----\n", format_user_message(rows[0]["text"]))
        print(f"\n[dry-run] would label {len(rows)} rows -> {out} with {args.model}")
        return

    done = {r["id"] for r in load_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already labeled, {len(todo)} to go -> {out}")
    if not todo:
        return

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set (put it in .env)")
    from google import genai
    client = genai.Client(api_key=key)
    limiter = RateLimiter(args.rpm)

    n_ok, n_rej, lat, ptoks, otoks = 0, 0, [], [], []
    with open(out, "a", encoding="utf-8", newline="\n") as f_ok, \
         open(rejects_path, "a", encoding="utf-8", newline="\n") as f_rej, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(label_one, r, client=client, model=args.model, system_prompt=system_prompt,
                            limiter=limiter, max_attempts=args.max_attempts,
                            thinking_budget=args.thinking_budget) for r in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="labeling"):
            status, rec = fut.result()
            if status == "ok":
                f_ok.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_ok.flush()
                n_ok += 1
                lat.append(rec["latency_s"])
                ptoks.append(rec["usage"]["prompt_tokens"] or 0)
                otoks.append((rec["usage"]["output_tokens"] or 0) + (rec["usage"]["thoughts_tokens"] or 0))
            else:
                f_rej.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_rej.flush()
                n_rej += 1

    # compact rejects: drop ids that have since been labeled successfully, keep latest per id
    labeled = {r["id"] for r in load_jsonl(out)}
    latest = {r["id"]: r for r in load_jsonl(rejects_path) if r["id"] not in labeled}
    with open(rejects_path, "w", encoding="utf-8", newline="\n") as f:
        for r in latest.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "model": args.model, "ok": n_ok, "rejected": n_rej,
        "valid_rate_this_run": round(n_ok / max(1, n_ok + n_rej), 4),
        "latency_s_mean": round(statistics.mean(lat), 3) if lat else None,
        "latency_s_p95": round(p95(lat), 3) if lat else None,
        "prompt_tokens_mean": round(statistics.mean(ptoks), 1) if ptoks else None,
        "output_tokens_mean_incl_thinking": round(statistics.mean(otoks), 1) if otoks else None,
    }, indent=2))


if __name__ == "__main__":
    main()
