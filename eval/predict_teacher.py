"""Teacher row: run the teacher API on the test set, ONE message per request (production-like),
and save raw outputs + per-request latency and tokens for eval/eval.py.

Same model, prompt (v3, single-message form) and structured-output schema as labeling.
Invalid outputs are recorded as-is and NOT retried (they are part of what we measure);
only API/network errors are retried. Requests are sent one at a time (latency = one request).
Resumable: --max-requests caps API calls per run (daily quota); re-run to finish.

  python -m eval.predict_teacher --max-requests 115
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

from labeling.label import (PROMPT_VERSION, RateLimiter, build_system_prompt, call_teacher,
                            format_single, label_schema)
from labeling.splits import ROOT, load_jsonl

OUT = ROOT / "eval" / "predictions" / "teacher.jsonl"


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "gemini-3.5-flash-lite"))
    ap.add_argument("--rpm", type=float, default=10)
    ap.add_argument("--max-requests", type=int, required=True, help="API calls this run, incl. retries")
    ap.add_argument("--api-attempts", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--key-env", default="GEMINI_API_KEY",
                    help="name of the .env variable holding the API key (e.g. GEMINI_API_KEY_FALLBACK)")
    args = ap.parse_args()

    test = [{"id": r["id"], "text": r["text"]} for r in load_jsonl(ROOT / "data" / "final" / "test.jsonl")]
    done = {r["id"] for r in load_jsonl(args.out)} if args.out.exists() else set()
    todo = [r for r in test if r["id"] not in done]
    print(f"{len(test)} messages, {len(done)} done, {len(todo)} to go -> {args.out} "
          f"[{args.model}, {args.rpm} rpm, max {args.max_requests} requests]", flush=True)
    if not todo:
        return

    from google import genai
    key = os.getenv(args.key_env)
    if not key:
        raise SystemExit(f"{args.key_env} is not set in .env")
    client = genai.Client(api_key=key)
    limiter = RateLimiter(args.rpm)
    system, schema = build_system_prompt(batch=False), label_schema()
    n_req, n_ok, failed = 0, 0, 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8", newline="\n") as f:
        for r in todo:
            if n_req >= args.max_requests:
                break
            for attempt in range(1, args.api_attempts + 1):
                if n_req >= args.max_requests:
                    break
                limiter.wait()
                n_req += 1
                try:
                    raw, usage, latency = call_teacher(client, args.model, system, format_single(r["text"]),
                                                       schema, thinking_budget=None)
                except Exception as e:
                    print(f"  API error on {r['id']} (attempt {attempt}): {type(e).__name__}: {str(e)[:120]}", flush=True)
                    time.sleep(min(60, 5 * 2 ** attempt) + random.random())
                    continue
                f.write(json.dumps({
                    "id": r["id"], "raw_output": raw, "latency_s": round(latency, 4),
                    "tokens": {"prompt": usage["prompt_tokens"],
                               "output": usage["output_tokens"] + usage["thoughts_tokens"]},
                    "model": args.model, "prompt_version": PROMPT_VERSION, "mode": "single, structured output",
                    "key": args.key_env,   # which key/account served it (the name only, never the value)
                }, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
                break
            else:
                failed += 1
            if n_ok and n_ok % 20 == 0:
                print(f"  {len(done) + n_ok}/{len(test)} done, {n_req} requests used", flush=True)

    total = len(done) + n_ok
    print(json.dumps({"requests_this_run": n_req, "recorded_this_run": n_ok, "api_failures": failed,
                      "done_total": total, "remaining": len(test) - total}, indent=2))


if __name__ == "__main__":
    main()
