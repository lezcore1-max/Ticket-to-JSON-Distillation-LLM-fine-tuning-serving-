"""Label prepared messages with the teacher model (Gemini) and keep only valid outputs.

Batch mode (default --batch-size 20): several numbered messages per request, so the ~2K-token
instruction prompt is paid once per batch. This is what makes labeling fit the free tier.
The model returns {"items": [{"id": n, "label": {...}}, ...]}; every label is validated on its
own, and missing/invalid ones are re-queued and retried in later rounds.
(Per-request teacher cost/latency for the results table is measured separately in eval, with
--batch-size 1 semantics: one message per request.)

Robustness: global rate limit across worker threads, exponential backoff on API errors,
resumable (skips ids already in the output file). Rows that keep producing invalid labels go to
<out>.rejects.jsonl; rows that only hit API errors (e.g. daily quota exhausted) are left
unlabeled and are picked up by simply re-running the same command.

Examples (from repo root):
  python -m labeling.label --list-models
  python -m labeling.label --split train --n 50 --stratify --out data/labeled/pilot.jsonl
  python -m labeling.label --split train
  python -m labeling.label --split train --dry-run      # print prompt + first batch, no API call
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
from labeling.validate import SCHEMA, TAXONOMY, parse_and_validate, validate_label

HERE = Path(__file__).resolve().parent
PROMPT_VERSION = "v3"  # v3: order-number rule for track_order, use-profile = switch_account,
#                        profile = account, newsletter is not a product (v2: urgency rules)

BATCH_INSTRUCTIONS = """

## Batch format
You will receive several messages, each starting with [id=N]. Label every message independently,
as if it were the only one: never let one message influence another's label.
Return {"items": [{"id": N, "label": {...}}, ...]} with exactly one item per message, using the
same ids. Each "label" follows all the rules above."""


# --------------------------------------------------------------------------- prompts & schemas
def build_system_prompt(batch: bool = True) -> str:
    lines = [f"- {intent} [{v['category']}]: {v['description']}" for intent, v in TAXONOMY.items()]
    p = (HERE / "prompt.md").read_text(encoding="utf-8").replace("<<INTENT_LIST>>", "\n".join(lines))
    return p + (BATCH_INSTRUCTIONS if batch else "")


def label_schema() -> dict:
    """schema.json minus JSON-Schema meta keys the Gemini API does not need."""
    return {k: v for k, v in SCHEMA.items() if k not in ("$schema", "title")}


def batch_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "label"],
            "properties": {"id": {"type": "integer"}, "label": label_schema()},
        }}},
    }


def format_single(text: str) -> str:
    return f"Message: {text}"


def format_batch(texts: list[str]) -> str:
    return "\n".join(f"[id={i}] {t}" for i, t in enumerate(texts, 1))


# --------------------------------------------------------------------------- API
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


def call_teacher(client, model: str, system_prompt: str, contents: str, schema: dict,
                 thinking_budget: int | None):
    """One API call. Returns (raw_text, usage_dict, latency_s). Raises on API error."""
    from google.genai import types

    cfg = dict(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_json_schema=schema,
        temperature=0.0,
    )
    if thinking_budget is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    t0 = time.perf_counter()
    resp = client.models.generate_content(model=model, contents=contents,
                                          config=types.GenerateContentConfig(**cfg))
    latency = time.perf_counter() - t0
    um = resp.usage_metadata
    usage = {
        "prompt_tokens": getattr(um, "prompt_token_count", None) or 0,
        "output_tokens": getattr(um, "candidates_token_count", None) or 0,
        "thoughts_tokens": getattr(um, "thoughts_token_count", None) or 0,
    }
    return resp.text, usage, latency


def parse_batch(raw: str, n: int) -> tuple[dict[int, dict], dict[int, list[str]]]:
    """Split a batch response into {local_id: valid label} and {local_id: errors}.
    Ids that are missing from the response get an error too."""
    good, bad = {}, {}
    try:
        items = json.loads(raw)["items"]
        assert isinstance(items, list)
    except Exception as e:
        return {}, {i: [f"unparseable batch: {type(e).__name__}: {str(raw)[:150]!r}"] for i in range(1, n + 1)}
    for it in items:
        i = it.get("id") if isinstance(it, dict) else None
        if not isinstance(i, int) or not 1 <= i <= n or i in good:
            continue
        errs = validate_label(it.get("label"))
        if errs:
            bad[i] = errs[:3]
        else:
            good[i] = it["label"]
            bad.pop(i, None)
    for i in range(1, n + 1):
        if i not in good and i not in bad:
            bad[i] = ["missing from batch response"]
    return good, bad


def run_batch(rows, *, client, model, system_prompt, limiter, thinking_budget, api_attempts):
    """Label one batch. Returns (labels{row_id: label}, errors{row_id: [..]}, meta) or raises
    RuntimeError if every API attempt failed (rows then stay unlabeled for a later run)."""
    batch_mode = len(rows) > 1
    contents = format_batch([r["text"] for r in rows]) if batch_mode else format_single(rows[0]["text"])
    schema = batch_schema() if batch_mode else label_schema()
    last_err = None
    for attempt in range(1, api_attempts + 1):
        limiter.wait()
        try:
            raw, usage, latency = call_teacher(client, model, system_prompt, contents, schema, thinking_budget)
            break
        except Exception as e:  # network / 429 / 5xx
            last_err = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(min(90, 5 * 2 ** attempt) + random.random() * 3)
    else:
        raise RuntimeError(last_err)

    if batch_mode:
        good, bad = parse_batch(raw, len(rows))
    else:
        obj, errs = parse_and_validate(raw)
        good, bad = ({1: obj}, {}) if not errs else ({}, {1: errs[:3]})
    meta = {"batch_size": len(rows), "latency_s": round(latency, 3), "usage": usage}
    return ({rows[i - 1]["id"]: lab for i, lab in good.items()},
            {rows[i - 1]["id"]: e for i, e in bad.items()}, meta)


# --------------------------------------------------------------------------- helpers
def stratified_take(rows: list[dict], n: int) -> list[dict]:
    """Round-robin over bitext intents (rows are already shuffled) -> balanced sample."""
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


def chunks(xs, k):
    return [xs[i:i + k] for i in range(0, len(xs), k)]


def list_models() -> None:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for m in client.models.list():
        acts = getattr(m, "supported_actions", None) or []
        if not acts or "generateContent" in acts:
            print(m.name.removeprefix("models/"), "-", getattr(m, "display_name", ""))


# --------------------------------------------------------------------------- main
def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=None, help="label only the first N rows")
    ap.add_argument("--stratify", action="store_true", help="with --n: balance across intents")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "gemini-3.5-flash-lite"))
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rpm", type=float, default=float(os.getenv("TEACHER_RPM", "12")),
                    help="stay a bit under the model's RPM limit")
    ap.add_argument("--max-request-count", type=int, default=None,
                    help="stop after this many API requests (protects the daily quota)")
    ap.add_argument("--rounds", type=int, default=3, help="max label attempts per message")
    ap.add_argument("--api-attempts", type=int, default=4, help="retries per request on API errors")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="tokens of model 'thinking'; 0 disables. Default: model default")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        return list_models()
    if not args.split:
        ap.error("--split is required")

    rows = load_jsonl(PREPARED / f"{args.split}.jsonl")
    if args.n:
        rows = stratified_take(rows, args.n) if args.stratify else rows[: args.n]
    out = args.out or ROOT / "data" / "labeled" / f"{args.split}.jsonl"
    rejects_path = out.with_suffix(".rejects.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    batch_mode = args.batch_size > 1
    system_prompt = build_system_prompt(batch=batch_mode)
    if args.dry_run:
        first = rows[: args.batch_size]
        print(system_prompt, "\n\n---- user ----\n",
              format_batch([r["text"] for r in first]) if batch_mode else format_single(first[0]["text"]))
        print(f"\n[dry-run] {len(rows)} rows -> ~{-(-len(rows) // args.batch_size)} requests "
              f"(batch {args.batch_size}) -> {out} with {args.model}")
        return

    done = {r["id"] for r in load_jsonl(out)} if out.exists() else set()
    pending = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already labeled, {len(pending)} to go -> {out} "
          f"[{args.model}, batch {args.batch_size}, {args.rpm} rpm]")
    if not pending:
        return

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set (put it in .env)")
    from google import genai
    client = genai.Client(api_key=key)
    limiter = RateLimiter(args.rpm)

    attempts = defaultdict(int)
    last_errors: dict[str, list] = {}
    n_req, n_ok, n_rej, api_failed, skipped = 0, 0, 0, [], []
    batch_lat, batch_meta = [], []
    stop = False
    with open(out, "a", encoding="utf-8", newline="\n") as f_ok, \
         open(rejects_path, "a", encoding="utf-8", newline="\n") as f_rej, \
         ThreadPoolExecutor(max_workers=args.workers) as pool, \
         tqdm(total=len(pending), desc="labeled") as bar:
        for rnd in range(1, args.rounds + 1):
            if not pending or stop:
                break
            # retry rounds use smaller batches: failures are often caused by batch size
            bs = args.batch_size if rnd == 1 else max(1, args.batch_size // (2 ** (rnd - 1)))
            batches = chunks(pending, bs)
            if args.max_request_count is not None:
                room = args.max_request_count - n_req
                if room <= 0:
                    break
                if len(batches) > room:
                    skipped.extend(r for b in batches[room:] for r in b)
                    batches, stop = batches[:room], True
            n_req += len(batches)
            next_pending = []
            futs = {pool.submit(run_batch, b, client=client, model=args.model,
                                system_prompt=build_system_prompt(batch=len(b) > 1), limiter=limiter,
                                thinking_budget=args.thinking_budget, api_attempts=args.api_attempts): b
                    for b in batches}
            for fut in as_completed(futs):
                b = futs[fut]
                try:
                    good, bad, meta = fut.result()
                except RuntimeError as e:  # API kept failing: leave unlabeled for next run
                    tqdm.write(f"[round {rnd}] API failure on batch of {len(b)}: {e}")
                    api_failed.extend(b)
                    continue
                batch_lat.append(meta["latency_s"])
                batch_meta.append(meta)
                per_row = {k: round(v / meta["batch_size"], 1) for k, v in meta["usage"].items()}
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for r in b:
                    attempts[r["id"]] += 1
                    if r["id"] in good:
                        f_ok.write(json.dumps({
                            "id": r["id"], "text": r["text"], "label": good[r["id"]],
                            "teacher": args.model, "prompt_version": PROMPT_VERSION,
                            "attempts": attempts[r["id"]], "batch": meta,
                            "usage_per_row": per_row, "ts": ts,
                        }, ensure_ascii=False) + "\n")
                        n_ok += 1
                        bar.update(1)
                    else:
                        last_errors[r["id"]] = bad.get(r["id"], ["?"])
                        if attempts[r["id"]] >= args.rounds:
                            f_rej.write(json.dumps({"id": r["id"], "text": r["text"], "teacher": args.model,
                                                    "attempts": attempts[r["id"]],
                                                    "errors": last_errors[r["id"]]}, ensure_ascii=False) + "\n")
                            n_rej += 1
                            bar.update(1)
                        else:
                            next_pending.append(r)
                f_ok.flush(); f_rej.flush()
            pending = next_pending

    # compact rejects: drop ids that have since been labeled, keep latest per id
    labeled = {r["id"] for r in load_jsonl(out)}
    latest = {r["id"]: r for r in load_jsonl(rejects_path) if r["id"] not in labeled}
    with open(rejects_path, "w", encoding="utf-8", newline="\n") as f:
        for r in latest.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    leftover = len(api_failed) + len(pending) + len(skipped)
    print(json.dumps({
        "model": args.model, "requests": n_req, "labeled": n_ok, "rejected": n_rej,
        "left_unlabeled_rerun_to_finish": leftover,
        "label_valid_rate": round(n_ok / max(1, n_ok + n_rej), 4),
        "batch_latency_s_mean": round(statistics.mean(batch_lat), 2) if batch_lat else None,
        "prompt_tokens_per_request_mean": round(statistics.mean(m["usage"]["prompt_tokens"] for m in batch_meta), 1) if batch_meta else None,
        "total_tokens_this_run": sum(sum(m["usage"].values()) for m in batch_meta),
    }, indent=2))


if __name__ == "__main__":
    main()
