"""Benchmark the live FastAPI + vLLM /predict endpoint on data/final/test.jsonl.

Saves predictions in eval/eval.py format so vLLM + guided JSON can be compared
directly against HF transformers in README.md.

  python -m serving.eval_api --out eval/predictions/vllm_guided.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from labeling.splits import ROOT, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/predict")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "predictions" / "vllm_guided.jsonl")
    ap.add_argument("--no-guided", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    test = [{"id": r["id"], "text": r["text"]} for r in load_jsonl(ROOT / "data" / "final" / "test.jsonl")]
    if args.n:
        test = test[: args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0) as client:
        # warm-up on first message (not recorded)
        client.post(args.url, json={"text": test[0]["text"], "guided_json": not args.no_guided}).raise_for_status()

        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            for i, r in enumerate(test, 1):
                t0 = time.perf_counter()
                resp = client.post(args.url, json={"text": r["text"], "guided_json": not args.no_guided})
                resp.raise_for_status()
                dt = time.perf_counter() - t0
                body = resp.json()
                f.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "raw_output": body["raw_output"],
                            "latency_s": round(dt, 4),
                            "tokens": body.get("tokens"),
                            "valid": body.get("valid"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if i % 20 == 0 or i == len(test):
                    print(f"  {i}/{len(test)}  last {dt*1000:.0f} ms", flush=True)
    print("-> wrote", args.out)


if __name__ == "__main__":
    main()

