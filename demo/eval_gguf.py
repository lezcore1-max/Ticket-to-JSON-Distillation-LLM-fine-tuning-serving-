"""Evaluate the exported Q4_K_M GGUF model on data/final/test.jsonl.

Supports either an OpenAI-compatible llama.cpp server (--server-url, e.g. from
ghcr.io/ggml-org/llama.cpp:full) or in-process llama-cpp-python.

  python -m demo.eval_gguf --server-url http://localhost:8081
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import httpx

from labeling.splits import ROOT, load_jsonl
from training.dataset import build_compact_prompt, format_user_message


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=ROOT / "demo" / "models" / "ticket-qwen2.5-1.5b-q4_k_m.gguf")
    ap.add_argument("--server-url", default=None, help="llama-server base URL (e.g. http://localhost:8081)")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "predictions" / "gguf_q4_k_m.jsonl")
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    test = [{"id": r["id"], "text": r["text"]} for r in load_jsonl(ROOT / "data" / "final" / "test.jsonl")]
    if args.n:
        test = test[: args.n]
    done = {r["id"] for r in load_jsonl(args.out)} if args.out.exists() else set()
    todo = [r for r in test if r["id"] not in done]
    print(f"GGUF ({args.gguf.name}): {len(test)} total, {len(done)} done, {len(todo)} to go")
    if not todo:
        return

    sys_prompt = build_compact_prompt()

    if args.server_url:
        client = httpx.Client(base_url=args.server_url.rstrip("/"), timeout=120.0)

        def run_one(text: str):
            payload = {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": format_user_message(text)},
                ],
                "temperature": 0.0,
                "max_tokens": 120,
            }
            t0 = time.perf_counter()
            r = client.post("/v1/chat/completions", json=payload)
            r.raise_for_status()
            dt = time.perf_counter() - t0
            data = r.json()
            raw = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            return raw, dt, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    else:
        if not args.gguf.is_file():
            raise SystemExit(f"GGUF file not found: {args.gguf}")
        from llama_cpp import Llama

        llm = Llama(model_path=str(args.gguf), n_ctx=args.n_ctx, n_threads=args.threads, verbose=False)

        def run_one(text: str):
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": format_user_message(text)},
            ]
            t0 = time.perf_counter()
            resp = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=120)
            dt = time.perf_counter() - t0
            raw = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage") or {}
            return raw, dt, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    run_one(todo[0]["text"])  # warm-up; not recorded

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(todo, 1):
            raw, dt, n_in, n_out = run_one(r["text"])
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "raw_output": raw,
                        "latency_s": round(dt, 4),
                        "tokens": {"prompt": n_in, "output": n_out},
                        "gguf": args.gguf.name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  last {dt:.2f}s", flush=True)

    meta = {
        "gguf": str(args.gguf.name),
        "size_mib": round(args.gguf.stat().st_size / 2**20, 1) if args.gguf.exists() else None,
        "backend": args.server_url or "llama-cpp-python",
        "threads": args.threads,
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
