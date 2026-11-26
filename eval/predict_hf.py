"""Run a Hugging Face chat model on the test set and save raw outputs for eval/eval.py.

Day 3 baseline: Qwen2.5-1.5B-Instruct with the SAME system prompt the teacher gets (prompt v3 with
examples, one message per request), greedy decoding, no constrained decoding: we want to see
how often an untrained small model breaks the JSON / schema.

Latency = wall time of one generate() call on the GPU (batch size 1), after one warm-up call.
A sidecar <out>.meta.json records model, GPU, dtype, versions and peak memory.

  python -m eval.predict_hf --out eval/predictions/qwen_prompted.jsonl            # all 200
  python -m eval.predict_hf --out eval/scratch.jsonl --n 3                        # smoke test
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from labeling.label import PROMPT_VERSION, build_system_prompt, format_single
from labeling.splits import ROOT, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: install the CUDA build of torch")
    # only id + text are read from the test file; gold labels never reach the model
    test = [{"id": r["id"], "text": r["text"]} for r in load_jsonl(ROOT / "data" / "final" / "test.jsonl")]
    if args.n:
        test = test[: args.n]
    done = {r["id"] for r in load_jsonl(args.out)} if args.out.exists() else set()
    todo = [r for r in test if r["id"] not in done]
    print(f"{len(test)} messages, {len(done)} done, {len(todo)} to go -> {args.out}")
    if not todo:
        return

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda")
    model.eval()
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
    system = build_system_prompt(batch=False)

    def run(text: str):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": format_single(text)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to("cuda")
        ids = enc["input_ids"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[0, ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True), dt, ids.shape[1], int(new.shape[0])

    run(todo[0]["text"])                       # warm-up (CUDA kernels, caches); not recorded
    torch.cuda.reset_peak_memory_stats()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(todo, 1):
            raw, dt, n_in, n_out = run(r["text"])
            f.write(json.dumps({"id": r["id"], "raw_output": raw, "latency_s": round(dt, 4),
                                "tokens": {"prompt": n_in, "output": n_out}, "model": args.model,
                                "prompt_version": PROMPT_VERSION}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  last {dt:.2f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    meta = {"model": args.model, "dtype": args.dtype, "gpu": torch.cuda.get_device_name(0),
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "batch_size": 1,
            "decoding": "greedy", "max_new_tokens": args.max_new_tokens, "prompt_version": PROMPT_VERSION,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "python": platform.python_version()}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
