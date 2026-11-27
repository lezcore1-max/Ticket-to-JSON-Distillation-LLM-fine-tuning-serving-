"""Evaluate QLoRA checkpoints on val (for checkpoint selection) or test (for eval.py).

Examples:
  # 1. Pick best checkpoint on data/final/val.jsonl
  python -m training.eval_qlora --checkpoints-dir training/checkpoints/run1 --pick-best --local-files-only

  # 2. Run chosen checkpoint on data/final/test.jsonl (batch=1 latency) for eval/eval.py
  python -m training.eval_qlora --adapter training/checkpoints/run1/checkpoint-376 --split test \
      --out eval/predictions/qwen_ft_run1.jsonl --local-files-only
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path


def _ensure_writable_cache() -> None:
    d = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if d:
        try:
            os.makedirs(d, exist_ok=True)
            return
        except OSError:
            pass
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "torchinductor")


_ensure_writable_cache()

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from eval.eval import headline, score_rows  # noqa: E402
from labeling.splits import ROOT, load_jsonl  # noqa: E402
from training.dataset import (  # noqa: E402
    BASE_MODEL,
    build_compact_prompt,
    format_user_message,
    load_final_split,
)


def load_model(base_model: str, adapter_dir: Path, load_in_4bit: bool, local_files_only: bool):
    tok = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb, device_map="cuda", local_files_only=local_files_only
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=compute_dtype, device_map="cuda", local_files_only=local_files_only
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    model.eval()
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
    return model, tok


def predict_batch_val(model, tok, sys_prompt: str, rows: list[dict], batch_size: int, max_new_tokens: int) -> dict[str, dict]:
    tok.padding_side = "left"
    preds = {}
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": format_user_message(r["text"])}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for r in rows
    ]
    for start in range(0, len(rows), batch_size):
        chunk_rows = rows[start: start + batch_size]
        enc = tok(prompts[start: start + batch_size], return_tensors="pt", padding=True).to("cuda")
        in_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        for j, r in enumerate(chunk_rows):
            raw = tok.decode(out[j, in_len:], skip_special_tokens=True)
            preds[r["id"]] = {"id": r["id"], "raw_output": raw}
    return preds


def pick_best_checkpoint(
    checkpoints_dir: Path,
    base_model: str,
    load_in_4bit: bool,
    val_batch_size: int,
    max_new_tokens: int,
    local_files_only: bool,
) -> Path:
    val_rows = load_final_split("val")
    val_inputs = [{"id": r["id"], "text": r["text"]} for r in val_rows]
    sys_prompt = build_compact_prompt()

    ckpts = sorted(
        [p for p in checkpoints_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if (checkpoints_dir / "final").is_dir():
        ckpts.append(checkpoints_dir / "final")
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {checkpoints_dir}")

    results = []
    for ckpt in ckpts:
        t0 = time.perf_counter()
        model, tok = load_model(base_model, ckpt, load_in_4bit=load_in_4bit, local_files_only=local_files_only)
        preds = predict_batch_val(model, tok, sys_prompt, val_inputs, val_batch_size, max_new_tokens)
        scored = score_rows(val_rows, preds)
        h = headline(scored)
        schema_valid = sum(r["schema"] for r in scored) / len(scored)
        rec = {
            "checkpoint": ckpt.name,
            "path": str(ckpt),
            "exact_match": round(h["exact_match"], 4),
            "field_acc_mean": round(h["field_acc_mean"], 4),
            "schema_valid": round(schema_valid, 4),
            "acc_intent": round(h["acc_intent"], 4),
            "acc_urgency": round(h["acc_urgency"], 4),
            "acc_product": round(h["acc_product"], 4),
            "acc_order_id": round(h["acc_order_id"], 4),
            "acc_sentiment": round(h["acc_sentiment"], 4),
            "eval_seconds": round(time.perf_counter() - t0, 1),
        }
        results.append(rec)
        print(json.dumps(rec), flush=True)
        del model
        torch.cuda.empty_cache()

    best = max(results, key=lambda r: (r["exact_match"], r["schema_valid"], r["field_acc_mean"]))
    summary = {"val_n": len(val_rows), "best": best, "all": results}
    (checkpoints_dir / "val_selection.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"BEST: {best['checkpoint']} (val exact_match={best['exact_match']}, schema_valid={best['schema_valid']})")
    return Path(best["path"])


def predict_split_single(
    adapter_dir: Path,
    split: str,
    out_path: Path,
    base_model: str,
    load_in_4bit: bool,
    max_new_tokens: int,
    local_files_only: bool,
) -> None:
    # Only id + text are read during prediction; gold labels never reach the model.
    src_path = ROOT / "data" / "final" / f"{split}.jsonl"
    rows = [{"id": r["id"], "text": r["text"]} for r in load_jsonl(src_path)]
    done = {r["id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{split}: {len(rows)} messages, {len(done)} done, {len(todo)} to go -> {out_path}")
    if not todo:
        return

    model, tok = load_model(base_model, adapter_dir, load_in_4bit=load_in_4bit, local_files_only=local_files_only)
    sys_prompt = build_compact_prompt()

    def run_one(text: str):
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": format_user_message(text)}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")
        ids = enc["input_ids"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[0, ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True), dt, int(ids.shape[1]), int(new.shape[0])

    run_one(todo[0]["text"])  # warm-up; not recorded
    torch.cuda.reset_peak_memory_stats()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(todo, 1):
            raw, dt, n_in, n_out = run_one(r["text"])
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "raw_output": raw,
                        "latency_s": round(dt, 4),
                        "tokens": {"prompt": n_in, "output": n_out},
                        "model": base_model,
                        "adapter": str(adapter_dir),
                        "prompt": "compact_v1",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(
                    f"  {i}/{len(todo)}  last {dt:.2f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                    flush=True,
                )

    meta = {
        "base_model": base_model,
        "adapter": str(adapter_dir),
        "mode": "4bit_nf4" if load_in_4bit else "merged_bfloat16",
        "gpu": torch.cuda.get_device_name(0),
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "batch_size": 1,
        "decoding": "greedy",
        "max_new_tokens": max_new_tokens,
        "prompt": "compact_v1",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--checkpoints-dir", type=Path, default=None)
    ap.add_argument("--pick-best", action="store_true")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--val-batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    adapter = args.adapter
    if args.pick_best:
        if not args.checkpoints_dir:
            ap.error("--checkpoints-dir is required with --pick-best")
        adapter = pick_best_checkpoint(
            args.checkpoints_dir,
            args.model,
            load_in_4bit=args.load_in_4bit,
            val_batch_size=args.val_batch_size,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only,
        )

    if args.out:
        if not adapter:
            ap.error("--adapter or --pick-best is required when --out is set")
        predict_split_single(
            adapter,
            args.split,
            args.out,
            args.model,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only,
        )


if __name__ == "__main__":
    main()

