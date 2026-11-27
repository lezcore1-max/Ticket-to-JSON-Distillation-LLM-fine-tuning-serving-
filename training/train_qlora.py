"""QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on data/final/{train,val}.jsonl.

Uses the compact prompt in training/dataset.py, 4-bit NF4 quantization,
bfloat16 compute, and completion-only loss masking.

  python -m training.train_qlora --run-name run1 --local-files-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from labeling.splits import ROOT  # noqa: E402
from training.dataset import (  # noqa: E402
    BASE_MODEL,
    build_compact_prompt,
    load_final_split,
    to_sft_records,
)


def oversample_minority(rows: list[dict], factor: int, seed: int) -> list[dict]:
    """Repeat hard/rare rows (high urgency, negative sentiment, or non-null entities)."""
    if factor <= 1:
        return rows
    out = list(rows)
    for r in rows:
        lab = r["label"]
        is_rare = (
            lab["urgency"] == "high"
            or lab["sentiment"] == "negative"
            or lab["product"] is not None
            or lab["order_id"] is not None
        )
        if is_rare:
            out.extend([r] * (factor - 1))
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="run1")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--oversample-factor", type=int, default=1)
    ap.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing (faster on >=16GB GPUs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: install the CUDA build of torch")

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    grad_ckpt = not args.no_grad_ckpt

    out_dir = args.out_dir or (ROOT / "training" / "checkpoints" / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_final_split("train")
    val_rows = load_final_split("val")
    train_rows = oversample_minority(train_rows, args.oversample_factor, args.seed)

    sys_prompt = build_compact_prompt()
    train_ds = Dataset.from_list(to_sft_records(train_rows, sys_prompt))
    val_ds = Dataset.from_list(to_sft_records(val_rows, sys_prompt))
    print(f"[{args.run_name}] train={len(train_ds)} rows, val={len(val_ds)} rows, dtype={compute_dtype}, grad_ckpt={grad_ckpt} -> {out_dir}")

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="cuda",
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=6,
        save_only_model=True,
        logging_steps=25,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    train_res = trainer.train()
    elapsed = time.perf_counter() - t0

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    meta = {
        "run_name": args.run_name,
        "base_model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "oversample_factor": args.oversample_factor,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_params": trainable,
        "total_params": total,
        "train_loss": round(float(train_res.training_loss), 4),
        "train_seconds": round(elapsed, 1),
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()

