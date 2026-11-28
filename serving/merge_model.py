"""Merge a trained QLoRA adapter into the base model for vLLM / GGUF serving.

Merges on CPU in bfloat16 (no GPU VRAM needed) and writes a standalone Hugging Face
model directory to serving/merged_model/ (gitignored).

  python -m serving.merge_model --local-files-only
"""
from __future__ import annotations

import argparse
import json
import os
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
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from labeling.splits import ROOT  # noqa: E402
from training.dataset import BASE_MODEL  # noqa: E402


def default_adapter() -> Path:
    sel_file = ROOT / "training" / "checkpoints" / "run1" / "val_selection.json"
    if sel_file.exists():
        sel = json.loads(sel_file.read_text(encoding="utf-8"))
        return ROOT / "training" / "checkpoints" / "run1" / sel["best"]["checkpoint"]
    return ROOT / "training" / "checkpoints" / "run1" / "checkpoint-376"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "serving" / "merged_model")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    adapter = args.adapter or default_adapter()
    if not adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {adapter}")

    dtype = getattr(torch, args.dtype)
    t0 = time.perf_counter()
    print(f"Merging {args.base_model} + {adapter} ({args.dtype}) -> {args.out_dir}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="cpu",
        local_files_only=args.local_files_only,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    merged = model.merge_and_unload()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out_dir), safe_serialization=True)
    tok.save_pretrained(str(args.out_dir))

    files = {p.name: p.stat().st_size for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    total_gb = round(sum(files.values()) / 2**30, 2)
    meta = {
        "base_model": args.base_model,
        "adapter": str(adapter.relative_to(ROOT)).replace("\\", "/") if adapter.is_relative_to(ROOT) else str(adapter),
        "dtype": args.dtype,
        "total_gib": total_gb,
        "merge_seconds": round(time.perf_counter() - t0, 1),
        "files": sorted(files),
    }
    (args.out_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

