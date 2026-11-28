"""Export serving/merged_model to quantized GGUF (Q4_K_M) using llama.cpp in Docker.

Produces demo/models/ticket-qwen2.5-1.5b-q4_k_m.gguf (gitignored via *.gguf).

  python -m demo.export_gguf
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from labeling.splits import ROOT

LLAMA_IMAGE = "ghcr.io/ggml-org/llama.cpp:full"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROOT / "serving" / "merged_model")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "demo" / "models")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--keep-f16", action="store_true")
    args = ap.parse_args()

    if not args.model_dir.is_dir():
        raise SystemExit(f"Merged model not found: {args.model_dir} (run serving.merge_model first)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    f16_name = "ticket-qwen2.5-1.5b-f16.gguf"
    q_name = f"ticket-qwen2.5-1.5b-{args.quant.lower()}.gguf"
    f16_path = args.out_dir / f16_name
    q_path = args.out_dir / q_name

    t0 = time.perf_counter()
    model_mount = f"{args.model_dir.resolve()}:/model:ro"
    out_mount = f"{args.out_dir.resolve()}:/out"

    # transformers 5.x writes extra_special_tokens as a list; older converter expects dict or absent
    extra_mounts: list[str] = []
    tmp_cfg_path: Path | None = None
    tok_cfg_file = args.model_dir / "tokenizer_config.json"
    if tok_cfg_file.exists():
        tok_cfg = json.loads(tok_cfg_file.read_text(encoding="utf-8"))
        if isinstance(tok_cfg.get("extra_special_tokens"), list):
            tok_cfg.pop("extra_special_tokens")
            tmp_fd, tmp_name = tempfile.mkstemp(suffix=".json")
            os_path = Path(tmp_name)
            os.close(tmp_fd)
            os_path.write_text(json.dumps(tok_cfg, indent=2) + "\n", encoding="utf-8")
            tmp_cfg_path = os_path
            extra_mounts = ["-v", f"{tmp_cfg_path.resolve()}:/model/tokenizer_config.json:ro"]

    try:
        run([
            "docker", "run", "--rm", "--entrypoint", "python3",
            "-v", model_mount, *extra_mounts, "-v", out_mount,
            LLAMA_IMAGE,
            "/app/convert_hf_to_gguf.py", "/model", "--outfile", f"/out/{f16_name}", "--outtype", "f16",
        ])
    finally:
        if tmp_cfg_path and tmp_cfg_path.exists():
            tmp_cfg_path.unlink()
    run([
        "docker", "run", "--rm", "--entrypoint", "/app/llama-quantize",
        "-v", out_mount,
        LLAMA_IMAGE,
        f"/out/{f16_name}", f"/out/{q_name}", args.quant,
    ])
    if not args.keep_f16 and f16_path.exists():
        f16_path.unlink()

    meta = {
        "source_model": str(args.model_dir.relative_to(ROOT)).replace("\\", "/"),
        "quant": args.quant,
        "gguf_file": str(q_path.relative_to(ROOT)).replace("\\", "/"),
        "size_mib": round(q_path.stat().st_size / 2**20, 1),
        "export_seconds": round(time.perf_counter() - t0, 1),
        "docker_image": LLAMA_IMAGE,
    }
    (args.out_dir / "gguf_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
