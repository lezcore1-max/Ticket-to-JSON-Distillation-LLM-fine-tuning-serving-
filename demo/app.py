"""Gradio demo for Ticket-to-JSON Distillation (runs on HF Spaces CPU or locally).

Backend selection:
  1. If FASTAPI_URL is set (e.g. http://localhost:8080), calls FastAPI POST /predict.
  2. If LLAMA_SERVER_URL is set (e.g. http://localhost:8081), calls llama-server's OpenAI API.
  3. Otherwise loads the local Q4_K_M GGUF (or downloads from HF_GGUF_REPO) via llama-cpp-python.

  python -m demo.app
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import gradio as gr
import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labeling.validate import parse_and_validate  # noqa: E402
from training.dataset import build_compact_prompt, format_user_message  # noqa: E402

LOCAL_GGUF = ROOT / "demo" / "models" / "ticket-qwen2.5-1.5b-q4_k_m.gguf"
HF_GGUF_REPO = os.getenv("HF_GGUF_REPO", "")
HF_GGUF_FILE = os.getenv("HF_GGUF_FILE", "ticket-qwen2.5-1.5b-q4_k_m.gguf")
FASTAPI_URL = os.getenv("FASTAPI_URL", "").rstrip("/")
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "").rstrip("/")
SYSTEM_PROMPT = build_compact_prompt()

_LLM = None


def get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM
    from llama_cpp import Llama

    model_path = LOCAL_GGUF
    if not model_path.is_file():
        if not HF_GGUF_REPO:
            raise FileNotFoundError(
                f"GGUF not found at {LOCAL_GGUF}. Run `python -m demo.export_gguf`, set HF_GGUF_REPO, or set LLAMA_SERVER_URL."
            )
        from huggingface_hub import hf_hub_download

        model_path = Path(hf_hub_download(repo_id=HF_GGUF_REPO, filename=HF_GGUF_FILE))

    _LLM = Llama(
        model_path=str(model_path),
        n_ctx=1024,
        n_threads=min(8, os.cpu_count() or 4),
        verbose=False,
    )
    return _LLM


def extract_ticket(message: str):
    text = (message or "").strip()
    if not text:
        return {}, "Enter a customer support message.", ""

    if FASTAPI_URL:
        t0 = time.perf_counter()
        r = httpx.post(f"{FASTAPI_URL}/predict", json={"text": text, "guided_json": True}, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        latency_ms = data.get("latency_ms") or round((time.perf_counter() - t0) * 1000, 1)
        toks = data.get("tokens") or {}
        obj = data.get("label")
        errors = data.get("errors") or []
        raw = data.get("raw_output", "")
        status = (
            "Valid JSON (passes schema.json + taxonomy consistency)"
            if data.get("valid") and not errors
            else "Validation error: " + "; ".join(errors)
        )
        stats = f"Latency: {latency_ms} ms | Tokens: {toks.get('prompt', 0)} prompt + {toks.get('output', 0)} output"
        return (obj if obj is not None else {"raw_output": raw}), status, stats

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_message(text)},
    ]
    t0 = time.perf_counter()
    if LLAMA_SERVER_URL:
        r = httpx.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json={"messages": messages, "temperature": 0.0, "max_tokens": 120},
            timeout=60.0,
        )
        r.raise_for_status()
        resp = r.json()
    else:
        resp = get_llm().create_chat_completion(messages=messages, temperature=0.0, max_tokens=120)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    raw = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage") or {}

    obj, errors = parse_and_validate(raw)
    status = (
        "Valid JSON (passes schema.json + taxonomy consistency)"
        if not errors
        else "Validation error: " + "; ".join(errors)
    )
    stats = (
        f"Latency: {latency_ms} ms | "
        f"Tokens: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} output"
    )
    return (obj if obj is not None else {"raw_output": raw}), status, stats


EXAMPLES = [
    ["my order #4521 hasn't arrived, it's been 10 days, very frustrated!!"],
    ["can i upgrade to the platnum proffile"],
    ["how do i download invoice #20931"],
    ["i was charged twice for my Gold account, fix this NOW"],
    ["could you tell me what payment options you accept?"],
]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Ticket-to-JSON Distillation") as demo:
        gr.Markdown(
            "# Ticket-to-JSON Distillation (`Qwen2.5-1.5B` QLoRA, `Q4_K_M` GGUF)\n"
            "Converts raw customer-support tickets into a strict 6-field JSON payload "
            "validated against `labeling/schema.json` and `labeling/taxonomy.json`."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(
                    label="Customer Support Message",
                    placeholder="Type or pick an example ticket below...",
                    lines=4,
                )
                btn = gr.Button("Extract JSON", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=inp)
            with gr.Column():
                out_json = gr.JSON(label="Extracted JSON")
                out_status = gr.Textbox(label="Schema Validation", interactive=False)
                out_stats = gr.Textbox(label="Inference Stats", interactive=False)

        btn.click(fn=extract_ticket, inputs=inp, outputs=[out_json, out_status, out_stats])
        inp.submit(fn=extract_ticket, inputs=inp, outputs=[out_json, out_status, out_stats])
    return demo


if __name__ == "__main__":
    build_demo().launch()
