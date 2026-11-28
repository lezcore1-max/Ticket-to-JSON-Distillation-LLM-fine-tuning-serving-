---
title: Ticket-to-JSON Distillation
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Ticket-to-JSON Demo

Runs the QLoRA fine-tuned `Qwen2.5-1.5B-Instruct` (`Q4_K_M` GGUF) model on CPU via `llama-cpp-python` and validates predictions with `labeling/validate.py`.

## Deploy to Hugging Face Spaces (CPU Basic)
1. Upload the `Q4_K_M` GGUF (`demo/models/ticket-qwen2.5-1.5b-q4_k_m.gguf`) to a Hugging Face model repo.
2. Create a Gradio Space (`cpu-basic`), copy `demo/app.py`, `demo/requirements.txt`, `labeling/`, and `training/dataset.py` (plus `training/__init__.py`), and set Space variable `HF_GGUF_REPO=<your-username>/<your-gguf-repo>`.
