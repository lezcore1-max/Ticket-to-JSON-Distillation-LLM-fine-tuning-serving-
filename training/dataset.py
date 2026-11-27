"""Shared dataset formatting and compact prompt for QLoRA fine-tuning and inference."""
from __future__ import annotations

import json

from labeling.splits import ROOT, assert_no_test_leak, load_jsonl
from labeling.validate import FIELDS, TAXONOMY

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINAL_DIR = ROOT / "data" / "final"


def build_compact_prompt() -> str:
    """Compact system instruction (~300 tokens vs ~2,364 in prompt v3)."""
    intents = "\n".join(f"- {name} [{meta['category']}]" for name, meta in TAXONOMY.items())
    return (
        "Convert the customer-support message into a single JSON object with keys: "
        "intent, category, urgency, product, order_id, sentiment.\n"
        "Return only valid JSON.\n\n"
        "Rules:\n"
        "- category must match the chosen intent's bracketed category.\n"
        "- urgency: low | medium | high\n"
        "- sentiment: negative | neutral | positive\n"
        "- product: specific product, plan, or account tier in lowercase (null if none; "
        "never generic words like order/item/account/purchase/newsletter or IDs).\n"
        "- order_id: exact order identifier without leading '#' (null if none; never invoice IDs).\n\n"
        f"Intents [category]:\n{intents}"
    )


def format_user_message(text: str) -> str:
    return f"Message: {text}"


def format_target_json(label: dict) -> str:
    ordered = {k: label[k] for k in FIELDS}
    return json.dumps(ordered, ensure_ascii=False)


def load_final_split(split: str) -> list[dict]:
    """Load data/final/{split}.jsonl; enforce test-leak guard on train/val."""
    rows = load_jsonl(FINAL_DIR / f"{split}.jsonl")
    if split in ("train", "val"):
        assert_no_test_leak(rows)
    return rows


def to_sft_records(rows: list[dict], system_prompt: str | None = None) -> list[dict]:
    """Convert rows into TRL prompt/completion conversational records."""
    sys_prompt = system_prompt or build_compact_prompt()
    return [
        {
            "id": r["id"],
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": format_user_message(r["text"])},
            ],
            "completion": [
                {"role": "assistant", "content": format_target_json(r["label"])},
            ],
        }
        for r in rows
    ]

