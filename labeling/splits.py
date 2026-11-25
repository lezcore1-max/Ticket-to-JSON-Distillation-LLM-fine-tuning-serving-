"""Split bookkeeping. The test set must never leak into train/val.

Every training / eval script should call `assert_no_test_leak` on whatever it loads.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREPARED = ROOT / "data" / "prepared"
TEST_IDS_FILE = ROOT / "data" / "test_ids.txt"


def norm_text(s: str) -> str:
    return " ".join(s.lower().split())


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_test_ids() -> set[str]:
    return {l.strip() for l in TEST_IDS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}


def assert_no_test_leak(rows: list[dict], test_rows: list[dict] | None = None) -> None:
    """Raise if any row shares an id (or normalized text) with the test set."""
    if test_rows is None:
        test_ids = load_test_ids()
        test_rows = load_jsonl(PREPARED / "test.jsonl")
    else:
        test_ids = {r["id"] for r in test_rows}
    test_texts = {norm_text(r["text"]) for r in test_rows}
    test_templates = {norm_text(r["template"]) for r in test_rows}
    bad_id = [r["id"] for r in rows if r["id"] in test_ids]
    bad_txt = [r["id"] for r in rows if norm_text(r["text"]) in test_texts]
    bad_tpl = [r["id"] for r in rows if "template" in r and norm_text(r["template"]) in test_templates]
    if bad_id or bad_txt or bad_tpl:
        raise AssertionError(
            f"TEST LEAK: {len(bad_id)} id overlaps, {len(bad_txt)} text overlaps, "
            f"{len(bad_tpl)} template overlaps. e.g. {(bad_id + bad_txt + bad_tpl)[:5]}"
        )
