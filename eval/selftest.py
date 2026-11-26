"""Self-test for eval.py. Run before trusting any number it produces.

1. Known answer: scoring the teacher's own test labels must reproduce data/final/review_stats.json
   (exact match and per-field accuracy computed independently by labeling/merge_review.py).
2. Edge cases: fenced JSON, prose around JSON, garbage, missing rows, product/order_id
   normalization, schema-invalid but partly correct outputs.

  python -m eval.selftest
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from eval.eval import evaluate, parse_lenient, parse_strict, score_rows
from labeling.splits import ROOT, load_jsonl


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def known_answer():
    review = json.loads((ROOT / "data" / "final" / "review_stats.json").read_text(encoding="utf-8"))
    teacher = load_jsonl(ROOT / "data" / "labeled" / "test.jsonl")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "teacher_labels.jsonl"
        p.write_text("".join(json.dumps({"id": t["id"], "raw_output": json.dumps(t["label"])}) + "\n" for t in teacher),
                     encoding="utf-8")
        res = evaluate(p, "selftest teacher labels")
    check(res["missing_predictions"] == 0 and res["schema_valid"] == 1.0, "teacher labels: all 200 present and schema-valid")
    check(abs(res["exact_match"] - review["teacher_exact_match_vs_reviewed"]) < 1e-9,
          f"exact match {res['exact_match']} == review_stats {review['teacher_exact_match_vs_reviewed']}")
    for f, v in review["teacher_field_accuracy_vs_reviewed"].items():
        check(abs(res[f"acc_{f}"] - v) < 1e-9, f"{f} accuracy {res[f'acc_{f}']} == review_stats {v}")
    lo, hi = res["ci95"]["exact_match"]
    check(lo <= res["exact_match"] <= hi, f"95% CI {lo}-{hi} contains the point estimate")


def edge_cases():
    gold_label = {"intent": "track_order", "category": "order", "urgency": "medium",
                  "product": "gold account", "order_id": "4521", "sentiment": "neutral"}
    good = json.dumps(gold_label)
    cases = {
        "exact": good,
        "fenced": f"```json\n{good}\n```",
        "prose": f"Sure! Here is the JSON: {good} Hope this helps.",
        "garbage": "I cannot help with that.",
        "norm": json.dumps({**gold_label, "product": "the Gold profile"}),
        "upper_intent": json.dumps({**gold_label, "intent": "TRACK_ORDER"}),
        "hash_id": json.dumps({**gold_label, "order_id": "#4521"}),
        "bad_category": json.dumps({**gold_label, "category": "refund"}),
        "nested_braces": '{"note": "a {b} c"} then ' + good,
    }
    gold = [{"id": k, "label": gold_label} for k in list(cases) + ["missing"]]
    rows = {r["id"]: r for r in score_rows(gold, {k: {"raw_output": v} for k, v in cases.items()})}

    check(rows["exact"]["strict"] and rows["exact"]["exact"], "plain JSON: strict-valid and exact")
    check(not rows["fenced"]["strict"] and rows["fenced"]["lenient"] and rows["fenced"]["exact"],
          "fenced JSON: strict-invalid, lenient-valid, exact")
    check(not rows["prose"]["strict"] and rows["prose"]["exact"], "JSON inside prose: recovered by lenient parse")
    check(not rows["garbage"]["lenient"] and not any(rows["garbage"]["correct"].values()), "garbage: wrong on every field")
    check(rows["missing"]["missing"] and not any(rows["missing"]["correct"].values()), "missing prediction: wrong on every field")
    check(rows["norm"]["exact"], "'the Gold profile' == 'gold account'")
    check(rows["upper_intent"]["correct"]["intent"] and not rows["upper_intent"]["exact"],
          "'TRACK_ORDER': intent field credited, but not a schema enum value, so not exact")
    check(rows["hash_id"]["correct"]["order_id"] and not rows["hash_id"]["schema"] and not rows["hash_id"]["exact"],
          "'#4521': order_id field correct, but not schema-valid (non-canonical), so not exact")
    check(rows["bad_category"]["correct"]["intent"] and not rows["bad_category"]["exact"],
          "category/intent mismatch: intent credited, exact match not")
    check(rows["nested_braces"]["lenient"] and not rows["nested_braces"]["exact"],
          "first balanced object is taken (the note object), so not exact")
    check(parse_strict("[1, 2]") is None and parse_lenient(None) is None, "non-object JSON and None rejected")


if __name__ == "__main__":
    known_answer()
    edge_cases()
    print("all checks passed")
