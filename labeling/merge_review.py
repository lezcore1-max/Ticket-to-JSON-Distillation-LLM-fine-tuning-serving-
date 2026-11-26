"""Merge the hand-checked sheet into the final dataset and measure teacher quality.

Reads data/review/test_review.csv (all 200 rows must have reviewed=y), validates every final
label, and writes:
  data/final/test.jsonl       reviewed labels (the ONLY labels used for test metrics);
                              label_source = hand_checked (--reviewer human) or claude_checked
  data/final/train.jsonl      teacher labels
  data/final/val.jsonl        teacher labels
  data/final/review_stats.json  share of teacher labels corrected, per field + overall
Every row: {"id", "text", "label", "label_source"} (test rows also keep "teacher_label").

  python -m labeling.merge_review --reviewer human     # or --reviewer claude
"""
from __future__ import annotations

import csv
import json
import re

from labeling.splits import PREPARED, ROOT, assert_no_test_leak, load_jsonl, load_test_ids, write_jsonl
from labeling.validate import FIELDS as LABEL_FIELDS, INTENT_TO_CATEGORY, validate_label

SHEET = ROOT / "data" / "review" / "test_review.csv"
FINAL = ROOT / "data" / "final"
EDITABLE = ["intent", "urgency", "product", "order_id", "sentiment"]


def clean(field: str, v: str):
    v = (v or "").strip()
    if v.lower() in ("", "null", "none"):
        return None
    if field == "order_id":
        return v.lstrip("'").lstrip("#").strip()
    if field == "product":
        return " ".join(v.lower().split())
    return v.lower()


def excel_mangled(final: str | None, teacher: str | None) -> bool:
    """Excel turns 00123842 into 123842 and 732201349959 into 7.32201E+11."""
    if final is None or teacher is None or final == teacher:
        return False
    if re.fullmatch(r"[\d.]+E\+\d+", final, re.I):
        return True
    return final.isdigit() and teacher.isdigit() and int(final) == int(teacher)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewer", choices=["human", "claude"], required=True,
                    help="who checked the sheet; recorded in every test row and in review_stats.json")
    args = ap.parse_args()
    source = "hand_checked" if args.reviewer == "human" else "claude_checked"
    with open(SHEET, encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    test_ids = load_test_ids()
    ids = [r["id"] for r in sheet]
    problems = []
    if set(ids) != test_ids or len(ids) != len(test_ids):
        problems.append(f"sheet ids do not match data/test_ids.txt ({len(set(ids))} unique ids in sheet)")
    not_reviewed = [r["row"] for r in sheet if r["reviewed"].strip().lower() not in ("y", "yes")]
    if not_reviewed:
        problems.append(f"{len(not_reviewed)} rows not marked reviewed=y, e.g. rows {not_reviewed[:10]}")

    teacher = {l["id"]: l for l in load_jsonl(ROOT / "data" / "labeled" / "test.jsonl")}
    test_rows, changed = [], {f: 0 for f in EDITABLE}
    n_changed_rows, change_examples = 0, []
    for r in sheet:
        lab = {f: clean(f, r[f"final_{f}"]) for f in EDITABLE}
        lab["category"] = INTENT_TO_CATEGORY.get(lab["intent"], "?")
        lab = {k: lab[k] for k in LABEL_FIELDS}
        t = teacher[r["id"]]["label"]
        if excel_mangled(lab["order_id"], t["order_id"]):
            problems.append(f"row {r['row']}: order_id {lab['order_id']!r} looks changed by Excel "
                            f"(teacher had {t['order_id']!r}); fix it or open the CSV as text")
        errs = validate_label(lab)
        if errs:
            problems.append(f"row {r['row']} ({r['id']}): {errs[0]}")
            continue
        diff = [f for f in EDITABLE if lab[f] != t[f]]
        for f in diff:
            changed[f] += 1
        if diff:
            n_changed_rows += 1
            change_examples.append({"id": r["id"], "text": r["text"],
                                    **{f: [t[f], lab[f]] for f in diff}})
        test_rows.append({"id": r["id"], "text": r["text"], "label": lab, "label_source": source,
                          "teacher_label": t, "notes": r.get("notes", "")})

    if problems:
        print("NOT merged. Fix these in the sheet and re-run:")
        for p in problems[:40]:
            print("  -", p)
        raise SystemExit(1)

    tv = {}
    for split in ("train", "val"):
        tv[split] = [{"id": l["id"], "text": l["text"], "label": l["label"], "label_source": l["teacher"]}
                     for l in load_jsonl(ROOT / "data" / "labeled" / f"{split}.jsonl")]
        expected = len(load_jsonl(PREPARED / f"{split}.jsonl"))
        if len(tv[split]) != expected:
            raise SystemExit(f"NOT merged: {split} has {len(tv[split])}/{expected} teacher labels; "
                             f"finish labeling first (re-run labeling.label --split {split})")
    assert_no_test_leak(tv["train"] + tv["val"])

    test_rows.sort(key=lambda x: x["id"])
    write_jsonl(FINAL / "test.jsonl", test_rows)
    for split, rows in tv.items():
        write_jsonl(FINAL / f"{split}.jsonl", rows)

    n = len(test_rows)
    stats = {
        "reviewer": args.reviewer,
        "label_source": source,
        "test_rows": n,
        "rows_with_any_correction": n_changed_rows,
        "share_rows_corrected": round(n_changed_rows / n, 4),
        "teacher_exact_match_vs_reviewed": round(1 - n_changed_rows / n, 4),
        "corrections_per_field": changed,
        "teacher_field_accuracy_vs_reviewed": {f: round(1 - c / n, 4) for f, c in changed.items()},
        "teacher": sorted({teacher[r["id"]]["teacher"] for r in test_rows}),
        "prompt_version": sorted({teacher[r["id"]]["prompt_version"] for r in test_rows}),
        "train_rows": len(tv["train"]), "val_rows": len(tv["val"]),
        "corrections": change_examples,
    }
    (FINAL / "review_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "corrections"}, indent=2))
    print(f"-> wrote {FINAL}/test.jsonl, train.jsonl, val.jsonl, review_stats.json")


if __name__ == "__main__":
    main()
