"""Build the hand-check sheet for the 200 test messages.

Each row shows the message, the teacher's label, and the reference signals we have (Bitext's
intent, the known order id, the known product). Rows where the teacher disagrees with a reference
(or is otherwise worth a second look) get check_first=YES and are sorted to the top.

The final_* columns are pre-filled with the teacher's label: edit them where the teacher is
wrong, then put "y" in `reviewed` for every row you have checked. Empty cell = null.
`category` is not in the sheet: it is fixed by the intent and filled in by merge_review.py.

  python -m labeling.make_review_sheet      # -> data/review/test_review.csv
"""
from __future__ import annotations

import csv

from labeling.splits import PREPARED, ROOT, load_jsonl, load_test_ids

OUT = ROOT / "data" / "review" / "test_review.csv"
FIELDS = ["intent", "urgency", "product", "order_id", "sentiment"]
COLUMNS = (["row", "id", "check_first", "flags", "text", "bitext_intent", "known_order_id", "known_product"]
           + [f"teacher_{f}" for f in FIELDS] + [f"final_{f}" for f in FIELDS] + ["reviewed", "notes"])


def norm(s):
    return None if s is None else " ".join(str(s).lower().split())


def flags_for(ref: dict, y: dict) -> list[str]:
    f = []
    if y["intent"] != ref["bitext_intent"]:
        f.append(f"intent differs from Bitext ({ref['bitext_intent']})")
    if (ref["gt_order_id"] or None) != y["order_id"]:
        f.append(f"order_id differs from known ({ref['gt_order_id']})")
    if norm(ref["hint_product"]) != norm(y["product"]):
        f.append(f"product differs from known ({ref['hint_product']})")
    if y["urgency"] == "high":
        f.append("high urgency (rare class)")
    if y["sentiment"] != "neutral":
        f.append(f"{y['sentiment']} sentiment (rare class)")
    return f


def main() -> None:
    ref = {r["id"]: r for r in load_jsonl(PREPARED / "test.jsonl")}
    labs = {l["id"]: l for l in load_jsonl(ROOT / "data" / "labeled" / "test.jsonl")}
    test_ids = load_test_ids()
    missing = test_ids - set(labs)
    if missing:
        raise SystemExit(f"{len(missing)} test messages have no teacher label yet; finish labeling first")

    rows = []
    for i in sorted(test_ids):
        r, y = ref[i], labs[i]["label"]
        fl = flags_for(r, y)
        row = {"id": i, "check_first": "YES" if fl else "", "flags": "; ".join(fl), "text": r["text"],
               "bitext_intent": r["bitext_intent"], "known_order_id": r["gt_order_id"] or "",
               "known_product": r["hint_product"] or "", "reviewed": "", "notes": ""}
        for f in FIELDS:
            v = y[f] if y[f] is not None else ""
            row[f"teacher_{f}"] = v
            row[f"final_{f}"] = v
        rows.append(row)
    # flagged rows first; inside each group, group by intent so similar messages sit together
    rows.sort(key=lambda r: (r["check_first"] != "YES", r["teacher_intent"], r["id"]))
    for n, r in enumerate(rows, 1):
        r["row"] = n

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig so Excel shows £/€ correctly
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    n_flag = sum(r["check_first"] == "YES" for r in rows)
    print(f"wrote {OUT} : {len(rows)} rows, {n_flag} flagged check_first=YES (sorted to the top)")


if __name__ == "__main__":
    main()
