"""Reference row: always predict the most common value of each field in the TRAINING labels.

Shows what a model with zero understanding scores (e.g. always "medium" urgency). Category is
taken from the majority intent so the output is schema-valid. Uses train only, never test.

  python -m eval.predict_majority      # -> eval/predictions/majority.jsonl
"""
from __future__ import annotations

import json
from collections import Counter

from labeling.splits import ROOT, load_jsonl, write_jsonl
from labeling.validate import INTENT_TO_CATEGORY, validate_label

OUT = ROOT / "eval" / "predictions" / "majority.jsonl"


def main() -> None:
    train = [r["label"] for r in load_jsonl(ROOT / "data" / "final" / "train.jsonl")]
    pick = {f: Counter(json.dumps(l[f]) for l in train).most_common(1)[0][0] for f in ("intent", "urgency", "product", "order_id", "sentiment")}
    label = {f: json.loads(v) for f, v in pick.items()}
    label["category"] = INTENT_TO_CATEGORY[label["intent"]]
    assert not validate_label(label), validate_label(label)
    test = load_jsonl(ROOT / "data" / "final" / "test.jsonl")
    write_jsonl(OUT, [{"id": t["id"], "raw_output": json.dumps(label), "latency_s": 0.0} for t in test])
    print("majority label from train:", label, "->", OUT)


if __name__ == "__main__":
    main()
