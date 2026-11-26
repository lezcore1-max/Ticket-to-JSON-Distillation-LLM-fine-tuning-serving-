"""Day 2 quality check of the full teacher-labeled dataset.

Per split: coverage (labeled / rejected / missing), agreement with reference signals
(Bitext intent+category, injected order ids, product answer key), urgency/sentiment
distributions, urgency consistency within each intent, most common intent confusions,
and token usage. Also re-runs the test-leak guard on the labeled train/val files.

  python -m labeling.label_stats            # writes labeling/label_report.md + data/labeled/stats.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from labeling.splits import PREPARED, ROOT, assert_no_test_leak, load_jsonl

LABELED = ROOT / "data" / "labeled"
SPLITS = ("train", "val", "test")


def norm_prod(p):
    return None if p is None else " ".join(p.lower().replace("-", " ").split())


def ratio(a, b):
    return round(a / b, 4) if b else None


def fmt(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "n/a"


def split_stats(split: str) -> dict:
    ref = {r["id"]: r for r in load_jsonl(PREPARED / f"{split}.jsonl")}
    path = LABELED / f"{split}.jsonl"
    labs = load_jsonl(path) if path.exists() else []
    rej_path = path.with_suffix(".rejects.jsonl")
    rejects = load_jsonl(rej_path) if rej_path.exists() else []
    rows = [(ref[l["id"]], l["label"], l) for l in labs]
    n = len(rows)

    with_oid = [(r, y) for r, y, _ in rows if r["gt_order_id"]]
    no_oid = [(r, y) for r, y, _ in rows if not r["gt_order_id"]]
    with_prod = [(r, y) for r, y, _ in rows if r["hint_product"]]
    no_prod = [(r, y) for r, y, _ in rows if not r["hint_product"]]

    # urgency consistency: share of each intent's messages that got that intent's most common urgency
    by_intent = defaultdict(list)
    for r, y, _ in rows:
        by_intent[y["intent"]].append(y["urgency"])
    consistency = {k: (Counter(v).most_common(1)[0][0], ratio(Counter(v).most_common(1)[0][1], len(v)), len(v))
                   for k, v in by_intent.items()}
    overall_cons = ratio(sum(Counter(v).most_common(1)[0][1] for v in by_intent.values()), n)

    per_intent_agree = defaultdict(lambda: [0, 0])
    for r, y, _ in rows:
        per_intent_agree[r["bitext_intent"]][1] += 1
        per_intent_agree[r["bitext_intent"]][0] += r["bitext_intent"] == y["intent"]

    tok_p = sum(l["usage_per_row"]["prompt_tokens"] for _, _, l in rows)
    tok_o = sum(l["usage_per_row"]["output_tokens"] + l["usage_per_row"]["thoughts_tokens"] for _, _, l in rows)
    approx_requests = round(sum(1 / l["batch"]["batch_size"] for _, _, l in rows))

    return {
        "prepared": len(ref), "labeled": n, "rejected": len(rejects),
        "missing": len(ref) - n - len(rejects),
        "teachers": dict(Counter(l["teacher"] for _, _, l in rows)),
        "prompt_versions": dict(Counter(l["prompt_version"] for _, _, l in rows)),
        "needed_retry": sum(l["attempts"] > 1 for _, _, l in rows),
        "intent_agree": [sum(r["bitext_intent"] == y["intent"] for r, y, _ in rows), n],
        "category_agree": [sum(r["bitext_category"] == y["category"] for r, y, _ in rows), n],
        "order_id_correct": [sum(y["order_id"] == r["gt_order_id"] for r, y in with_oid), len(with_oid)],
        "order_id_false_pos": [sum(y["order_id"] is not None for r, y in no_oid), len(no_oid)],
        "product_correct": [sum(norm_prod(y["product"]) == norm_prod(r["hint_product"]) for r, y in with_prod), len(with_prod)],
        "product_extra": [sum(y["product"] is not None for r, y in no_prod), len(no_prod)],
        "urgency": dict(Counter(y["urgency"] for _, y, _ in rows).most_common()),
        "sentiment": dict(Counter(y["sentiment"] for _, y, _ in rows).most_common()),
        "urgency_consistency_overall": overall_cons,
        "urgency_consistency_by_intent": {k: {"mode": m, "share": s, "n": c}
                                          for k, (m, s, c) in sorted(consistency.items(), key=lambda kv: kv[1][1])},
        "intent_agree_by_bitext_intent": {k: ratio(a, b) for k, (a, b) in
                                          sorted(per_intent_agree.items(), key=lambda kv: kv[1][0] / kv[1][1])},
        "top_confusions": [[f"{a} -> {b}", c] for (a, b), c in Counter(
            (r["bitext_intent"], y["intent"]) for r, y, _ in rows if r["bitext_intent"] != y["intent"]).most_common(10)],
        "product_extra_examples": Counter(norm_prod(y["product"]) for r, y in no_prod if y["product"]).most_common(10),
        "reject_examples": [(x["text"], x["errors"][0][:100]) for x in rejects[:5]],
        "tokens": {"prompt": round(tok_p), "output_incl_thinking": round(tok_o), "approx_requests": approx_requests},
    }


def main() -> None:
    stats = {s: split_stats(s) for s in SPLITS}

    # leak guard on what we will train on
    tv = [l for s in ("train", "val") if (LABELED / f"{s}.jsonl").exists() for l in load_jsonl(LABELED / f"{s}.jsonl")]
    assert_no_test_leak(tv)
    stats["leak_check"] = "passed: no labeled train/val message shares id or text with test"

    (LABELED / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8", newline="\n")

    L = ["# Full labeling report", "",
         "Generated by `labeling/label_stats.py` (numbers also in `data/labeled/stats.json`).", "",
         "| | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    def row(name, f):
        L.append(f"| {name} | " + " | ".join(f(stats[s]) for s in SPLITS) + " |")
    row("labeled / prepared", lambda d: fmt(d["labeled"], d["prepared"]))
    row("rejected / missing", lambda d: f"{d['rejected']} / {d['missing']}")
    row("teacher, prompt", lambda d: ", ".join(d["teachers"]) + " " + ",".join(d["prompt_versions"]))
    row("needed a retry", lambda d: str(d["needed_retry"]))
    row("intent = Bitext", lambda d: fmt(*d["intent_agree"]))
    row("category = Bitext", lambda d: fmt(*d["category_agree"]))
    row("order_id correct", lambda d: fmt(*d["order_id_correct"]))
    row("order_id false alarms", lambda d: fmt(*d["order_id_false_pos"]))
    row("product correct (known)", lambda d: fmt(*d["product_correct"]))
    row("product given, none known", lambda d: fmt(*d["product_extra"]))
    row("urgency low/med/high", lambda d: "/".join(str(d["urgency"].get(k, 0)) for k in ("low", "medium", "high")))
    row("sentiment neg/neu/pos", lambda d: "/".join(str(d["sentiment"].get(k, 0)) for k in ("negative", "neutral", "positive")))
    row("urgency consistency", lambda d: f"{100 * (d['urgency_consistency_overall'] or 0):.1f}%")
    row("tokens (prompt / output)", lambda d: f"{d['tokens']['prompt']:,} / {d['tokens']['output_incl_thinking']:,}")
    row("~requests", lambda d: str(d["tokens"]["approx_requests"]))
    t = stats["train"]
    L += ["", f"Leak check: {stats['leak_check']}", "",
          "## Train: most common intent disagreements (Bitext -> teacher)", ""]
    L += [f"- {k}: {c}" for k, c in t["top_confusions"]] or ["_none_"]
    L += ["", "## Train: lowest intent agreement by Bitext intent", ""]
    L += [f"- {k}: {100 * v:.1f}%" for k, v in list(t["intent_agree_by_bitext_intent"].items())[:6]]
    L += ["", "## Train: least consistent urgency by intent (most common level, share, n)", ""]
    L += [f"- {k}: {v['mode']} {100 * v['share']:.0f}% of {v['n']}" for k, v in list(t["urgency_consistency_by_intent"].items())[:8]]
    L += ["", "## Train: products given where none is known (most common)", ""]
    L += [f"- {p}: {c}" for p, c in t["product_extra_examples"]] or ["_none_"]
    L += ["", "## Rejected examples", ""]
    L += [f"- [{s}] {x} — {e}" for s in SPLITS for x, e in stats[s]["reject_examples"]] or ["_none_"]

    (ROOT / "labeling" / "label_report.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
