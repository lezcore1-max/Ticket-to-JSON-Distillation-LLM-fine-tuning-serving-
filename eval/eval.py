"""Score any model's predictions against the reviewed test set. One scorer for every row of the
results table.

Predictions file (JSONL), one line per test message:
  {"id": "...", "raw_output": "<exactly what the model returned>",
   "latency_s": 0.84, "tokens": {"prompt": 2400, "output": 60}}      # latency/tokens optional

Definitions (also in the README):
  strict valid JSON   raw_output parses as a JSON object as-is
  lenient valid JSON  parses after stripping ``` fences / surrounding text (first balanced {...})
  schema-valid        lenient-parsed object passes labeling/validate.py (types, enums, intent/category match)
  field accuracy      per field, on the lenient-parsed object; unparseable or missing output = wrong on
                      every field. "Field acc" in the table = mean over the 6 fields
  exact match         schema-valid AND all 6 fields correct
  comparison          case-insensitive; order_id ignores a leading '#'; product is normalized
                      (lowercase, 'profile' -> 'account', articles/possessives dropped)
  95% CI              percentile bootstrap over test messages (2,000 resamples, fixed seed)

  python -m eval.eval --pred eval/predictions/qwen_prompted.jsonl --name "Small model, prompted" \
      --hardware "RTX 3050 Laptop 6GB, HF transformers, batch 1" [--price-key gemini-3.5-flash-lite]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from labeling.splits import ROOT, load_jsonl
from labeling.validate import FIELDS, validate_label

GOLD = ROOT / "data" / "final" / "test.jsonl"
RESULTS = ROOT / "eval" / "results"
PRICES = ROOT / "eval" / "prices.json"
SEED, N_BOOT = 7, 2000


# ----------------------------------------------------------------------------- parsing
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def parse_strict(raw):
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except (TypeError, ValueError):
        return None


def first_balanced_object(s: str) -> str | None:
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        start = s.find("{", start + 1)
    return None


def parse_lenient(raw):
    obj = parse_strict(raw)
    if obj is not None or not isinstance(raw, str):
        return obj
    m = FENCE_RE.search(raw)
    candidates = ([m.group(1)] if m else []) + [raw]
    for c in candidates:
        obj = parse_strict(c.strip())
        if obj is not None:
            return obj
        sub = first_balanced_object(c)
        if sub and (obj := parse_strict(sub)) is not None:
            return obj
    return None


# ----------------------------------------------------------------------------- normalization
def _null(v):
    if v is None:
        return None
    v = str(v).strip()
    return None if v.lower() in ("", "null", "none", "n/a") else v


def norm_value(field: str, v):
    v = _null(v)
    if v is None:
        return None
    v = " ".join(v.lower().split())
    if field == "order_id":
        return v.lstrip("#").strip()
    if field == "product":
        v = v.replace("-", " ")
        v = re.sub(r"^(the|a|an|my|your|our)\s+", "", v)
        v = re.sub(r"\bprofile\b", "account", v)
        return " ".join(v.split())
    return v


# ----------------------------------------------------------------------------- scoring
def score_rows(gold_rows: list[dict], preds: dict[str, dict]) -> list[dict]:
    """Per-message outcome records (the unit the bootstrap resamples)."""
    out = []
    for g in gold_rows:
        p = preds.get(g["id"])
        raw = p.get("raw_output") if p else None
        strict = parse_strict(raw) is not None
        obj = parse_lenient(raw)
        schema_ok = obj is not None and not validate_label(obj)
        correct = {f: obj is not None and norm_value(f, obj.get(f)) == norm_value(f, g["label"][f]) for f in FIELDS}
        out.append({
            "id": g["id"], "missing": p is None, "strict": strict, "lenient": obj is not None,
            "schema": schema_ok, "correct": correct, "exact": schema_ok and all(correct.values()),
            "gold": g["label"], "pred": {f: (obj.get(f) if obj else None) for f in FIELDS},
            "latency_s": p.get("latency_s") if p else None,
            "tokens": p.get("tokens") if p else None,
        })
    return out


def headline(rows):
    n = len(rows)
    fa = {f: sum(r["correct"][f] for r in rows) / n for f in FIELDS}
    return {"exact_match": sum(r["exact"] for r in rows) / n, "field_acc_mean": sum(fa.values()) / len(FIELDS),
            **{f"acc_{f}": v for f, v in fa.items()}}


def bootstrap(rows, n_boot=N_BOOT, seed=SEED):
    rnd = random.Random(seed)
    samples = [headline([rows[rnd.randrange(len(rows))] for _ in rows]) for _ in range(n_boot)]
    ci = {}
    for k in samples[0]:
        vals = sorted(s[k] for s in samples)
        ci[k] = [round(vals[int(0.025 * n_boot)], 4), round(vals[int(0.975 * n_boot) - 1], 4)]
    return ci


def per_class_acc(rows, field):
    by = {}
    for r in rows:
        by.setdefault(r["gold"][field], []).append(r["correct"][field])
    return {k: {"acc": round(sum(v) / len(v), 4), "n": len(v)} for k, v in sorted(by.items())}


def macro_f1(rows, field="intent"):
    labels = sorted({r["gold"][field] for r in rows})
    f1s = []
    for lab in labels:
        tp = sum(r["gold"][field] == lab and norm_value(field, r["pred"][field]) == lab for r in rows)
        fp = sum(r["gold"][field] != lab and norm_value(field, r["pred"][field]) == lab for r in rows)
        fn = sum(r["gold"][field] == lab and norm_value(field, r["pred"][field]) != lab for r in rows)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return round(sum(f1s) / len(f1s), 4)


def presence_stats(rows, field):
    has = [r for r in rows if r["gold"][field] is not None]
    nul = [r for r in rows if r["gold"][field] is None]
    return {
        "acc_when_present": round(sum(r["correct"][field] for r in has) / len(has), 4) if has else None,
        "n_present": len(has),
        "false_alarm_rate": round(sum(norm_value(field, r["pred"][field]) is not None for r in nul) / len(nul), 4) if nul else None,
        "n_null": len(nul),
    }


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def cost_per_1k(rows, price_key):
    if not price_key:
        return None
    prices = json.loads(PRICES.read_text(encoding="utf-8"))
    pr = prices["models"][price_key]
    toks = [r["tokens"] for r in rows if r["tokens"]]
    if not toks:
        return None
    pin = statistics.mean(t.get("prompt", 0) for t in toks)
    pout = statistics.mean(t.get("output", 0) for t in toks)
    usd = 1000 * (pin * pr["input_per_1m_usd"] + pout * pr["output_per_1m_usd"]) / 1e6
    return {"usd": round(usd, 4), "mean_prompt_tokens": round(pin, 1), "mean_output_tokens": round(pout, 1),
            "price_key": price_key, "price_source": prices.get("source"), "price_checked": prices.get("checked")}


def evaluate(pred_path: Path, name: str, hardware: str | None = None, price_key: str | None = None,
             notes: str | None = None) -> dict:
    gold = load_jsonl(GOLD)
    preds = {p["id"]: p for p in load_jsonl(pred_path)}
    unknown = set(preds) - {g["id"] for g in gold}
    if unknown:
        raise SystemExit(f"{len(unknown)} prediction ids are not in the test set, e.g. {sorted(unknown)[:3]}")
    rows = score_rows(gold, preds)
    n = len(rows)
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    h = headline(rows)
    return {
        "name": name, "predictions": str(pred_path.relative_to(ROOT)).replace("\\", "/") if pred_path.is_relative_to(ROOT) else str(pred_path),
        "gold": "data/final/test.jsonl", "n": n, "hardware": hardware, "notes": notes,
        "missing_predictions": sum(r["missing"] for r in rows),
        "valid_json_strict": round(sum(r["strict"] for r in rows) / n, 4),
        "valid_json_lenient": round(sum(r["lenient"] for r in rows) / n, 4),
        "schema_valid": round(sum(r["schema"] for r in rows) / n, 4),
        **{k: round(v, 4) for k, v in h.items()},
        "ci95": bootstrap(rows),
        "urgency_by_class": per_class_acc(rows, "urgency"),
        "sentiment_by_class": per_class_acc(rows, "sentiment"),
        "intent_macro_f1": macro_f1(rows),
        "order_id": presence_stats(rows, "order_id"),
        "product": presence_stats(rows, "product"),
        "latency_s": {"p50": round(pct(lat, 0.5), 3), "p95": round(pct(lat, 0.95), 3), "n": len(lat)} if lat else None,
        "cost_per_1k_requests": cost_per_1k(rows, price_key),
        "errors_sample": [{"id": r["id"], "gold": r["gold"], "pred": r["pred"]} for r in rows if not r["exact"]][:15],
    }


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--name", required=True, help="row name in the results table")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--price-key", default=None, help="key in eval/prices.json (API models)")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    res = evaluate(args.pred, args.name, args.hardware, args.price_key, args.notes)
    if not args.no_save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / f"{slug(args.name)}.json"
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"-> {out}")
    show = {k: res[k] for k in ("name", "n", "missing_predictions", "valid_json_strict", "valid_json_lenient",
                                "schema_valid", "field_acc_mean", "exact_match", "intent_macro_f1")}
    show["ci95_exact_match"] = res["ci95"]["exact_match"]
    show["per_field"] = {f: res[f"acc_{f}"] for f in FIELDS}
    show["urgency_by_class"] = res["urgency_by_class"]
    show["latency_s"] = res["latency_s"]
    show["cost_per_1k_requests"] = res["cost_per_1k_requests"]
    print(json.dumps(show, indent=2))


if __name__ == "__main__":
    main()
