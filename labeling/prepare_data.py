"""Day 1: build the 3,500-message pool and the train/val/test split.

Steps
  1. Load Bitext customer-support dataset (26,872 synthetic messages, 27 intents).
  2. Deduplicate on the *template* text (before placeholder filling), so two copies of
     one template can never land in different splits.
  3. Stratified sample of 3,500 messages by Bitext intent.
  4. Stratified split 3,000 / 300 / 200 (train / val / test), done BEFORE any labeling.
  5. Fill {{placeholders}} with realistic values. Order numbers get known ground truth
     (gt_order_id); account tiers give a product hint (hint_product) used to sanity-check
     the teacher.

Run from repo root:  python -m labeling.prepare_data
"""
from __future__ import annotations

import json
import random
import re
import zlib
from collections import Counter

from datasets import load_dataset

from labeling.splits import PREPARED, ROOT, TEST_IDS_FILE, assert_no_test_leak, norm_text, write_jsonl

SEED = 13
N_TOTAL, N_VAL, N_TEST = 3500, 300, 200
DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

PH_RE = re.compile(r"\{\{([^}]+)\}\}")
TIERS = ["premium", "gold", "platinum", "standard", "pro", "business", "freemium", "family", "student"]
NAMES = ["Priya Sharma", "John Miller", "Aisha Khan", "Carlos Gomez", "Mei Lin", "Ahmed Hassan",
         "Sarah O'Brien", "Rahul Verma", "Emma Schmidt", "Kwame Mensah", "Yuki Tanaka", "Olga Petrova"]
CITIES = ["Mumbai", "Berlin", "Austin", "Toronto", "Lagos", "Sydney", "Madrid", "Seoul", "Chicago", "Delhi"]
COUNTRIES = ["India", "Germany", "Canada", "Brazil", "Australia", "Spain", "Japan", "Kenya", "Mexico", "France"]
CURRENCIES = ["$", "€", "£", "₹"]
# Tier words Bitext itself writes before "account" (found by counting the word before "account"
# in the raw data: standard 389, pro 265, freemium 247, platinum 243, gold 231, premium 230),
# plus our own placeholder tiers. Tolerates small typos in "account" (acount, accout).
TIER_RE = re.compile(r"\b(" + "|".join(sorted(set(TIERS) | {"standard", "pro", "freemium", "platinum",
                                                             "gold", "premium"})) + r")\s+(?:ac+o?u?n?t|profile)\b", re.I)
# Order numbers Bitext wrote literally (not via placeholder), sometimes glued: "cancel purchase732201349959"
LITERAL_ORDER_RE = re.compile(r"\b(?:order|purchase)\s*(?:number|no\.?)?\s*#?\s*(\d{5,})\b", re.I)


def make_order_number(rng: random.Random) -> tuple[str, str]:
    """Return (surface form as typed by customer, canonical gt order_id)."""
    kind = rng.choice(["hash", "digits", "ord", "alnum"])
    if kind == "hash":
        d = str(rng.randint(1000, 999999))
        return "#" + d, d
    if kind == "digits":
        d = str(rng.randint(100000, 99999999))
        return d, d
    if kind == "ord":
        d = f"ORD-{rng.randint(10000, 99999)}"
        return d, d
    d = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2)) + str(rng.randint(100000, 999999))
    return d, d


def fill(template: str, rng: random.Random) -> tuple[str, str | None, str | None]:
    """Fill placeholders. Returns (text, gt_order_id, hint_product)."""
    gt_order, hint_product = None, None
    out, pos = [], 0
    for m in PH_RE.finditer(template):
        name = m.group(1)
        prefix = template[pos:m.start()]
        out.append(prefix)
        if name == "Order Number":
            surface, gt_order = make_order_number(rng)
            if prefix.endswith("#"):          # template already typed the '#'
                surface = surface.lstrip("#")
            value = surface
        elif name in ("Account Type", "Account Category"):
            tier = rng.choice(TIERS)
            value = tier.title() if rng.random() < 0.3 else tier
            after = template[m.end():m.end() + 9].lower()
            hint_product = f"{tier} account" if after.strip().startswith("account") else tier
        elif name == "Person Name":
            value = rng.choice(NAMES)
        elif name == "Refund Amount":
            value = rng.choice([str(rng.randint(5, 500)), f"{rng.randint(5, 500)}.{rng.randint(0, 99):02d}"])
        elif name == "Currency Symbol":
            value = rng.choice(CURRENCIES)
        elif name == "Delivery City":
            value = rng.choice(CITIES)
        elif name == "Delivery Country":
            value = rng.choice(COUNTRIES)
        elif name == "Invoice Number":
            value = f"INV-{rng.randint(100000, 999999)}"   # NOT an order id (hard negative)
        else:
            raise ValueError(f"unknown placeholder {name!r} in {template!r}")
        out.append(value)
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), gt_order, hint_product


def largest_remainder(total: int, sizes: dict[str, int]) -> dict[str, int]:
    """Allocate `total` across groups proportionally to `sizes`, summing exactly to total."""
    s = sum(sizes.values())
    raw = {k: total * v / s for k, v in sizes.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def main() -> None:
    rng = random.Random(SEED)
    df = load_dataset(DATASET, split="train", cache_dir=str(ROOT / "data" / "raw")).to_pandas()
    df["id"] = [f"bx-{i:05d}" for i in range(len(df))]
    n_raw = len(df)

    # 2. dedupe on normalized template text, treating all placeholders as identical
    #    ("a {{Account Type}} account" == "a {{Account Category}} account")
    df["norm"] = df["instruction"].map(lambda t: norm_text(PH_RE.sub("{{X}}", t)))
    df = df.drop_duplicates("norm", keep="first").reset_index(drop=True)
    n_dedup = len(df)

    # 3. stratified sample by intent (equal share per intent; all intents have >= 900 rows)
    intents = sorted(df["intent"].unique())
    per_intent = largest_remainder(N_TOTAL, {i: 1 for i in intents})
    pools = {i: df[df["intent"] == i].sample(n=per_intent[i], random_state=SEED) for i in intents}

    # 4. stratified split
    sizes = {i: len(p) for i, p in pools.items()}
    test_q = largest_remainder(N_TEST, sizes)
    val_q = largest_remainder(N_VAL, sizes)
    rows = {"train": [], "val": [], "test": []}
    for i in intents:
        g = pools[i].to_dict("records")
        bounds = {"test": g[: test_q[i]], "val": g[test_q[i]: test_q[i] + val_q[i]], "train": g[test_q[i] + val_q[i]:]}
        for split, recs in bounds.items():
            for r in recs:
                # per-row RNG keyed on id -> filling is reproducible and independent of order
                rrng = random.Random(zlib.crc32(f"{SEED}:{r['id']}".encode()))
                text, gt_order, hint_product = fill(r["instruction"], rrng)
                if hint_product is None and (m := TIER_RE.search(text)):
                    hint_product = f"{m.group(1).lower()} account"   # tier written literally by Bitext
                if gt_order is None and (m := LITERAL_ORDER_RE.search(text)):
                    gt_order = m.group(1)                            # order number written literally
                rows[split].append({
                    "id": r["id"],
                    "split": split,
                    "text": text,
                    "template": r["instruction"],
                    "bitext_intent": r["intent"],
                    "bitext_category": r["category"].lower(),
                    "flags": r["flags"],
                    "gt_order_id": gt_order,
                    "hint_product": hint_product,
                })
    for split in rows:
        rng.shuffle(rows[split])

    # guard: test set disjoint from train+val by id, filled text, and template
    assert_no_test_leak(rows["train"] + rows["val"], test_rows=rows["test"])

    for split, recs in rows.items():
        write_jsonl(PREPARED / f"{split}.jsonl", recs)
    TEST_IDS_FILE.write_text("\n".join(sorted(r["id"] for r in rows["test"])) + "\n", encoding="utf-8", newline="\n")

    all_rows = [r for recs in rows.values() for r in recs]
    stats = {
        "seed": SEED,
        "raw_rows": n_raw,
        "after_template_dedupe": n_dedup,
        "sampled": len(all_rows),
        "split_sizes": {k: len(v) for k, v in rows.items()},
        "n_intents": len(intents),
        "per_split_intent_min_max": {k: [min(Counter(r["bitext_intent"] for r in v).values()),
                                         max(Counter(r["bitext_intent"] for r in v).values())] for k, v in rows.items()},
        "with_gt_order_id": sum(r["gt_order_id"] is not None for r in all_rows),
        "with_hint_product": sum(r["hint_product"] is not None for r in all_rows),
        "placeholders_filled": dict(Counter(m for r in all_rows for m in PH_RE.findall(r["template"]))),
    }
    (PREPARED / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
