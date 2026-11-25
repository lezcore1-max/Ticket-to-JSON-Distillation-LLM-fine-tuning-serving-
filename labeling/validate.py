"""Single definition of a valid label, shared by labeling, eval and serving.

A label is valid iff it
  1. passes labeling/schema.json (types, enums, required keys, no extra keys), and
  2. is internally consistent: category matches the intent's category in taxonomy.json,
     and order_id is in canonical form (no leading '#', no surrounding whitespace).
"""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
TAXONOMY = json.loads((HERE / "taxonomy.json").read_text(encoding="utf-8"))["intents"]
INTENT_TO_CATEGORY = {k: v["category"] for k, v in TAXONOMY.items()}
FIELDS = SCHEMA["required"]

_validator = Draft202012Validator(SCHEMA)

# Fail fast if schema.json and taxonomy.json drift apart.
_schema_intents = set(SCHEMA["properties"]["intent"]["enum"])
_schema_cats = set(SCHEMA["properties"]["category"]["enum"])
assert _schema_intents == set(INTENT_TO_CATEGORY), "intent enum != taxonomy intents"
assert _schema_cats == set(INTENT_TO_CATEGORY.values()), "category enum != taxonomy categories"


def validate_label(obj) -> list[str]:
    """Return a list of problems; empty list means valid."""
    errors = [e.message for e in _validator.iter_errors(obj)]
    if errors:
        return errors
    if INTENT_TO_CATEGORY[obj["intent"]] != obj["category"]:
        errors.append(
            f"category '{obj['category']}' inconsistent with intent '{obj['intent']}' "
            f"(expected '{INTENT_TO_CATEGORY[obj['intent']]}')"
        )
    oid = obj["order_id"]
    if oid is not None and (oid != oid.strip() or oid.startswith("#") or not oid):
        errors.append(f"order_id not canonical: {oid!r}")
    return errors


def parse_and_validate(text: str):
    """Parse model output text. Returns (obj_or_None, errors)."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        return None, [f"invalid JSON: {e}"]
    return obj, validate_label(obj)
