"""FastAPI gateway in front of vLLM for Ticket-to-JSON extraction.

Validates every response against labeling/schema.json and labeling/taxonomy.json
using labeling/validate.py, and optionally passes the JSON schema to vLLM for
constrained decoding.

  uvicorn serving.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from labeling.validate import SCHEMA, parse_and_validate
from training.dataset import build_compact_prompt, format_user_message

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "ticket-json")
SYSTEM_PROMPT = build_compact_prompt()
JSON_SCHEMA = {k: v for k, v in SCHEMA.items() if k not in ("$schema", "title")}


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    guided_json: bool = True


class PredictResponse(BaseModel):
    label: dict | None
    valid: bool
    errors: list[str]
    raw_output: str
    latency_ms: float
    tokens: dict[str, int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=VLLM_URL, timeout=60.0)
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Ticket-to-JSON API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    try:
        r = await app.state.client.get("/health")
        ok = r.status_code == 200
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"vLLM unreachable: {e}") from e
    if not ok:
        raise HTTPException(status_code=503, detail=f"vLLM unhealthy: {r.status_code}")
    return {"status": "ok", "vllm_url": VLLM_URL, "model": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    payload: dict = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_message(req.text)},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }
    if req.guided_json:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "TicketLabel", "strict": True, "schema": JSON_SCHEMA},
        }

    t0 = time.perf_counter()
    try:
        r = await app.state.client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"vLLM error {e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"vLLM connection error: {e}") from e
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    data = r.json()
    raw = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    tokens = {
        "prompt": int(usage.get("prompt_tokens", 0)),
        "output": int(usage.get("completion_tokens", 0)),
    }
    obj, errors = parse_and_validate(raw)
    return PredictResponse(
        label=obj if not errors else None,
        valid=not errors,
        errors=errors,
        raw_output=raw,
        latency_ms=latency_ms,
        tokens=tokens,
    )

