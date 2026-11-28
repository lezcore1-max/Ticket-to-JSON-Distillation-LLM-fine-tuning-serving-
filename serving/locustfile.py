"""Locust load test for the Ticket-to-JSON FastAPI + vLLM service.

Draws realistic ticket texts from data/final/val.jsonl (never test.jsonl) and
verifies both HTTP 200 and schema validity on every response.

  locust -f serving/locustfile.py --host http://localhost:8080
  locust -f serving/locustfile.py --host http://localhost:8080 --headless -u 10 -r 5 -t 60s
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent.parent
VAL_FILE = ROOT / "data" / "final" / "val.jsonl"
TEXTS = [json.loads(line)["text"] for line in VAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


class TicketUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        text = random.choice(TEXTS)
        with self.client.post(
            "/predict",
            json={"text": text, "guided_json": True},
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:150]}")
                return
            body = resp.json()
            if not body.get("valid"):
                resp.failure(f"invalid schema: {body.get('errors')}")
            else:
                resp.success()

