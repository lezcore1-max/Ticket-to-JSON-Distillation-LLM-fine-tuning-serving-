"""Compute self-hosted cost per 1K requests from Day 5 Locust throughput and eval/prices.json.

Updates cost_per_1k_requests in:
  - eval/results/small_model_vllm_guided_json.json (from serving/loadtest_stats.csv + AWS g4dn.xlarge T4)
  - eval/results/small_model_gguf_q4_k_m_cpu.json (HF Spaces free cpu-basic tier)

  python -m eval.compute_cost
"""
from __future__ import annotations

import csv
import json

from labeling.splits import ROOT

PRICES = ROOT / "eval" / "prices.json"
LOCUST_CSV = ROOT / "serving" / "loadtest_stats.csv"
VLLM_RES = ROOT / "eval" / "results" / "small_model_vllm_guided_json.json"
GGUF_RES = ROOT / "eval" / "results" / "small_model_gguf_q4_k_m_cpu.json"


def main() -> None:
    prices = json.loads(PRICES.read_text(encoding="utf-8"))["self_hosted"]

    with open(LOCUST_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    agg = next(r for r in rows if r["Name"] == "Aggregated")
    rps = float(agg["Requests/s"])

    gpu_ref = prices["aws_g4dn_xlarge_t4"]
    req_per_hour = rps * 3600.0
    usd_per_1k = 1000.0 * gpu_ref["usd_per_hour"] / req_per_hour

    vllm = json.loads(VLLM_RES.read_text(encoding="utf-8"))
    vllm["cost_per_1k_requests"] = {
        "usd": round(usd_per_1k, 4),
        "rps": round(rps, 2),
        "requests_per_hour": round(req_per_hour),
        "gpu_usd_per_hour": gpu_ref["usd_per_hour"],
        "reference": gpu_ref["source"],
    }
    VLLM_RES.write_text(json.dumps(vllm, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    cpu_ref = prices["hf_spaces_cpu_basic"]
    gguf = json.loads(GGUF_RES.read_text(encoding="utf-8"))
    gguf["cost_per_1k_requests"] = {
        "usd": cpu_ref["usd_per_hour"],
        "reference": cpu_ref["source"],
    }
    GGUF_RES.write_text(json.dumps(gguf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps({"vllm_cost_per_1k": vllm["cost_per_1k_requests"], "gguf_cpu_cost_per_1k": gguf["cost_per_1k_requests"]}, indent=2))


if __name__ == "__main__":
    main()
