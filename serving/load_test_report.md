# Day 5 Serving & Load-Test Report

Evaluated on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** running `vllm/vllm-openai:latest` (`bfloat16`, `--max-model-len 1024`, `--gpu-memory-utilization 0.78`, `--max-num-seqs 16`) behind FastAPI (`serving/app.py`) with JSON-schema guided decoding (`labeling/schema.json`) + `parse_and_validate()` (`labeling/validate.py`).

## Single-Request Test Evaluation (`data/final/test.jsonl`, `n=200`)

| Engine | Exact match | Schema-valid | p50 latency | p95 latency |
|---|---|---|---|---|
| HF `transformers` (`bf16`, `batch=1`) | 92.0% | 100.0% | 1.903 s | 4.167 s |
| **vLLM + FastAPI (`guided_json`)** | **92.0%** | **100.0%** | **0.873 s** | **1.000 s** |

## Locust Load Test (`serving/locustfile.py`, 10 concurrent users, 60 s)

Source: `serving/loadtest_stats.csv` (sampling `data/final/val.jsonl`, checking HTTP 200 + `valid == true`).

| Metric | Value |
|---|---|
| Total requests (stats window) | 420 (424 incl. ramp-down) |
| Failures / schema errors | **0 (0.00%)** |
| Sustained throughput | **7.29 req/s** |
| Median latency (p50) | **1,000 ms** |
| Mean latency | 1,042 ms |
| p95 latency | **1,200 ms** |
| Max latency | 1,239 ms |

