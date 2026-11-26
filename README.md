# Ticket-to-JSON Distillation

Distilling a large LLM ("teacher") into a small fine-tuned open model that turns raw
customer-support messages into structured JSON, then serving it in production.

```text
Input:  "my order #4521 hasn't arrived, it's been 10 days, very frustrated"
Output: {"intent": "delivery_period", "category": "delivery", "urgency": "high",
         "product": null, "order_id": "4521", "sentiment": "negative"}
```

**Pipeline:** Bitext messages → teacher LLM labels (Gemini, schema-validated) → QLoRA fine-tune of
Qwen2.5-1.5B-Instruct (Unsloth) → vLLM + FastAPI (Docker, load-tested) → GGUF + Gradio demo.

> 🚧 Work in progress: a 7-day build. Day 1 (data + labeling pipeline) is done.

## Progress

- [x] **Day 1:** dataset prep, output schema, teacher prompt, batched labeling pipeline, pilot report
- [ ] Day 2: label all messages, hand-check the 200-message test set
- [ ] Day 3: evaluation script and baselines (teacher, prompted small model)
- [ ] Day 4: QLoRA fine-tuning (2 runs) and DistilBERT baseline
- [ ] Day 5: vLLM + FastAPI + Docker serving, Locust load test
- [ ] Day 6: GGUF export, Gradio demo on HF Spaces, cost per 1K requests
- [ ] Day 7: final results, architecture diagram, write-up

## Results

All numbers will be produced by `eval/eval.py`; none are filled in by hand.

| Model | Field acc | Exact match | Valid JSON % | p95 latency | Cost / 1K req |
|---|---|---|---|---|---|
| Teacher (API) | – | – | – | – | – |
| Small model, prompted | – | – | – | – | – |
| Small model, fine-tuned (run 1) | – | – | – | – | – |
| Small model, fine-tuned (run 2) | – | – | – | – | – |
| DistilBERT (categorical fields) | – | – | – | – | – |

## Output schema

Defined in [`labeling/schema.json`](labeling/schema.json); the same validator
([`labeling/validate.py`](labeling/validate.py)) is used for labeling, evaluation and serving.

| Field | Type | Values |
|---|---|---|
| `intent` | enum | 27 intents (Bitext taxonomy, see [`labeling/taxonomy.json`](labeling/taxonomy.json)) |
| `category` | enum | 11 categories; must match the intent |
| `urgency` | enum | `low`, `medium`, `high` |
| `product` | string or null | named item / plan / account tier, e.g. `"gold account"` |
| `order_id` | string or null | as written, without a leading `#` |
| `sentiment` | enum | `negative`, `neutral`, `positive` |

## Data

Source: [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(synthetic, with deliberate typos and colloquial phrasing). Prepared by
[`labeling/prepare_data.py`](labeling/prepare_data.py) (stats in
[`data/prepared/stats.json`](data/prepared/stats.json)):

1. 26,872 raw messages → **24,025** after removing duplicate templates (placeholders treated as equal).
2. Stratified sample of **3,500** across 27 intents.
3. Split **3,000 train / 300 val / 200 test** *before* labeling. A shared guard
   ([`labeling/splits.py`](labeling/splits.py)) fails if any test id, text or template appears in train/val.
4. `{{placeholders}}` filled with realistic values. Injected order numbers (#940252, ORD-40917, …)
   and account tiers serve as known ground truth for checking the teacher.

## Labeling

[`labeling/label.py`](labeling/label.py) sends 20 numbered messages per request to the teacher
(fits the Gemini free tier: about 175 requests for 3,500 messages). Gemini's structured-output
mode is used with the schema, and every label is checked on its own. Missing or invalid labels
are resent in smaller batches. Runs are resumable and cap their request count.
[`labeling/pilot_report.py`](labeling/pilot_report.py) compares teacher labels against Bitext
labels and injected values, and compares two teachers against each other.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env              # add GEMINI_API_KEY

python -m labeling.prepare_data                                   # build splits
python -m labeling.label --list-models                            # see available model ids
python -m labeling.label --split train --n 50 --stratify --out data/labeled/pilot.jsonl
python -m labeling.pilot_report --labels data/labeled/pilot.jsonl --out labeling/pilot_report.md
```

## Repo layout

```text
data/        prepared splits (train/val/test JSONL), test ids; labeled data from Day 2
labeling/    schema, taxonomy, teacher prompt, data prep, labeler, pilot report
training/    QLoRA fine-tuning (Day 4)
eval/        evaluation script (Day 3)
serving/     FastAPI + vLLM + Docker + Locust (Day 5)
demo/        Gradio app for HF Spaces (Day 6)
```

## License

Data in `data/` is derived from the Bitext dataset and shared under its license,
[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
