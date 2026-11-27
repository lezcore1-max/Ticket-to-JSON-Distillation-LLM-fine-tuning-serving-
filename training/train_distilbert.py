"""Train and evaluate a multi-head DistilBERT baseline for the categorical ticket fields.

Predicts intent (27 classes -> deterministic category via taxonomy), urgency (3), and
sentiment (3); always emits product=null and order_id=null so the output matches the
6-field schema in labeling/schema.json for apples-to-apples scoring in eval/eval.py.

  python -m training.train_distilbert
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import tempfile
import time
from pathlib import Path


def _ensure_writable_cache() -> None:
    d = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if d:
        try:
            os.makedirs(d, exist_ok=True)
            return
        except OSError:
            pass
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "torchinductor")


_ensure_writable_cache()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

from eval.eval import headline, score_rows  # noqa: E402
from labeling.splits import ROOT, load_jsonl  # noqa: E402
from labeling.validate import FIELDS, INTENT_TO_CATEGORY, SCHEMA  # noqa: E402
from training.dataset import load_final_split  # noqa: E402

INTENTS = SCHEMA["properties"]["intent"]["enum"]
URGENCIES = SCHEMA["properties"]["urgency"]["enum"]
SENTIMENTS = SCHEMA["properties"]["sentiment"]["enum"]

INTENT2ID = {k: i for i, k in enumerate(INTENTS)}
URG2ID = {k: i for i, k in enumerate(URGENCIES)}
SENT2ID = {k: i for i, k in enumerate(SENTIMENTS)}


class TicketClsDataset(Dataset):
    def __init__(self, rows: list[dict], tok, max_length: int):
        self.rows = rows
        enc = tok(
            [r["text"] for r in rows],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.has_labels = "label" in rows[0]
        if self.has_labels:
            self.y_intent = [INTENT2ID[r["label"]["intent"]] for r in rows]
            self.y_urg = [URG2ID[r["label"]["urgency"]] for r in rows]
            self.y_sent = [SENT2ID[r["label"]["sentiment"]] for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.has_labels:
            item["y_intent"] = self.y_intent[idx]
            item["y_urg"] = self.y_urg[idx]
            item["y_sent"] = self.y_sent[idx]
        return item


def make_collate(pad_id: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        ids, mask = [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            ids.append(x["input_ids"] + [pad_id] * pad)
            mask.append(x["attention_mask"] + [0] * pad)
        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }
        if "y_intent" in batch[0]:
            out["y_intent"] = torch.tensor([x["y_intent"] for x in batch], dtype=torch.long)
            out["y_urg"] = torch.tensor([x["y_urg"] for x in batch], dtype=torch.long)
            out["y_sent"] = torch.tensor([x["y_sent"] for x in batch], dtype=torch.long)
        return out

    return collate


class MultiHeadDistilBERT(nn.Module):
    def __init__(self, model_name: str, local_files_only: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head_intent = nn.Linear(h, len(INTENTS))
        self.head_urg = nn.Linear(h, len(URGENCIES))
        self.head_sent = nn.Linear(h, len(SENTIMENTS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pooled = self.drop(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0])
        return self.head_intent(pooled), self.head_urg(pooled), self.head_sent(pooled)


def logits_to_json(intent_idx: int, urg_idx: int, sent_idx: int) -> str:
    intent = INTENTS[intent_idx]
    obj = {
        "intent": intent,
        "category": INTENT_TO_CATEGORY[intent],
        "urgency": URGENCIES[urg_idx],
        "product": None,
        "order_id": None,
        "sentiment": SENTIMENTS[sent_idx],
    }
    return json.dumps({k: obj[k] for k in FIELDS}, ensure_ascii=False)


def eval_val(model: MultiHeadDistilBERT, loader: DataLoader, val_rows: list[dict], dtype: torch.dtype) -> dict:
    model.eval()
    preds_raw = []
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        for batch in loader:
            li, lu, ls = model(batch["input_ids"].to("cuda"), batch["attention_mask"].to("cuda"))
            pi = li.argmax(dim=-1).tolist()
            pu = lu.argmax(dim=-1).tolist()
            ps = ls.argmax(dim=-1).tolist()
            for ii, uu, ss in zip(pi, pu, ps):
                preds_raw.append(logits_to_json(ii, uu, ss))
    preds = {r["id"]: {"id": r["id"], "raw_output": raw} for r, raw in zip(val_rows, preds_raw)}
    scored = score_rows(val_rows, preds)
    h = headline(scored)
    cat_exact = sum(
        r["correct"]["intent"] and r["correct"]["urgency"] and r["correct"]["sentiment"]
        for r in scored
    ) / len(scored)
    return {
        "val_exact_match": round(h["exact_match"], 4),
        "val_categorical_exact": round(cat_exact, 4),
        "val_acc_intent": round(h["acc_intent"], 4),
        "val_acc_urgency": round(h["acc_urgency"], 4),
        "val_acc_sentiment": round(h["acc_sentiment"], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "training" / "checkpoints" / "distilbert")
    ap.add_argument("--pred-out", type=Path, default=ROOT / "eval" / "predictions" / "distilbert.jsonl")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: install the CUDA build of torch")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_final_split("train")
    val_rows = load_final_split("val")

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    train_ds = TicketClsDataset(train_rows, tok, args.max_length)
    val_ds = TicketClsDataset(val_rows, tok, args.max_length)
    collate = make_collate(tok.pad_token_id or 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate)

    model = MultiHeadDistilBERT(args.model, local_files_only=args.local_files_only).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
    ce = nn.CrossEntropyLoss()

    best_score = (-1.0, -1.0)
    best_path = args.out_dir / "best.pt"
    history = []
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                li, lu, ls = model(batch["input_ids"].to("cuda"), batch["attention_mask"].to("cuda"))
                loss = (
                    ce(li, batch["y_intent"].to("cuda"))
                    + ce(lu, batch["y_urg"].to("cuda"))
                    + ce(ls, batch["y_sent"].to("cuda"))
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())

        metrics = eval_val(model, val_loader, val_rows, amp_dtype)
        metrics["epoch"] = ep
        metrics["train_loss"] = round(tot_loss / len(train_loader), 4)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

        key = (metrics["val_categorical_exact"], metrics["val_exact_match"])
        if key > best_score:
            best_score = key
            torch.save(model.state_dict(), best_path)

    train_secs = round(time.perf_counter() - t0, 1)
    (args.out_dir / "train_meta.json").write_text(
        json.dumps({"model": args.model, "epochs": args.epochs, "lr": args.lr, "train_seconds": train_secs,
                    "best_val": max(history, key=lambda x: (x["val_categorical_exact"], x["val_exact_match"])),
                    "history": history}, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )

    # Test inference (batch_size=1, warm-up, GPU synchronized latency)
    model.load_state_dict(torch.load(best_path, map_location="cuda", weights_only=True))
    model.to(amp_dtype)
    model.eval()
    test_rows = [{"id": r["id"], "text": r["text"]} for r in load_jsonl(ROOT / "data" / "final" / "test.jsonl")]

    def predict_one(text: str):
        enc = tok(text, truncation=True, max_length=args.max_length, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        ts = time.perf_counter()
        with torch.inference_mode():
            li, lu, ls = model(enc["input_ids"], enc["attention_mask"])
        torch.cuda.synchronize()
        dt = time.perf_counter() - ts
        raw = logits_to_json(int(li.argmax(-1).item()), int(lu.argmax(-1).item()), int(ls.argmax(-1).item()))
        return raw, dt, int(enc["input_ids"].shape[1])

    predict_one(test_rows[0]["text"])  # warm-up; not recorded
    torch.cuda.reset_peak_memory_stats()

    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_out, "w", encoding="utf-8", newline="\n") as f:
        for r in test_rows:
            raw, dt, n_in = predict_one(r["text"])
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "raw_output": raw,
                        "latency_s": round(dt, 4),
                        "tokens": {"prompt": n_in, "output": 0},
                        "model": args.model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    meta = {
        "model": args.model,
        "dtype": str(amp_dtype).replace("torch.", ""),
        "gpu": torch.cuda.get_device_name(0),
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "batch_size": 1,
        "epochs": args.epochs,
        "train_seconds": train_secs,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    args.pred_out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("-> wrote", args.pred_out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

