from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_utils import prepare_graph_data
from .generate_llm_cache import DEFAULT_GRAPH_DATA_DIR, DEFAULT_OUTPUT_JSONL, DEFAULT_PREPARED_DIR
from .llm_utils import load_llm_cache, normalize_llm_payload

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - local fallback
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED_JSONL = PROJECT_ROOT / "graph" / "outputs" / "llm_cache" / "yelpzip_llm_seed.jsonl"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "graph" / "outputs" / "local_annotator" / "t5_abnormal_extractor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local seq2seq annotator from LLM seed labels and fill the full cache.")
    parser.add_argument("--graph_data_dir", default=str(DEFAULT_GRAPH_DATA_DIR))
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--prepared_output_dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--reviews_csv", default=None)
    parser.add_argument("--seed_jsonl", default=str(DEFAULT_SEED_JSONL))
    parser.add_argument("--output_jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--model_name_or_path", default="google/flan-t5-base")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--overwrite_output", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_source_length", type=int, default=768)
    parser.add_argument("--max_target_length", type=int, default=384)
    parser.add_argument("--generation_max_new_tokens", type=int, default=384)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--limit_generate", type=int, default=0)
    parser.add_argument("--continue_on_error", action="store_true", default=True)
    parser.add_argument("--min_seed_rows", type=int, default=200)
    return parser.parse_args()


def import_training_modules():
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    except Exception as exc:  # pragma: no cover - depends on server runtime
        raise ImportError("Local annotator training/generation requires torch and transformers on the server.") from exc
    return torch, Dataset, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments


def compact_payload_for_target(payload: dict[str, Any]) -> dict[str, Any]:
    payload = normalize_llm_payload(payload)
    return {
        "abnormal_patterns": payload["abnormal_patterns"],
        "sentiment": payload["sentiment"],
        "specificity_score": payload["specificity_score"],
        "template_score": payload["template_score"],
        "exaggeration_score": payload["exaggeration_score"],
        "experience_detail_score": payload["experience_detail_score"],
        "claim_summary": payload["claim_summary"],
    }


def make_source(review_text: str, rating: Any) -> str:
    return (
        "Extract abnormal writing patterns from one Yelp review. "
        "Do not classify fake or real. Return strict JSON with keys: "
        "abnormal_patterns, sentiment, specificity_score, template_score, "
        "exaggeration_score, experience_detail_score, claim_summary. "
        "Every evidence_span must be copied exactly from the review. "
        f"Rating: {rating}\nReview: {review_text}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in local annotator output: {text[:200]}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:index + 1])
    raise ValueError(f"Unable to parse JSON object from local annotator output: {text[:200]}")


def validate_spans(payload: dict[str, Any], review_text: str) -> dict[str, Any]:
    review_text = str(review_text or "")
    patterns = []
    for pattern in payload.get("abnormal_patterns", []) or []:
        span = str(pattern.get("evidence_span", "") or "").strip()
        pattern_type = str(pattern.get("pattern_type", "none") or "none")
        if pattern_type == "none" or not span:
            continue
        if span.lower() not in review_text.lower():
            continue
        patterns.append(pattern)
    payload["abnormal_patterns"] = patterns or [
        {
            "pattern_type": "none",
            "description": "",
            "evidence_span": "",
            "confidence": 0.0,
        }
    ]
    return payload


def resolve_reviews(args: argparse.Namespace) -> pd.DataFrame:
    if args.reviews_csv:
        return pd.read_csv(args.reviews_csv)
    review_csv = Path(args.prepared_output_dir) / "reviews_canonical.csv"
    if review_csv.exists():
        return pd.read_csv(review_csv)
    prepared = prepare_graph_data(
        graph_data_dir=args.graph_data_dir,
        output_dir=args.prepared_output_dir,
        data_path=args.data_path,
        seed=args.seed,
    )
    return prepared.review_df


def build_training_frame(review_df: pd.DataFrame, seed_jsonl: str | Path) -> pd.DataFrame:
    cache = load_llm_cache(seed_jsonl)
    seed_df = review_df[review_df["review_node_id"].astype(int).isin(cache.keys())].copy()
    rows = []
    for row in seed_df.itertuples(index=False):
        payload = cache[int(row.review_node_id)]
        rows.append(
            {
                "source": make_source(row.review_text, row.rating),
                "target": json.dumps(compact_payload_for_target(payload), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def train_model(args: argparse.Namespace, review_df: pd.DataFrame) -> None:
    torch, Dataset, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments = import_training_modules()
    train_df = build_training_frame(review_df, args.seed_jsonl)
    if len(train_df) < args.min_seed_rows:
        raise ValueError(f"Seed JSONL has only {len(train_df)} rows; expected at least {args.min_seed_rows}.")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(train_df))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.9))
    train_indices = indices[:split]
    eval_indices = indices[split:] if split < len(indices) else indices[:1]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    class Seq2SeqDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, selected_indices: np.ndarray) -> None:
            self.frame = frame.iloc[selected_indices].reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = self.frame.iloc[index]
            encoded = tokenizer(
                item["source"],
                max_length=args.max_source_length,
                truncation=True,
            )
            labels = tokenizer(
                text_target=item["target"],
                max_length=args.max_target_length,
                truncation=True,
            )
            encoded["labels"] = labels["input_ids"]
            return encoded

    training_args = TrainingArguments(
        output_dir=str(args.model_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        fp16=bool(torch.cuda.is_available()),
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Seq2SeqDataset(train_df, train_indices),
        eval_dataset=Seq2SeqDataset(train_df, eval_indices),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.model_dir))
    tokenizer.save_pretrained(str(args.model_dir))


def load_done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
                done.add(int(payload["review_node_id"]))
            except Exception:
                continue
    return done


def write_seed_rows(seed_jsonl: Path, output_jsonl: Path) -> None:
    if not seed_jsonl.exists():
        return
    done = load_done_ids(output_jsonl)
    with seed_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open("a", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            payload = json.loads(line)
            if int(payload["review_node_id"]) in done:
                continue
            dst.write(json.dumps(normalize_llm_payload(payload), ensure_ascii=False) + "\n")
            done.add(int(payload["review_node_id"]))


def generate_cache(args: argparse.Namespace, review_df: pd.DataFrame) -> None:
    torch, _, AutoModelForSeq2SeqLM, AutoTokenizer, _, _, _ = import_training_modules()
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_output and output_jsonl.exists():
        output_jsonl.unlink()
    write_seed_rows(Path(args.seed_jsonl), output_jsonl)

    done_ids = load_done_ids(output_jsonl)
    target_df = review_df[~review_df["review_node_id"].astype(int).isin(done_ids)].copy()
    target_df = target_df.sort_values("review_node_id").reset_index(drop=True)
    if args.limit_generate > 0:
        target_df = target_df.head(args.limit_generate)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with output_jsonl.open("a", encoding="utf-8") as handle:
        iterator = tqdm(target_df.itertuples(index=False), total=len(target_df), desc="Local annotator generating")
        for row in iterator:
            source = make_source(row.review_text, row.rating)
            encoded = tokenizer(
                source,
                return_tensors="pt",
                max_length=args.max_source_length,
                truncation=True,
            ).to(device)
            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        **encoded,
                        max_new_tokens=args.generation_max_new_tokens,
                        num_beams=args.num_beams,
                    )
                text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                payload = extract_json_object(text)
                payload["review_node_id"] = int(row.review_node_id)
                payload = normalize_llm_payload(payload)
                payload = validate_spans(payload, row.review_text)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                payload = normalize_llm_payload(
                    {
                        "review_node_id": int(row.review_node_id),
                        "abnormal_patterns": [],
                        "sentiment": "neutral",
                        "specificity_score": 0.0,
                        "template_score": 0.0,
                        "exaggeration_score": 0.0,
                        "experience_detail_score": 0.0,
                        "claim_summary": f"LOCAL_ANNOTATOR_ERROR: {exc}",
                    }
                )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()


def main() -> None:
    args = parse_args()
    if not args.train and not args.generate:
        args.train = True
        args.generate = True
    review_df = resolve_reviews(args)
    if args.train:
        train_model(args, review_df)
    if args.generate:
        generate_cache(args, review_df)


if __name__ == "__main__":
    main()
