from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from .data_utils import ensure_dir, prepare_graph_data
from .llm_utils import normalize_llm_payload

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - local fallback
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_DATA_DIR = PROJECT_ROOT / "graph data"
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "graph" / "outputs" / "yelpzip_final" / "prepared_data"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "graph" / "outputs" / "llm_cache" / "yelpzip_llm_abnormal_patterns.jsonl"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "graph" / "prompts" / "llm_abnormal_pattern_extraction.txt"
DEFAULT_COMPACT_PROMPT_PATH = PROJECT_ROOT / "graph" / "prompts" / "llm_abnormal_pattern_extraction_compact.txt"


class DualRateLimiter:
    def __init__(
        self,
        rpm_limit: int,
        tpm_limit: int,
        safety: float,
        expected_output_tokens: int,
    ) -> None:
        self.request_capacity = max(1, int(rpm_limit * safety))
        self.token_capacity = max(1, int(tpm_limit * safety))
        self.expected_output_tokens = max(0, expected_output_tokens)
        self.request_tokens = float(self.request_capacity)
        self.token_tokens = float(self.token_capacity)
        self.request_refill_rate = self.request_capacity / 60.0
        self.token_refill_rate = self.token_capacity / 60.0
        self.last_update = time.monotonic()
        self.lock = Lock()

    @staticmethod
    def _estimate_prompt_tokens(prompt: str) -> int:
        return max(1, int(len(prompt) / 4))

    def acquire(self, prompt: str) -> None:
        cost = self._estimate_prompt_tokens(prompt) + self.expected_output_tokens
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                if elapsed > 0:
                    self.request_tokens = min(
                        float(self.request_capacity),
                        self.request_tokens + elapsed * self.request_refill_rate,
                    )
                    self.token_tokens = min(
                        float(self.token_capacity),
                        self.token_tokens + elapsed * self.token_refill_rate,
                    )
                    self.last_update = now

                if self.request_tokens >= 1.0 and self.token_tokens >= float(cost):
                    self.request_tokens -= 1.0
                    self.token_tokens -= float(cost)
                    return

                wait_request = 0.0
                if self.request_tokens < 1.0:
                    wait_request = (1.0 - self.request_tokens) / self.request_refill_rate

                wait_tokens = 0.0
                if self.token_tokens < float(cost):
                    wait_tokens = (float(cost) - self.token_tokens) / self.token_refill_rate

                wait_time = max(wait_request, wait_tokens, 0.05)

            time.sleep(wait_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM abnormal-pattern cache JSONL for YelpZip reviews.")
    parser.add_argument("--graph_data_dir", default=str(DEFAULT_GRAPH_DATA_DIR))
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--prepared_output_dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--reviews_csv", default=None, help="Use an existing reviews_canonical.csv instead of preparing data.")
    parser.add_argument("--output_jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--prompt_path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--compact_prompt", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--senior_protocol", action="store_true", default=False)
    parser.add_argument("--balance_user_labels", action="store_true", default=False)
    parser.add_argument("--balanced_user_count", type=int, default=0)
    parser.add_argument("--min_user_reviews", type=int, default=3)
    parser.add_argument("--min_product_reviews", type=int, default=3)
    parser.add_argument("--prefer_corrected_reviews", action="store_true", default=True)
    parser.add_argument("--no_prefer_corrected_reviews", dest="prefer_corrected_reviews", action="store_false")
    parser.add_argument("--overwrite_combined_files", action="store_true", default=False)
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. Formal generation should leave this as 0.")
    parser.add_argument("--sample_size", type=int, default=0, help="Generate a representative seed subset instead of all reviews.")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--sample_strategy",
        choices=["random", "balanced"],
        default="balanced",
        help="balanced stratifies by label/rating/length bins when those columns are available.",
    )
    parser.add_argument("--start_review_node_id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_in_flight", type=int, default=0)
    parser.add_argument("--rpm_limit", type=int, default=0)
    parser.add_argument("--tpm_limit", type=int, default=0)
    parser.add_argument("--rate_limit_safety", type=float, default=1.0)
    parser.add_argument("--expected_output_tokens", type=int, default=0)
    parser.add_argument("--min_retry_after", type=float, default=0.0)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base_url", default=os.environ.get("LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    parser.add_argument("--api_key", default=os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY")))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument(
        "--enable_thinking",
        choices=["auto", "true", "false"],
        default=os.environ.get("LLM_ENABLE_THINKING", "auto"),
        help="Set SiliconFlow/Qwen/DeepSeek thinking mode. Use false for strict JSON extraction.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--no_response_format", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def maybe_apply_senior_protocol_defaults(args: argparse.Namespace) -> None:
    if not args.senior_protocol:
        return
    args.balance_user_labels = True
    if args.balanced_user_count <= 0:
        args.balanced_user_count = 6742
    args.train_ratio = 0.64
    args.val_ratio = 0.16
    args.test_ratio = 0.20


def completion_url(base_url: str) -> str:
    base_url = str(base_url).rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def load_done_ids(output_jsonl: Path) -> set[int]:
    if not output_jsonl.exists():
        return set()

    done_ids: set[int] = set()
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
                done_ids.add(int(payload["review_node_id"]))
            except Exception:
                continue
    return done_ids


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
        raise ValueError(f"LLM response does not contain a JSON object: {text[:200]}")

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

    raise ValueError(f"Unable to parse JSON object from LLM response: {text[:200]}")


def render_prompt(prompt_template: str, review_text: str, rating: Any) -> str:
    return (
        prompt_template
        .replace("{review_text}", str(review_text or ""))
        .replace("{rating}", str(rating))
    )


def call_chat_completion(
    prompt: str,
    args: argparse.Namespace,
    rate_limiter: DualRateLimiter | None = None,
) -> str:
    if rate_limiter is not None:
        rate_limiter.acquire(prompt)

    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "Return strict JSON only. Do not classify the review as fake or real.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if not args.no_response_format:
        body["response_format"] = {"type": "json_object"}
    if args.enable_thinking != "auto":
        body["enable_thinking"] = args.enable_thinking == "true"

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    request = urllib.request.Request(
        completion_url(args.base_url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, str):
        return obj.encode("utf-8", "ignore").decode("utf-8", "ignore")
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    return obj


def generate_one(
    record: dict[str, Any],
    prompt_template: str,
    args: argparse.Namespace,
    rate_limiter: DualRateLimiter | None = None,
) -> dict[str, Any]:
    review_node_id = int(record["review_node_id"])
    prompt = render_prompt(prompt_template, record["review_text"], record["rating"])
    last_error: Exception | None = None

    for attempt in range(args.retries + 1):
        try:
            content = call_chat_completion(prompt, args, rate_limiter=rate_limiter)
            payload = extract_json_object(content)
            payload["review_node_id"] = review_node_id
            return normalize_llm_payload(payload)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            sleep_seconds = args.retry_sleep * (2 ** attempt)
            if isinstance(exc, urllib.error.HTTPError):
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after is not None:
                    try:
                        sleep_seconds = max(sleep_seconds, float(retry_after))
                    except ValueError:
                        pass
                if exc.code == 429:
                    sleep_seconds = max(sleep_seconds, args.min_retry_after)
            time.sleep(sleep_seconds)

    if args.continue_on_error:
        return {
            "review_node_id": review_node_id,
            "abnormal_patterns": [],
            "sentiment": "neutral",
            "specificity_score": 0.0,
            "template_score": 0.0,
            "exaggeration_score": 0.0,
            "experience_detail_score": 0.0,
            "claim_summary": f"LLM_ERROR: {last_error}",
        }
    raise RuntimeError(f"Failed to generate LLM payload for review_node_id={review_node_id}: {last_error}") from last_error


def resolve_review_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.reviews_csv:
        return pd.read_csv(args.reviews_csv)

    prepared = prepare_graph_data(
        graph_data_dir=args.graph_data_dir,
        output_dir=args.prepared_output_dir,
        data_path=args.data_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_user_reviews=args.min_user_reviews,
        min_product_reviews=args.min_product_reviews,
        prefer_corrected_reviews=args.prefer_corrected_reviews,
        overwrite_combined_files=args.overwrite_combined_files,
        balance_user_labels=args.balance_user_labels,
        balanced_user_count=args.balanced_user_count,
    )
    return prepared.review_df


def sample_reviews(review_df: pd.DataFrame, sample_size: int, strategy: str, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or len(review_df) <= sample_size:
        return review_df
    if strategy == "random":
        return review_df.sample(n=sample_size, random_state=seed).sort_values("review_node_id").reset_index(drop=True)

    work_df = review_df.copy()
    if "review_label" in work_df.columns:
        label_part = work_df["review_label"].astype(str)
    else:
        label_part = pd.Series(["unknown"] * len(work_df), index=work_df.index)
    rating_part = pd.cut(
        pd.to_numeric(work_df.get("rating", 0), errors="coerce").fillna(0.0),
        bins=[-0.1, 2.0, 3.5, 5.1],
        labels=["low", "mid", "high"],
    ).astype(str)
    length_part = pd.qcut(
        work_df["review_text"].fillna("").astype(str).str.len().rank(method="first"),
        q=min(4, max(1, len(work_df))),
        labels=False,
        duplicates="drop",
    ).astype(str)
    work_df["_sample_bucket"] = label_part + "|" + rating_part + "|" + length_part

    sampled_parts = []
    base_quota = max(1, sample_size // max(work_df["_sample_bucket"].nunique(), 1))
    for _, bucket_df in work_df.groupby("_sample_bucket"):
        sampled_parts.append(bucket_df.sample(n=min(base_quota, len(bucket_df)), random_state=seed))
    sampled_df = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["review_node_id"])

    if len(sampled_df) < sample_size:
        remaining = work_df[~work_df["review_node_id"].isin(sampled_df["review_node_id"])]
        if not remaining.empty:
            extra = remaining.sample(n=min(sample_size - len(sampled_df), len(remaining)), random_state=seed + 1)
            sampled_df = pd.concat([sampled_df, extra], ignore_index=True)
    elif len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size, random_state=seed + 2)

    return sampled_df.drop(columns=["_sample_bucket"], errors="ignore").sort_values("review_node_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    maybe_apply_senior_protocol_defaults(args)
    output_jsonl = Path(args.output_jsonl)
    ensure_dir(output_jsonl.parent)
    if args.overwrite_cache and output_jsonl.exists():
        output_jsonl.unlink()

    prompt_path = DEFAULT_COMPACT_PROMPT_PATH if args.compact_prompt else Path(args.prompt_path)
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    review_df = resolve_review_dataframe(args)
    required_columns = {"review_node_id", "review_text", "rating"}
    missing_columns = required_columns - set(review_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in review dataframe: {sorted(missing_columns)}")

    done_ids = load_done_ids(output_jsonl)
    target_df = review_df[review_df["review_node_id"].astype(int) >= int(args.start_review_node_id)].copy()
    target_df = target_df[~target_df["review_node_id"].astype(int).isin(done_ids)].copy()
    target_df = target_df.sort_values("review_node_id").reset_index(drop=True)
    target_df = sample_reviews(
        review_df=target_df,
        sample_size=args.sample_size,
        strategy=args.sample_strategy,
        seed=args.sample_seed,
    )
    if args.limit > 0:
        target_df = target_df.head(args.limit)

    records = target_df[["review_node_id", "review_text", "rating"]].to_dict("records")
    print(f"Prepared reviews: {len(review_df)}")
    print(f"Existing cache rows: {len(done_ids)}")
    print(f"Reviews to generate now: {len(records)}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    if args.rpm_limit > 0 or args.tpm_limit > 0:
        print(
            "Client-side rate limit: "
            f"rpm={int(args.rpm_limit * args.rate_limit_safety)} "
            f"tpm={int(args.tpm_limit * args.rate_limit_safety)} "
            f"max_in_flight={max(args.max_in_flight, args.workers, 1)}"
        )

    if args.dry_run:
        if records:
            print(render_prompt(prompt_template, records[0]["review_text"], records[0]["rating"])[:2000])
        return

    if not args.api_key and "localhost" not in str(args.base_url) and "127.0.0.1" not in str(args.base_url):
        raise ValueError("No API key found. Set OPENAI_API_KEY/LLM_API_KEY, or use a local --base_url endpoint.")

    rate_limiter = None
    if args.rpm_limit > 0 or args.tpm_limit > 0:
        rate_limiter = DualRateLimiter(
            rpm_limit=max(args.rpm_limit, 1),
            tpm_limit=max(args.tpm_limit, 1),
            safety=max(args.rate_limit_safety, 0.01),
            expected_output_tokens=max(args.expected_output_tokens, 0),
        )

    with output_jsonl.open("a", encoding="utf-8") as handle:
        if args.workers <= 1:
            iterator = tqdm(records, total=len(records), desc="Generating LLM cache")
            for record in iterator:
                payload = generate_one(record, prompt_template, args, rate_limiter=rate_limiter)
                handle.write(json.dumps(sanitize_for_json(payload), ensure_ascii=False) + "\n")
                handle.flush()
        else:
            max_in_flight = max(args.max_in_flight, args.workers, 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                pending: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
                record_iter = iter(records)

                def submit_next() -> bool:
                    try:
                        record = next(record_iter)
                    except StopIteration:
                        return False
                    future = executor.submit(generate_one, record, prompt_template, args, rate_limiter)
                    pending[future] = int(record["review_node_id"])
                    return True

                for _ in range(min(max_in_flight, len(records))):
                    if not submit_next():
                        break

                iterator = tqdm(total=len(records), desc="Generating LLM cache")
                while pending:
                    done, _ = concurrent.futures.wait(
                        set(pending.keys()),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        payload = future.result()
                        handle.write(json.dumps(sanitize_for_json(payload), ensure_ascii=False) + "\n")
                        handle.flush()
                        pending.pop(future, None)
                        iterator.update(1)
                        submit_next()

    print("LLM cache generation finished.")


if __name__ == "__main__":
    main()
