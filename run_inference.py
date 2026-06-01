#!/usr/bin/env python3
"""
Final reproducible inference pipeline for the CSE 151B Kaggle Math Reasoning Competition.

Core requirement satisfied:
    from run_inference import run_inference
    run_inference()

This script uses only Qwen/Qwen3-4B-Thinking-2507 or a LoRA/QLoRA adapter on top of that
base model. It performs the complete pipeline end-to-end:
    1. load data
    2. load model / optional adapter
    3. build prompts
    4. generate one or more samples
    5. apply answer extraction + self-consistency voting
    6. write Kaggle submission CSV
    7. write debug JSONL checkpoint for reproducibility / resume

Assumptions:
    - common_utils.py exists in the repo and provides the helpers imported below.
    - The input file is JSONL with at least {"id": ..., "question": ...}; MCQ examples may also
      include an "options" field.
    - Kaggle expects a CSV with columns: id, answer. If your competition uses a different answer
      column name, change DEFAULT_ANSWER_COLUMN below or pass --answer_column.
"""

from __future__ import annotations

# NOTE: If you want to force a specific GPU, it is safest to set it before running Python:
#   CUDA_VISIBLE_DEVICES=0 python run_inference.py ...
# This file intentionally does not set CUDA_VISIBLE_DEVICES after importing torch because that is
# often too late to reliably affect device discovery.

import argparse
import collections
import csv
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from common_utils import (
    MODEL_ID,
    extract_boxed,
    extract_letter,
    format_options,
    generate_batch,
    load_judger,
    load_jsonl,
    load_model_and_tokenizer,
    make_chat_prompt,
    score_response,
    summarize_results,
)

# Competition-required base model. This script refuses to run if common_utils.MODEL_ID points
# somewhere else, unless allow_model_id_override=True is explicitly passed to run_inference().
REQUIRED_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

DEFAULT_DATA_PATH = "data/public.jsonl"
DEFAULT_OUTPUT_CSV = "submission.csv"
DEFAULT_DEBUG_JSONL = "results/debug_generations.jsonl"
DEFAULT_ID_COLUMN = "id"
DEFAULT_ANSWER_COLUMN = "response"


PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "structured_cot": {
        "sys_math": (
            "You are an expert mathematician.\n"
            "Solve the problem step by step using concise reasoning.\n"
            "Avoid unnecessary repetition or self-correction.\n"
            "Do not restate the problem.\n"
            "Once you obtain the solution, immediately give the final answer.\n"
            "Place the final answer inside \\boxed{} and do not write anything after the boxed answer."
        ),
        "sys_mcq": (
            "You are an expert mathematician.\n"
            "Solve the problem using concise reasoning.\n"
            "Avoid unnecessary repetition.\n"
            "Determine the correct option efficiently.\n"
            "End with ONLY the final option letter inside \\boxed{}.\n"
            "Do not write anything after the boxed answer."
        ),
    },
    "verification": {
        "sys_math": (
        "You are an expert mathematician.\n"
        "Solve the problem with concise reasoning.\n"
        "Briefly verify the result once, without repeating the full solution.\n"
        "Avoid unnecessary self-correction or restating the problem.\n"
        "Once verified, immediately place the final answer inside \\boxed{}.\n"
        "Do not write anything after the boxed answer."
    ),
    "sys_mcq": (
        "You are an expert mathematician.\n"
        "Solve the problem with concise reasoning.\n"
        "Briefly verify which option matches the result.\n"
        "Avoid lengthy elimination unless necessary.\n"
        "End with ONLY the final option letter inside \\boxed{}.\n"
        "Do not write anything after the boxed answer."
    ),
    },
    "vanilla": {
        "sys_math": "You are an expert mathematician. Solve the problem and put the final answer in \\boxed{}.",
        "sys_mcq": "You are an expert mathematician. Choose the correct option and put only the letter in \\boxed{}.",
    },
}


@dataclass(frozen=True)
class InferenceConfig:
    data_path: str = DEFAULT_DATA_PATH
    output_csv: str = DEFAULT_OUTPUT_CSV
    debug_jsonl: str = DEFAULT_DEBUG_JSONL
    adapter_path: Optional[str] = None  # May be local path or HuggingFace Hub repo ID.
    prompt_strategy: str = "structured_cot"
    num_samples: int = 1
    inner_batch: int = 4
    max_tokens: int = 1024
    load_in_4bit: bool = False
    seed: int = 42
    id_column: str = DEFAULT_ID_COLUMN
    answer_column: str = DEFAULT_ANSWER_COLUMN
    subset_ids: Optional[str] = None
    overwrite: bool = False
    allow_model_id_override: bool = False


def set_reproducibility(seed: int) -> None:
    """Set best-effort deterministic seeds. Generation can still vary on GPU kernels/sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_competition_model(allow_model_id_override: bool = False) -> None:
    """Guardrail against accidentally using a non-competition model."""
    if MODEL_ID != REQUIRED_MODEL_ID and not allow_model_id_override:
        raise ValueError(
            f"Competition requires {REQUIRED_MODEL_ID}, but common_utils.MODEL_ID is {MODEL_ID!r}. "
            "Fix MODEL_ID or call run_inference(..., allow_model_id_override=True) only if your course staff "
            "explicitly permits this."
        )


def build_prompt(item: dict[str, Any], template_name: str) -> tuple[str, str]:
    """Create system/user prompt pair for free-response or multiple-choice examples."""
    cfg = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["structured_cot"])
    question = str(item.get("question", "")).strip()
    if not question:
        raise ValueError(f"Missing question text for item with id={item.get('id')!r}")

    if item.get("options"):
        user = f"{question}\n\nOptions:\n{format_options(item['options'])}"
        return cfg["sys_mcq"], user
    return cfg["sys_math"], question


def clean_free_response_answer(answer: str) -> str:
    """Normalize free-response strings without using any external tools."""
    answer = answer.strip()
    answer = re.sub(r"^\\boxed\{(.+)\}$", r"\1", answer)
    answer = answer.strip().strip("$ ")
    answer = re.sub(r"\s+", "", answer)
    # Remove common trailing punctuation produced by models.
    answer = answer.rstrip(".,;")
    return answer


def normalize_vote(item: dict[str, Any], response: str) -> str:
    """Extract the answer used for CSV submission and self-consistency voting."""
    response = response or ""
    if item.get("options"):
        letter = extract_letter(response)
        return letter.upper() if letter else ""

    boxed = extract_boxed(response)
    if boxed:
        return clean_free_response_answer(boxed)

    # Fallback: use the last short chunk if the model fails to box. This is not ideal, but prevents
    # blank submissions and makes failures inspectable in debug_jsonl.
    tail = response.strip()[-200:]
    tail = re.sub(r".*(?:answer is|final answer is|therefore)\s*", "", tail, flags=re.IGNORECASE | re.DOTALL)
    return clean_free_response_answer(tail)


def majority_vote(item: dict[str, Any], responses: list[str]) -> tuple[str, str, dict[str, int], list[str]]:
    """Vote over normalized answers. Tie-break by first occurrence, not longest reasoning."""
    votes = [normalize_vote(item, r) for r in responses]
    counts = collections.Counter(v for v in votes if v)

    if not counts:
        return "", responses[0] if responses else "", {}, votes

    # Deterministic tie-break: higher count, then earliest occurrence in sample order.
    first_index = {vote: votes.index(vote) for vote in counts}
    best_vote = max(counts.keys(), key=lambda v: (counts[v], -first_index[v]))
    best_response = responses[first_index[best_vote]]
    return best_vote, best_response, dict(counts), votes


def load_engine(adapter_path: Optional[str], load_in_4bit: bool):
    """Load required base model, optionally with a LoRA/QLoRA adapter."""
    if adapter_path:
        print(f"Loading base model [{MODEL_ID}] + LoRA adapter [{adapter_path}]...")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Adapter repos usually include tokenizer files if saved correctly; if not, fall back to base.
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        return tokenizer, model

    print(f"Loading required base model [{MODEL_ID}]...")
    tokenizer, model = load_model_and_tokenizer(MODEL_ID, load_in_4bit=load_in_4bit)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer, model


def load_data(data_path: str, subset_ids: Optional[str] = None) -> list[dict[str, Any]]:
    """Load JSONL data and optionally filter to a set of IDs."""
    data = load_jsonl(data_path)
    if subset_ids:
        with open(subset_ids, "r", encoding="utf-8") as f:
            raw = json.load(f)
        subset = {x["id"] if isinstance(x, dict) else x for x in raw}
        data = [d for d in data if d.get("id") in subset]
    return data


def get_sampling_params(num_samples: int) -> dict[str, Any]:
    if num_samples > 1:
        return {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
        }
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "do_sample": False,
    }


def write_submission_csv(rows: list[dict[str, Any]], output_csv: str, id_column: str, answer_column: str) -> None:
    """Write Kaggle-style submission CSV with stable column order."""
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[id_column, answer_column])
        writer.writeheader()
        for row in rows:
            writer.writerow({id_column: row[id_column], answer_column: row[answer_column]})


def load_completed_records(debug_jsonl: str) -> dict[Any, dict[str, Any]]:
    """
    Load successful records from an existing debug JSONL file.

    This makes results/debug_generations.jsonl act as a checkpoint. A record is considered
    completed only if it has no error and has a nonblank prediction. Failed/error records are
    intentionally not skipped, so reruns can retry them.
    """
    path = Path(debug_jsonl)
    completed: dict[Any, dict[str, Any]] = {}
    if not path.exists():
        return completed

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            pred = str(rec.get("prediction", "")).strip()
            if rec.get("error") is None and pred:
                completed[rec.get("id")] = rec

    return completed


def rebuild_submission_from_debug(
    data: list[dict[str, Any]],
    debug_jsonl: str,
    output_csv: str,
    id_column: str,
    answer_column: str,
) -> tuple[int, int]:
    """
    Rebuild the Kaggle CSV from debug_jsonl in the same order as the input data.

    Returns:
        (n_written, n_missing)
    """
    completed = load_completed_records(debug_jsonl)
    rows: list[dict[str, Any]] = []
    missing = 0

    for item in data:
        item_id = item.get("id")
        rec = completed.get(item_id)
        if rec is None:
            missing += 1
            continue
        rows.append({id_column: item_id, answer_column: rec["prediction"]})

    write_submission_csv(rows, output_csv, id_column, answer_column)
    return len(rows), missing


def run_inference(
    data_path: str = DEFAULT_DATA_PATH,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    debug_jsonl: str = DEFAULT_DEBUG_JSONL,
    adapter_path: Optional[str] = None,
    prompt_strategy: str = "structured_cot",
    num_samples: int = 1,
    inner_batch: int = 4,
    max_tokens: int = 1024,
    load_in_4bit: bool = False,
    seed: int = 42,
    id_column: str = DEFAULT_ID_COLUMN,
    answer_column: str = DEFAULT_ANSWER_COLUMN,
    subset_ids: Optional[str] = None,
    overwrite: bool = False,
    allow_model_id_override: bool = False,
) -> str:
    """
    Complete final pipeline. Calling this function produces the final Kaggle submission CSV.

    Returns:
        Path to the written CSV file as a string.
    """
    cfg = InferenceConfig(
        data_path=data_path,
        output_csv=output_csv,
        debug_jsonl=debug_jsonl,
        adapter_path=adapter_path,
        prompt_strategy=prompt_strategy,
        num_samples=num_samples,
        inner_batch=inner_batch,
        max_tokens=max_tokens,
        load_in_4bit=load_in_4bit,
        seed=seed,
        id_column=id_column,
        answer_column=answer_column,
        subset_ids=subset_ids,
        overwrite=overwrite,
        allow_model_id_override=allow_model_id_override,
    )

    validate_competition_model(allow_model_id_override=cfg.allow_model_id_override)
    if cfg.prompt_strategy not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt_strategy={cfg.prompt_strategy!r}. Choose from {list(PROMPT_TEMPLATES)}")
    if cfg.num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if cfg.inner_batch < 1:
        raise ValueError("inner_batch must be >= 1")

    set_reproducibility(cfg.seed)

    data = load_data(cfg.data_path, cfg.subset_ids)
    if not data:
        raise ValueError(f"No examples loaded from {cfg.data_path!r}")

    Path(cfg.debug_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_csv).parent.mkdir(parents=True, exist_ok=True) if Path(cfg.output_csv).parent != Path(".") else None

    if Path(cfg.debug_jsonl).exists() and cfg.overwrite:
        Path(cfg.debug_jsonl).unlink()
    if Path(cfg.output_csv).exists() and cfg.overwrite:
        Path(cfg.output_csv).unlink()

    print("Final inference configuration:")
    print(json.dumps(asdict(cfg), indent=2))

    tokenizer, llm = load_engine(cfg.adapter_path, cfg.load_in_4bit)
    judger = load_judger() if any("answer" in item for item in data) else None
    sampling_params = get_sampling_params(cfg.num_samples)

    completed_records = {} if cfg.overwrite else load_completed_records(cfg.debug_jsonl)
    completed_ids = set(completed_records.keys())

    generation_error_count = 0
    blank_prediction_count = 0
    skipped_count = len(completed_ids)

    remaining_count = sum(1 for item in data if item.get("id") not in completed_ids)
    print(f"Found {skipped_count} completed examples in checkpoint: {cfg.debug_jsonl}")
    print(f"Running inference on {remaining_count} remaining examples out of {len(data)} total...")

    debug_mode = "w" if cfg.overwrite else "a"
    with open(cfg.debug_jsonl, debug_mode, encoding="utf-8") as debug_fh:
        for item in tqdm(data, desc="Generating", total=len(data)):
            item_id = item.get("id")

            if item_id in completed_ids:
                continue

            try:
                system, user = build_prompt(item, cfg.prompt_strategy)
                prompt = make_chat_prompt(tokenizer, system, user)

                responses: list[str] = []
                if cfg.num_samples > 1:
                    for start in range(0, cfg.num_samples, cfg.inner_batch):
                        chunk = min(cfg.inner_batch, cfg.num_samples - start)
                        responses.extend(
                            generate_batch(
                                llm,
                                tokenizer,
                                [prompt] * chunk,
                                cfg.max_tokens,
                                sampling_params["temperature"],
                                sampling_params["top_p"],
                                sampling_params["top_k"],
                                sampling_params["do_sample"],
                            )
                        )
                    prediction, final_response, counts, votes = majority_vote(item, responses)
                else:
                    responses = generate_batch(
                        llm,
                        tokenizer,
                        [prompt],
                        cfg.max_tokens,
                        sampling_params["temperature"],
                        sampling_params["top_p"],
                        sampling_params["top_k"],
                        sampling_params["do_sample"],
                    )
                    final_response = responses[0] if responses else ""
                    prediction = normalize_vote(item, final_response)
                    counts, votes = {}, [prediction]

                if not prediction:
                    # Avoid blank CSV answers; debug_jsonl will show the underlying model failure.
                    blank_prediction_count += 1
                    prediction = "A" if item.get("options") else "0"

                rec: dict[str, Any] = {
                    "id": item_id,
                    "is_mcq": bool(item.get("options")),
                    "prediction": prediction,
                    "response": final_response,
                    "all_votes": votes,
                    "all_responses": responses if cfg.num_samples > 1 else None,
                    "metadata": {
                        "strategy": cfg.prompt_strategy,
                        "samples": cfg.num_samples,
                        "vote_spread": counts,
                        "max_tokens": cfg.max_tokens,
                        "sampling_params": sampling_params,
                    },
                    "error": None,
                }

                if "answer" in item and judger is not None:
                    rec["gold"] = item["answer"]
                    rec["correct"] = score_response(item, final_response, judger)

            except Exception as e:
                # Keep the run reproducible and complete even if one malformed example fails.
                generation_error_count += 1
                fallback = "A" if item.get("options") else "0"
                rec = {
                    "id": item_id,
                    "is_mcq": bool(item.get("options")),
                    "prediction": fallback,
                    "response": "",
                    "all_votes": [],
                    "all_responses": [],
                    "metadata": {"strategy": cfg.prompt_strategy, "samples": cfg.num_samples},
                    "error": repr(e),
                }

            debug_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            debug_fh.flush()

    n_written, n_missing = rebuild_submission_from_debug(
        data=data,
        debug_jsonl=cfg.debug_jsonl,
        output_csv=cfg.output_csv,
        id_column=cfg.id_column,
        answer_column=cfg.answer_column,
    )

    print(f"Wrote Kaggle submission CSV from checkpoint: {cfg.output_csv}")
    print(f"Wrote/updated debug generations JSONL checkpoint: {cfg.debug_jsonl}")
    print(f"CSV contains {n_written}/{len(data)} completed predictions.")
    if n_missing:
        print(f"WARNING: {n_missing} examples are still missing from the CSV. Rerun with resume to continue.")
    if generation_error_count:
        print(f"WARNING: {generation_error_count} examples hit exceptions. Inspect {cfg.debug_jsonl} before submitting.")
    if blank_prediction_count:
        print(f"WARNING: {blank_prediction_count} examples had blank extracted answers and used fallback values.")
    if any("answer" in item for item in data):
        try:
            print("\nEvaluation summary on labeled data:")
            print(summarize_results(Path(debug_jsonl)))
        except Exception as e:
            print(f"Could not summarize labeled results: {e!r}")

    return cfg.output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final reproducible Qwen3-4B math inference pipeline")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to private/test JSONL input file.")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Path to write Kaggle submission CSV.")
    parser.add_argument("--debug_jsonl", default=DEFAULT_DEBUG_JSONL, help="Path to write debug generations JSONL.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path or HuggingFace Hub repo ID.")
    parser.add_argument("--prompt_strategy", default="structured_cot", choices=list(PROMPT_TEMPLATES))
    parser.add_argument("--num_samples", type=int, default=1, help="Use >1 for self-consistency voting.")
    parser.add_argument("--inner_batch", type=int, default=4, help="Sub-batch size for self-consistency generation.")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id_column", default=DEFAULT_ID_COLUMN)
    parser.add_argument("--answer_column", default=DEFAULT_ANSWER_COLUMN)
    parser.add_argument("--subset_ids", default=None, help="Optional JSON file of IDs for local testing only.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start from scratch by deleting existing output/checkpoint files. Default behavior is resume.",
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Deprecated alias kept for compatibility. Resume is already the default.",
    )
    parser.add_argument(
        "--allow_model_id_override",
        action="store_true",
        help="Bypass required MODEL_ID guardrail. Use only if course staff explicitly permits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        data_path=args.data,
        output_csv=args.output_csv,
        debug_jsonl=args.debug_jsonl,
        adapter_path=args.adapter,
        prompt_strategy=args.prompt_strategy,
        num_samples=args.num_samples,
        inner_batch=args.inner_batch,
        max_tokens=args.max_tokens,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
        id_column=args.id_column,
        answer_column=args.answer_column,
        subset_ids=args.subset_ids,
        overwrite=args.overwrite,
        allow_model_id_override=args.allow_model_id_override,
    )


if __name__ == "__main__":
    main()
