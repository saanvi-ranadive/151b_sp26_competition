"""
Model 3: Structured Prompt Engineering for CSE 151B Math Reasoning Competition

Contribution A: Prompt Engineering
  - Structured chain-of-thought system prompts with type-aware formatting.
  - MCQ vs free-response split handled separately for each variant.
  - No majority voting — isolates the prompt engineering contribution cleanly.
  - Batched GPU inference for throughput; greedy decoding for determinism.

Variants:
  structured_cot      : decomposition + theorem/formula selection + boxed answer
  verification_prompt : solve, verify, then produce final answer
  answer_format_only  : strict competition-style final-answer formatting

Usage:
  python model3_prompt_engineering.py --variant all --subset_ids data/eval_subset.json --no_vllm
  python model3_prompt_engineering.py --variant structured_cot --subset_ids data/eval_subset.json --no_vllm
"""

import argparse
import json
import os
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
DATA_PATH = "data/public.jsonl"
MAX_TOKENS = 768
DEFAULT_BATCH = 8

# Greedy for the prompt-engineering ablation (no sampling noise — cleaner comparison)
SAMPLING_PARAMS = dict(
    temperature=1.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.05,
    do_sample=False,
)

# ---------------------------------------------------------------------------
# System-prompt variants  (Contribution A)
# ---------------------------------------------------------------------------

STRUCTURED_MATH = (
    "You are an expert mathematician. Solve the problem step by step using this structure:\n"
    "1. Identify the problem type and what is being asked.\n"
    "2. State the relevant theorem, formula, or strategy.\n"
    "3. Carry out the computation or proof carefully, showing every step.\n"
    "4. Check the result for consistency (units, edge cases, sign).\n"
    "5. State the final answer inside \\boxed{}. "
    "For multiple [ANS] blanks, place all values in one box as a comma-separated list."
)

STRUCTURED_MCQ = (
    "You are an expert mathematician. "
    "First solve the problem completely without looking at the answer choices. "
    "Then compare your result to the options and select the best match. "
    "End your response with ONLY the chosen option letter inside \\boxed{}, e.g. \\boxed{C}. "
    "Do not include any text after the box."
)

VERIFY_MATH = (
    "You are an expert mathematician. "
    "Solve the problem step by step, then independently verify your answer "
    "by substitution, recalculation, or checking boundary cases. "
    "If verification fails, revise and re-verify. "
    "Put the final verified answer inside \\boxed{}."
)

VERIFY_MCQ = (
    "You are an expert mathematician. "
    "Solve the problem independently, verify that your answer matches the correct option, "
    "and briefly rule out the other choices. "
    "End with ONLY the option letter inside \\boxed{}, e.g. \\boxed{B}."
)

FORMAT_MATH = (
    "You are an expert mathematician. Solve the problem step by step. "
    "The very last line of your response must be exactly:\n"
    "Final answer: \\boxed{<answer>}\n"
    "Do not include explanations or units inside the box."
)

FORMAT_MCQ = (
    "You are an expert mathematician. Read the problem and the answer choices, then solve. "
    "The very last line of your response must be exactly:\n"
    "Final answer: \\boxed{<letter>}\n"
    "where <letter> is one of A, B, C, D, E."
)

VARIANTS: dict[str, dict] = {
    "structured_cot": {
        "sys_math": STRUCTURED_MATH,
        "sys_mcq": STRUCTURED_MCQ,
        "description": "Contribution A: structured 5-step CoT with type-aware instructions",
    },
    "verification_prompt": {
        "sys_math": VERIFY_MATH,
        "sys_mcq": VERIFY_MCQ,
        "description": "Contribution A: explicit self-verification before committing to an answer",
    },
    "answer_format_only": {
        "sys_math": FORMAT_MATH,
        "sys_mcq": FORMAT_MCQ,
        "description": "Contribution A: strict competition answer-format enforcement",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(item: dict, cfg: dict) -> tuple[str, str]:
    """Return (system_prompt, user_message) for this item and variant config."""
    if item.get("options"):
        user = f'{item["question"]}\n\nOptions:\n{format_options(item["options"])}'
        return cfg["sys_mcq"], user
    return cfg["sys_math"], item["question"]


def load_done_ids(out_path: Path) -> set:
    """Return set of already-processed IDs for resumable runs."""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def load_subset_ids(path: str) -> set:
    """
    Load a set of IDs from a JSON file.
    Accepts either a plain list of IDs or a list of dicts with an 'id' key.
    """
    with open(path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(raw).__name__}.")
    if len(raw) == 0:
        return set()
    if isinstance(raw[0], dict):
        return {item["id"] for item in raw}
    return set(raw)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_variant(
    variant: str,
    llm,
    tokenizer,
    data: list[dict],
    judger,
    args,
) -> None:
    cfg = VARIANTS[variant]
    out_path = Path(args.out_dir) / f"model3_prompt_{variant}_results.jsonl"
    done = load_done_ids(out_path)
    remaining = [d for d in data if d.get("id") not in done]
    print(f"\nVariant : {variant}")
    print(f"Desc    : {cfg['description']}")
    print(f"Todo    : {len(remaining)} items  (skipping {len(done)} already done)")

    batch_size = args.batch_size
    with open(out_path, "a") as fh:
        for batch_start in tqdm(
            range(0, len(remaining), batch_size),
            desc=variant,
            unit="batch",
        ):
            batch = remaining[batch_start : batch_start + batch_size]

            prompts = []
            for item in batch:
                system, user = build_prompt(item, cfg)
                prompts.append(make_chat_prompt(tokenizer, system, user))

            responses = generate_batch(
                llm, tokenizer, prompts,
                max_new_tokens=args.max_tokens,
                **SAMPLING_PARAMS,
            )

            for item, response in zip(batch, responses):
                rec = {
                    "id": item.get("id"),
                    "variant": variant,
                    "is_mcq": bool(item.get("options")),
                    "response": response,
                }
                if "answer" in item:
                    rec["gold"] = item["answer"]
                    rec["correct"] = score_response(item, response, judger)

                fh.write(json.dumps(rec) + "\n")
            fh.flush()

    print(f"Saved  -> {out_path}")
    print("Summary:", summarize_results(out_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Model 3: Structured prompt engineering (Contribution A)")
    p.add_argument("--variant", default="all", choices=list(VARIANTS) + ["all"],
                   help="Which prompt variant to run, or 'all' to run all three.")
    p.add_argument("--data", default=DATA_PATH,
                   help="Path to the full dataset JSONL file.")
    p.add_argument("--subset_ids", default=None,
                   help="Path to JSON file of IDs to evaluate (e.g. data/eval_subset.json). "
                        "If provided, only questions whose ID appears in this file are run.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of items after subset filtering (for quick tests).")
    p.add_argument("--gpu", default="0",
                   help="CUDA device index/indices, e.g. '0' or '0,1'.")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH,
                   help="Number of prompts per GPU forward pass. Reduce if you hit OOM.")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load model in 4-bit (bitsandbytes) to reduce VRAM usage.")
    p.add_argument("--no_vllm", action="store_true",
                   help="Disable vLLM and use standard HuggingFace generation instead.")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.no_vllm:
        os.environ["USE_VLLM"] = "0"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}  ({props.total_memory // 1024**3} GB)")

    data = load_jsonl(args.data, args.limit)

    if args.subset_ids:
        subset = load_subset_ids(args.subset_ids)
        before = len(data)
        data = [d for d in data if d.get("id") in subset]
        print(f"Subset filter: {before} -> {len(data)} items  (from {args.subset_ids})")

    tokenizer, llm = load_model_and_tokenizer(MODEL_ID, load_in_4bit=args.load_in_4bit)
    judger = load_judger()

    variants_to_run = list(VARIANTS) if args.variant == "all" else [args.variant]
    for v in variants_to_run:
        run_variant(v, llm, tokenizer, data, judger, args)


if __name__ == "__main__":
    main()
