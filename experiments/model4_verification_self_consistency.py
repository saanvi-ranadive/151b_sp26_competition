"""
Model 4: Verification Prompt + Self-Consistency for CSE 151B Math Reasoning Competition

Contribution B: Tests whether explicit solve-then-verify prompting improves the
quality of individual self-consistency samples, combining Contributions A and B.

  - Uses the v3_verification prompt from model1_prompt_engineering (solve → verify → answer).
  - Majority-votes N sampled responses per question (N=8 or N=16 per milestone report).
  - Sub-batched GPU inference so all N samples are generated efficiently without OOM.
  - Improved tie-breaking: longer supporting responses preferred over arbitrary list order.
  - Robust vote normalisation: strips LaTeX whitespace before counting.

Usage:
  python model4_verification_self_consistency.py --subset_ids data/eval_subset.json --num_samples 8  --no_vllm
  python model4_verification_self_consistency.py --subset_ids data/eval_subset.json --num_samples 16 --no_vllm
"""

import argparse
import collections
import json
import os
import re
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
MAX_TOKENS = 768        # raised: verification pass needs extra tokens

# Diverse sampling for self-consistency
SAMPLING_PARAMS = dict(
    temperature=0.7,
    top_p=0.95,
    top_k=50,           # raised from 20 for better diversity at N=16
    repetition_penalty=1.05,
    do_sample=True,
)

# ---------------------------------------------------------------------------
# Prompt builder — pulls verification variant from model1
# Falls back to an inline verification prompt if model1 is absent
# ---------------------------------------------------------------------------

FALLBACK_VERIFY_MATH = (
    "You are an expert mathematician. "
    "Solve the problem step by step, then independently verify your answer "
    "by substitution or recalculation. If verification fails, revise. "
    "Put the final verified answer inside \\boxed{}."
)
FALLBACK_VERIFY_MCQ = (
    "You are an expert mathematician. "
    "Solve independently, verify your answer matches the correct option, "
    "and briefly rule out the others. "
    "End with ONLY the option letter inside \\boxed{}, e.g. \\boxed{B}."
)

def build_verification_prompt(item: dict) -> tuple[str, str]:
    try:
        from model1_prompt_engineering import VARIANTS
        cfg = VARIANTS["v3_verification"]
    except (ImportError, KeyError):
        cfg = {"sys_math": FALLBACK_VERIFY_MATH, "sys_mcq": FALLBACK_VERIFY_MCQ}

    if item.get("options"):
        user = f'{item["question"]}\n\nOptions:\n{format_options(item["options"])}'
        return cfg["sys_mcq"], user
    return cfg["sys_math"], item["question"]


# ---------------------------------------------------------------------------
# Vote normalisation & majority vote
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def normalize_vote(item: dict, response: str) -> str:
    if item.get("options"):
        letter = extract_letter(response)
        return letter.upper() if letter else ""
    boxed = extract_boxed(response)
    if boxed:
        return _normalise(boxed)
    return _normalise(response.strip()[-200:])


def majority_vote(item: dict, responses: list[str]) -> tuple[str, str, dict, list[str]]:
    votes = [normalize_vote(item, r) for r in responses]
    counts = collections.Counter(v for v in votes if v)

    if not counts:
        return "", responses[0] if responses else "", {}, votes

    def score(pair):
        vote, count = pair
        supporters = [r for r, v in zip(responses, votes) if v == vote]
        avg_len = sum(len(r) for r in supporters) / max(len(supporters), 1)
        return (count, avg_len)

    best_vote = max(counts.items(), key=score)[0]
    best_response = max(
        [r for r, v in zip(responses, votes) if v == best_vote],
        key=len,
    )
    return best_vote, best_response, dict(counts), votes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_done_ids(out_path: Path) -> set:
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
    with open(path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    if not raw:
        return set()
    return {item["id"] for item in raw} if isinstance(raw[0], dict) else set(raw)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Model 4: Verification + Self-Consistency (Contributions A+B)")
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--subset_ids", default=None,
                   help="Path to JSON file of IDs to evaluate (e.g. data/eval_subset.json).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--num_samples", type=int, default=8,
                   help="Responses per question for majority vote. Use 8 or 16 per milestone.")
    p.add_argument("--inner_batch", type=int, default=None,
                   help="GPU sub-batch size for the N samples. Defaults to min(num_samples, 8).")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_vllm", action="store_true",
                   help="Disable vLLM; use standard HuggingFace generation.")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.no_vllm:
        os.environ["USE_VLLM"] = "0"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}  ({props.total_memory // 1024**3} GB)")

    N = args.num_samples
    inner_batch = args.inner_batch or min(N, 8)
    out_path = Path(args.out_dir) / f"model4_verification_self_consistency_N{N}_results.jsonl"

    data = load_jsonl(args.data, args.limit)
    if args.subset_ids:
        subset = load_subset_ids(args.subset_ids)
        before = len(data)
        data = [d for d in data if d.get("id") in subset]
        print(f"Subset filter: {before} -> {len(data)} items  (from {args.subset_ids})")

    tokenizer, llm = load_model_and_tokenizer(MODEL_ID, load_in_4bit=args.load_in_4bit)
    judger = load_judger()

    done = load_done_ids(out_path)
    remaining = [d for d in data if d.get("id") not in done]
    print(f"Verification SC | N={N} | todo={len(remaining)}  (skipping {len(done)})")

    with open(out_path, "a") as fh:
        for item in tqdm(remaining, desc=f"verify_sc_N{N}"):
            system, user = build_verification_prompt(item)
            prompt = make_chat_prompt(tokenizer, system, user)

            responses: list[str] = []
            for start in range(0, N, inner_batch):
                chunk = min(inner_batch, N - start)
                responses.extend(generate_batch(
                    llm, tokenizer, [prompt] * chunk,
                    max_new_tokens=args.max_tokens, **SAMPLING_PARAMS,
                ))

            best_vote, best_response, counts, votes = majority_vote(item, responses)

            rec = {
                "id": item.get("id"),
                "is_mcq": bool(item.get("options")),
                "num_samples": N,
                "vote": best_vote,
                "vote_counts": counts,
                "votes": votes,
                "response": best_response,
                "all_responses": responses,
            }
            if "answer" in item:
                rec["gold"] = item["answer"]
                rec["correct"] = score_response(item, best_response, judger)

            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    print("Saved to:", out_path)
    print("Summary:", summarize_results(out_path))


if __name__ == "__main__":
    main()
