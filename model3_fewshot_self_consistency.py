"""
Model 3: Few-Shot + Self-Consistency for CSE 151B Math Reasoning Competition

Combines Contribution A (few-shot prompting) with Contribution B (self-consistency /
majority-vote decoding). Running both together tests whether structured examples
plus diverse reasoning paths compound their gains.

Usage:
  # Contribution B configs from the milestone report:
  python model3_fewshot_self_consistency.py --subset_ids data/eval_subset.json --num_samples 8 --no_vllm
  python model3_fewshot_self_consistency.py --subset_ids data/eval_subset.json --num_samples 16 --no_vllm

  # Quick sanity check:
  python model3_fewshot_self_consistency.py --subset_ids data/eval_subset.json --num_samples 4 --limit 20 --no_vllm
"""

import argparse
import collections
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
from model3_prompt_engineering import VARIANTS as PE_VARIANTS

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
DATA_PATH = "data/public.jsonl"
MAX_TOKENS = 768

# Sampling params for self-consistency — diversity is essential; do NOT use greedy
SAMPLING_PARAMS = dict(
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.05,
    do_sample=True,
)

# ---------------------------------------------------------------------------
# Prompt builder (Contribution A few-shot variant)
# ---------------------------------------------------------------------------

def build_fewshot_prompt(item: dict) -> tuple[str, str]:
    """
    Uses the few-shot variant from model1_prompt_engineering (v2_fewshot).
    Falls back to structured_cot from Contribution A if model1 is not present.
    """
    try:
        from model1_prompt_engineering import VARIANTS as M1_VARIANTS
        cfg = M1_VARIANTS["v2_fewshot"]
        if item.get("options"):
            user = cfg["prefix_mcq"] + f'{item["question"]}\n\nOptions:\n{format_options(item["options"])}'
            return cfg["sys_mcq"], user
        return cfg["sys_math"], cfg["prefix_math"] + item["question"]
    except (ImportError, KeyError):
        cfg = PE_VARIANTS["structured_cot"]
        if item.get("options"):
            user = f'{item["question"]}\n\nOptions:\n{format_options(item["options"])}'
            return cfg["sys_mcq"], user
        return cfg["sys_math"], item["question"]


# ---------------------------------------------------------------------------
# Vote extraction & normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Strip LaTeX whitespace variants so equivalent expressions count as the same answer."""
    import re
    return re.sub(r"\s+", "", text).strip().lower()


def normalize_vote(item: dict, response: str) -> str:
    """Extract the canonical answer string from a model response."""
    if item.get("options"):
        letter = extract_letter(response)
        return letter.upper() if letter else ""
    boxed = extract_boxed(response)
    if boxed:
        return _normalise(boxed)
    return _normalise(response.strip()[-200:])


def majority_vote(
    item: dict,
    responses: list[str],
) -> tuple[str, str, dict, list[str]]:
    """
    Run majority vote over responses.

    Returns
    -------
    best_vote     : the winning canonical answer string
    best_response : the full response text for the winner
    counts        : {canonical_answer: vote_count}
    votes         : per-response vote list (parallel to responses)
    """
    votes = [normalize_vote(item, r) for r in responses]
    counts = collections.Counter(v for v in votes if v)

    if not counts:
        return "", responses[0] if responses else "", {}, votes

    # Tie-break by average response length of supporting samples
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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Model 3: Few-Shot + Self-Consistency (Contributions A + B)"
    )
    p.add_argument("--data", default=DATA_PATH,
                   help="Path to the full dataset JSONL file.")
    p.add_argument("--subset_ids", default=None,
                   help="Path to JSON file of IDs to evaluate (e.g. data/eval_subset.json). "
                        "If provided, only questions whose ID appears in this file are run.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of items after subset filtering (for quick tests).")
    p.add_argument("--gpu", default="0",
                   help="CUDA device(s), e.g. '0' or '0,1'.")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--num_samples", type=int, default=8,
                   help="Number of sampled responses per question. Use 8 or 16 per milestone.")
    p.add_argument("--inner_batch", type=int, default=None,
                   help="GPU sub-batch size for the N samples. Defaults to min(num_samples, 8). "
                        "Reduce if you hit OOM.")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load model in 4-bit (bitsandbytes) to save VRAM.")
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

    N = args.num_samples
    inner_batch = args.inner_batch or min(N, 8)

    out_path = (
        Path(args.out_dir)
        / f"model3_fewshot_self_consistency_N{N}_results.jsonl"
    )

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
    print(f"Few-Shot + Self-Consistency | N={N} | todo={len(remaining)} (skipping {len(done)})")

    with open(out_path, "a") as fh:
        for item in tqdm(remaining, desc=f"fewshot_sc_N{N}"):
            system, user = build_fewshot_prompt(item)
            prompt = make_chat_prompt(tokenizer, system, user)

            # Sub-batched sampling to avoid OOM at large N
            responses: list[str] = []
            for start in range(0, N, inner_batch):
                chunk = min(inner_batch, N - start)
                chunk_responses = generate_batch(
                    llm, tokenizer,
                    [prompt] * chunk,
                    max_new_tokens=args.max_tokens,
                    **SAMPLING_PARAMS,
                )
                responses.extend(chunk_responses)

            best_vote, best_response, counts, votes = majority_vote(item, responses)

            rec: dict = {
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

    print(f"Saved -> {out_path}")
    print("Summary:", summarize_results(out_path))


if __name__ == "__main__":
    main()
