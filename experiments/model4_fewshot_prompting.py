"""
Model 4: Few-Shot Prompting for CSE 151B Math Reasoning Competition

Contribution A: Few-shot prompting with curated Math/AMC-style examples.
  - Separate example banks for MCQ and free-response question types.
  - Single-sample inference only — no majority voting — to isolate the
    contribution of few-shot examples from self-consistency.
  - Batched GPU inference for throughput.

Usage:
  python model4_fewshot_prompting.py --subset_ids data/eval_subset.json --no_vllm
  python model4_fewshot_prompting.py --subset_ids data/eval_subset.json --no_vllm --limit 10
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from common_utils import (
    MODEL_ID,
    STARTER_SYSTEM_PROMPT_MATH,
    STARTER_SYSTEM_PROMPT_MCQ,
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
MAX_TOKENS = 768        # raised: few-shot prefix + CoT chain needs more room
DEFAULT_BATCH = 8

# Greedy decoding — isolates the effect of few-shot examples, no sampling noise
SAMPLING_PARAMS = dict(
    temperature=1.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.05,
    do_sample=False,
)

# ---------------------------------------------------------------------------
# Few-shot example banks  (Contribution A)
# 3 examples each: covers numeric, multi-answer, and symbolic free-response;
# covers letter choice with shown working for MCQ.
# ---------------------------------------------------------------------------

FEWSHOT_FREE = """\
Here are worked examples showing the exact expected format.

Example 1:
Problem: Solve $3x - 7 = 14$. $x =$ [ANS]
Solution: Add 7 to both sides: $3x = 21$. Divide by 3: $x = 7$.
Answer: \\boxed{7}

Example 2:
Problem: Find all real solutions to $x^2 - 5x + 6 = 0$. Solutions: [ANS]
Solution: Factor: $(x-2)(x-3) = 0$, so $x = 2$ or $x = 3$.
Answer: \\boxed{2, 3}

Example 3:
Problem: A geometric sequence has first term $a_1 = 3$ and common ratio $r = 2$. \
What is the 5th term? [ANS]
Solution: The $n$-th term is $a_n = a_1 \\cdot r^{n-1}$. \
So $a_5 = 3 \\cdot 2^4 = 3 \\cdot 16 = 48$.
Answer: \\boxed{48}

Now solve the following problem.
"""

FEWSHOT_MCQ = """\
Here are worked examples showing the exact expected format.

Example 1:
Problem: What is $\\binom{5}{2}$?
Options:
A. 5
B. 10
C. 15
D. 20
Solution: $\\binom{5}{2} = \\frac{5!}{2!3!} = 10$. The correct choice is B.
Answer: \\boxed{B}

Example 2:
Problem: If $f(x) = x^2 + 1$, what is $f(3)$?
Options:
A. 8
B. 9
C. 10
D. 12
Solution: $f(3) = 3^2 + 1 = 10$. The correct choice is C.
Answer: \\boxed{C}

Example 3:
Problem: How many ways can 4 people be arranged in a line?
Options:
A. 4
B. 8
C. 16
D. 24
Solution: The number of permutations of 4 people is $4! = 24$. The correct choice is D.
Answer: \\boxed{D}

Now solve the following problem.
"""

SYSTEM_FREE = STARTER_SYSTEM_PROMPT_MATH + (
    " Use the examples above to match the required answer format exactly."
)
SYSTEM_MCQ = STARTER_SYSTEM_PROMPT_MCQ + (
    " Use the examples above to match the required answer format exactly."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(item: dict) -> tuple[str, str]:
    if item.get("options"):
        user = FEWSHOT_MCQ + f'\nProblem: {item["question"]}\n\nOptions:\n{format_options(item["options"])}'
        return SYSTEM_MCQ, user
    return SYSTEM_FREE, FEWSHOT_FREE + f"\nProblem: {item['question']}"


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
    p = argparse.ArgumentParser(description="Model 4: Few-shot prompting (Contribution A)")
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--subset_ids", default=None,
                   help="Path to JSON file of IDs to evaluate (e.g. data/eval_subset.json).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH,
                   help="Prompts per GPU forward pass. Reduce if OOM.")
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

    data = load_jsonl(args.data, args.limit)
    if args.subset_ids:
        subset = load_subset_ids(args.subset_ids)
        before = len(data)
        data = [d for d in data if d.get("id") in subset]
        print(f"Subset filter: {before} -> {len(data)} items  (from {args.subset_ids})")

    tokenizer, llm = load_model_and_tokenizer(MODEL_ID, load_in_4bit=args.load_in_4bit)
    judger = load_judger()

    out_path = Path(args.out_dir) / "model4_fewshot_prompting_results.jsonl"
    done = load_done_ids(out_path)
    remaining = [d for d in data if d.get("id") not in done]
    print(f"Few-shot prompting | todo={len(remaining)}  (skipping {len(done)} already done)")

    with open(out_path, "a") as fh:
        for batch_start in tqdm(range(0, len(remaining), args.batch_size), desc="fewshot", unit="batch"):
            batch = remaining[batch_start : batch_start + args.batch_size]
            prompts = [make_chat_prompt(tokenizer, *build_prompt(item)) for item in batch]
            responses = generate_batch(llm, tokenizer, prompts,
                                       max_new_tokens=args.max_tokens, **SAMPLING_PARAMS)
            for item, response in zip(batch, responses):
                rec = {
                    "id": item.get("id"),
                    "model": "fewshot_prompting",
                    "is_mcq": bool(item.get("options")),
                    "response": response,
                }
                if "answer" in item:
                    rec["gold"] = item["answer"]
                    rec["correct"] = score_response(item, response, judger)
                fh.write(json.dumps(rec) + "\n")
            fh.flush()

    print("Saved to:", out_path)
    print("Summary:", summarize_results(out_path))


if __name__ == "__main__":
    main()
