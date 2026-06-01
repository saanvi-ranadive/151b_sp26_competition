"""
Model 1: Prompt Engineering for CSE 151B Math Reasoning Competition

Compares 4 prompt strategies while keeping model weights and sampling fixed:
  v0_baseline     — exact starter-code prompts (control, matches Model 0)
  v1_enhanced_cot — explicit numbered reasoning structure
  v2_fewshot      — worked examples prepended to the user turn
  v3_verification — solve, verify, then commit to final boxed answer

Key improvements over original:
  - vLLM batched inference with Transformers INT4 fallback (matches Model 0)
  - --subset_ids for apples-to-apples comparison across all models
  - MAX_TOKENS raised to 1024 (512 truncates thinking-model responses)
  - All variants share one model load (no redundant reloading)
  - left-padding set correctly for decoder-only batched generation

Usage:
  # Run all variants on the shared eval subset
  python model1_prompt_engineering.py --subset_ids data/eval_subset.json

  # Run a single variant
  python model1_prompt_engineering.py --variant v2_fewshot --subset_ids data/eval_subset.json

  # Quick smoke-test: 5 questions, one variant
  python model1_prompt_engineering.py --variant v1_enhanced_cot --limit 5
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "data/public.jsonl"
MAX_TOKENS = 8192   # 512 truncates chain-of-thought; raise if you still see cut-off answers
SAMPLING_PARAMS = dict(temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0)


# ── Prompt variants ───────────────────────────────────────────────────────────

# v0: exact starter prompts (control — should reproduce Model 0 scores)
V0_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
V0_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

# v1: numbered CoT structure
V1_MATH = (
    "You are an expert mathematician. Solve using this structure:\n"
    "1. Identify what is asked.\n"
    "2. Choose relevant formulas/theorems.\n"
    "3. Compute carefully, showing each step.\n"
    "4. Sanity-check the answer.\n"
    "5. Put the final answer inside \\boxed{}. "
    "For multiple sub-answers use \\boxed{a, b}."
)
V1_MCQ = (
    "You are an expert mathematician. "
    "Solve the problem independently first, then compare your result to the choices. "
    "Rule out wrong options briefly, then output ONLY the chosen letter inside \\boxed{}."
)

# v2: few-shot examples (prepended to the user turn)
FEWSHOT_FREE = (
    "Here are two examples of the expected format.\n\n"
    "Example 1: Solve $3x - 7 = 14$.\n"
    "Solution: Add 7 to both sides: $3x = 21$, so $x = 7$.\n"
    "Answer: \\boxed{7}\n\n"
    "Example 2: Solve $x^2 - 5x + 6 = 0$.\n"
    "Solution: Factor: $(x-2)(x-3) = 0$, so $x = 2$ or $x = 3$.\n"
    "Answer: \\boxed{2, 3}\n\n"
    "Now solve this problem:\n"
)
FEWSHOT_MCQ = (
    "Here are two examples of the expected format.\n\n"
    "Example 1: What is $\\binom{5}{2}$?\n"
    "Options:\nA. 5\nB. 10\nC. 15\nD. 20\n"
    "Solution: $\\binom{5}{2} = \\frac{5!}{2!3!} = 10$, which matches option B.\n"
    "Answer: \\boxed{B}\n\n"
    "Example 2: The derivative of $\\sin x$ is:\n"
    "Options:\nA. $-\\cos x$\nB. $\\tan x$\nC. $\\cos x$\nD. $-\\sin x$\n"
    "Solution: $\\frac{d}{dx}\\sin x = \\cos x$, which is option C.\n"
    "Answer: \\boxed{C}\n\n"
    "Now solve this problem:\n"
)

# v3: solve then verify
V3_MATH = (
    "You are an expert mathematician. "
    "First, solve the problem step-by-step. "
    "Then verify your answer by substitution, recalculation, or a dimensional/sanity check. "
    "If your verification reveals an error, correct it. "
    "Put the final verified answer inside \\boxed{}. "
    "For multiple sub-answers use \\boxed{a, b}."
)
V3_MCQ = (
    "You are an expert mathematician. "
    "Solve the problem independently, then verify that your chosen option is consistent "
    "and that the remaining options are clearly worse. "
    "Output ONLY the chosen letter inside \\boxed{}."
)

VARIANTS = {
    "v0_baseline": {
        "sys_math": V0_MATH, "sys_mcq": V0_MCQ,
        "prefix_math": "", "prefix_mcq": "",
        "description": "starter prompts (control)",
    },
    "v1_enhanced_cot": {
        "sys_math": V1_MATH, "sys_mcq": V1_MCQ,
        "prefix_math": "", "prefix_mcq": "",
        "description": "numbered CoT structure",
    },
    "v2_fewshot": {
        "sys_math": V1_MATH, "sys_mcq": V1_MCQ,
        "prefix_math": FEWSHOT_FREE, "prefix_mcq": FEWSHOT_MCQ,
        "description": "few-shot examples + numbered CoT",
    },
    "v3_verification": {
        "sys_math": V3_MATH, "sys_mcq": V3_MCQ,
        "prefix_math": "", "prefix_mcq": "",
        "description": "solve then verify",
    },
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str, limit: int | None = None, ids: list | None = None) -> list[dict]:
    data = [json.loads(line) for line in open(path)]
    if ids is not None:
        id_set = set(ids)
        data = [d for d in data if d.get("id") in id_set]
        print(f"  Filtered to {len(data)} questions matching subset IDs.")
    if limit:
        data = data[:limit]
    return data


def format_options(options: list) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    return "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))


def build_prompt(item: dict, cfg: dict) -> tuple[str, str]:
    if item.get("options"):
        user = cfg["prefix_mcq"] + f'{item["question"]}\n\nOptions:\n{format_options(item["options"])}'
        return cfg["sys_mcq"], user
    return cfg["sys_math"], cfg["prefix_math"] + item["question"]


def make_chat_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Scoring ───────────────────────────────────────────────────────────────────

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_response(item: dict, response: str, judger) -> bool:
    gold = item["answer"]
    if item.get("options"):
        return extract_letter(response) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list))
    except Exception:
        return False


def summarize_results(out_path: Path) -> dict:
    results = [json.loads(l) for l in open(out_path) if "correct" in l]
    if not results:
        return {"note": "no scored results"}
    mcq  = [r for r in results if r["is_mcq"]]
    free = [r for r in results if not r["is_mcq"]]
    def acc(s): return round(sum(r["correct"] for r in s) / len(s) * 100, 2) if s else 0.0
    return {
        "total":    len(results),
        "mcq_acc":  f"{acc(mcq)}%  ({sum(r['correct'] for r in mcq)}/{len(mcq)})",
        "free_acc": f"{acc(free)}%  ({sum(r['correct'] for r in free)}/{len(free)})",
        "overall":  f"{acc(results)}%  ({sum(r['correct'] for r in results)}/{len(results)})",
    }


# ── Per-variant runner ────────────────────────────────────────────────────────

def run_variant(variant: str, llm, tokenizer, data: list, judger, args):
    cfg      = VARIANTS[variant]
    out_path = Path(args.out_dir) / f"model1_{variant}_results.jsonl"

    done = set()
    if out_path.exists():
        for line in open(out_path):
            done.add(json.loads(line)["id"])
    remaining = [d for d in data if d.get("id") not in done]
    print(f"\n{'='*55}")
    print(f"Variant : {variant}  —  {cfg['description']}")
    print(f"Remaining: {len(remaining)}  (already done: {len(done)})")
    print(f"{'='*55}")
    if not remaining:
        print("Nothing to do. Summary:", summarize_results(out_path))
        return

    # Build all prompts for this variant
    prompts = [make_chat_prompt(tokenizer, *build_prompt(item, cfg)) for item in remaining]

    # Generate — vLLM path
    if hasattr(llm, "generate") and hasattr(llm, "llm_engine"):
        from vllm import SamplingParams
        sampling_params = SamplingParams(max_tokens=args.max_tokens, **SAMPLING_PARAMS)
        outputs   = llm.generate(prompts, sampling_params=sampling_params)
        responses = [out.outputs[0].text.strip() for out in outputs]

    # Generate — Transformers path (batched)
    else:
        import torch
        responses = []
        for start in tqdm(range(0, len(prompts), args.hf_batch_size), desc=f"{variant} batches"):
            batch = prompts[start:start + args.hf_batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_model_len,
            ).to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=SAMPLING_PARAMS["temperature"],
                    top_p=SAMPLING_PARAMS["top_p"],
                    top_k=SAMPLING_PARAMS["top_k"],
                    repetition_penalty=SAMPLING_PARAMS["repetition_penalty"],
                    do_sample=True,
                )
            plen = inputs["input_ids"].shape[1]
            for o in out_ids:
                responses.append(tokenizer.decode(o[plen:], skip_special_tokens=True).strip())

    # Score and save
    with open(out_path, "a") as f:
        for item, response in tqdm(zip(remaining, responses), total=len(remaining), desc="Scoring"):
            rec = {
                "id":      item.get("id"),
                "variant": variant,
                "is_mcq":  bool(item.get("options")),
                "response": response,
            }
            if "answer" in item:
                rec["gold"]    = item["answer"]
                rec["correct"] = score_response(item, response, judger)
            f.write(json.dumps(rec) + "\n")
            f.flush()

    print("Summary:", summarize_results(out_path))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model 1: Prompt Engineering")
    parser.add_argument("--variant",  default="all", choices=list(VARIANTS) + ["all"])
    parser.add_argument("--data",     default=DATA_PATH)
    parser.add_argument("--limit",    type=int, default=None,
                        help="Cap total questions (useful for smoke-tests)")
    parser.add_argument("--gpu",      default="0")
    parser.add_argument("--out_dir",  default="results")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)

    # Apples-to-apples subset (same file used by Model 0)
    parser.add_argument("--subset_ids", default=None,
                        help="Path to eval_subset.json created by model0_starter_baseline.py --make_subset")

    # vLLM / HF fallback
    parser.add_argument("--no_vllm",       action="store_true",
                        help="Skip vLLM and use Transformers+INT4 directly")
    parser.add_argument("--hf_batch_size", type=int, default=4)
    parser.add_argument("--gpu_mem_util",  type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int,   default=16384)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    Path(args.out_dir).mkdir(exist_ok=True)

    # Load data
    subset_ids = None
    if args.subset_ids:
        if not Path(args.subset_ids).exists():
            print(f"ERROR: subset file not found: {args.subset_ids}")
            print("Run model0_starter_baseline.py --make_subset first.")
            sys.exit(1)
        subset_ids = json.load(open(args.subset_ids))
        print(f"Using fixed subset: {args.subset_ids} ({len(subset_ids)} IDs)")

    data = load_jsonl(args.data, limit=args.limit, ids=subset_ids)
    print(f"Dataset: {args.data} | Questions: {len(data)}")

    # Load tokenizer
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "left"   # required for correct batched generation

    # Load model — try vLLM, fall back to Transformers INT4
    llm = None
    if not args.no_vllm:
        try:
            from vllm import LLM
            print("Loading model via vLLM...")
            llm = LLM(
                model=MODEL_ID,
                gpu_memory_utilization=args.gpu_mem_util,
                max_model_len=args.max_model_len,
                trust_remote_code=True,
            )
            print("vLLM model ready.")
        except Exception as e:
            print(f"vLLM failed ({e})\nFalling back to Transformers...")
            llm = None

    if llm is None:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        print("Loading model via Transformers (INT4)...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True,
            quantization_config=bnb, device_map="auto",
        )
        llm.eval()
        print("Transformers model ready.")

    # Load judger once, shared across all variants
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    # Run selected variants
    variants_to_run = list(VARIANTS) if args.variant == "all" else [args.variant]
    for v in variants_to_run:
        run_variant(v, llm, tokenizer, data, judger, args)

    print("\nAll done. Results in:", args.out_dir)


if __name__ == "__main__":
    main()
