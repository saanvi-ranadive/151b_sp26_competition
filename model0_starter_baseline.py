"""
Model 0: Optimized Starter Baseline for CSE 151B Math Reasoning Competition

Key improvements over the original:
  - vLLM backend for batched inference (5-10x faster than HF generate loop)
  - --subset_ids flag: pin evaluation to a fixed JSON list of IDs so every
    model is tested on the exact same questions (apples-to-apples)
  - --make_subset helper: create and save that shared ID list once
  - Resume support: skips already-completed IDs
  - Configurable batch size, token budget, and quantization

Usage
-----
# 1. Create a fixed 50-question subset (run once, shared across all models)
python model0_starter_baseline.py --make_subset --subset_size 50 --data data/public.jsonl --subset_ids data/eval_subset.json

# 2. Run Model 0 on that fixed subset
python model0_starter_baseline.py --data data/public.jsonl --subset_ids data/eval_subset.json

# 3. Run on the full dataset
python model0_starter_baseline.py --data data/public.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH  = "data/public.jsonl"
MAX_TOKENS = 1024   # 2048 was unnecessarily slow; increase if answers are truncated
SAMPLING_PARAMS = dict(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str, ids: list[str] | None = None) -> list[dict]:
    """Load JSONL, optionally filtering to a fixed list of IDs."""
    data = [json.loads(line) for line in open(path)]
    if ids is not None:
        id_set = set(ids)
        data = [d for d in data if d.get("id") in id_set]
        print(f"  Filtered to {len(data)} questions matching subset IDs.")
    return data


def make_and_save_subset(data_path: str, subset_size: int, out_path: str, seed: int = 42):
    """
    Sample a balanced subset of IDs (half MCQ, half free-form where possible)
    and save to a JSON file. Run this ONCE and reuse across all models.
    """
    data = [json.loads(line) for line in open(data_path)]
    mcq   = [d["id"] for d in data if d.get("options")]
    free  = [d["id"] for d in data if not d.get("options")]

    rng = random.Random(seed)
    half = subset_size // 2
    sampled_mcq  = rng.sample(mcq,  min(half, len(mcq)))
    sampled_free = rng.sample(free, min(subset_size - len(sampled_mcq), len(free)))
    subset_ids   = sampled_mcq + sampled_free
    rng.shuffle(subset_ids)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(subset_ids, f, indent=2)

    print(f"Saved {len(subset_ids)} subset IDs → {out_path}")
    print(f"  MCQ: {len(sampled_mcq)}  |  Free-form: {len(sampled_free)}")
    print("Share this file with all models to guarantee apples-to-apples comparison.")


# ── Prompt helpers ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: list | None) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def make_chat_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Scoring ────────────────────────────────────────────────────────────────────

import re

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
        return {"note": "No scored results yet (private set or no answers available)"}
    mcq  = [r for r in results if r["is_mcq"]]
    free = [r for r in results if not r["is_mcq"]]
    def acc(s): return round(sum(r["correct"] for r in s) / len(s) * 100, 2) if s else 0.0
    return {
        "total":     len(results),
        "mcq_acc":   f"{acc(mcq)}%  ({sum(r['correct'] for r in mcq)}/{len(mcq)})",
        "free_acc":  f"{acc(free)}%  ({sum(r['correct'] for r in free)}/{len(free)})",
        "overall":   f"{acc(results)}%  ({sum(r['correct'] for r in results)}/{len(results)})",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model 0: Optimized starter baseline")
    parser.add_argument("--data",        default=DATA_PATH)
    parser.add_argument("--gpu",         default="0")
    parser.add_argument("--out_dir",     default="results")
    parser.add_argument("--max_tokens",  type=int, default=MAX_TOKENS)

    # ── Subset / apples-to-apples flags ──
    parser.add_argument("--subset_ids",  default=None,
                        help="Path to a JSON list of IDs to evaluate on. "
                             "Generate once with --make_subset and reuse across all models.")
    parser.add_argument("--make_subset", action="store_true",
                        help="Sample a balanced subset, save IDs, then exit.")
    parser.add_argument("--subset_size", type=int, default=50,
                        help="Number of questions in the subset (default: 50).")
    parser.add_argument("--subset_seed", type=int, default=42)

    # ── vLLM tuning ──
    parser.add_argument("--gpu_mem_util",   type=float, default=0.90,
                        help="Fraction of GPU VRAM for vLLM (default: 0.90)")
    parser.add_argument("--max_model_len",  type=int,   default=16384)
    parser.add_argument("--quantization",   default=None,
                        help="e.g. 'bitsandbytes' for INT8; omit for BF16")
    parser.add_argument("--no_vllm",        action="store_true",
                        help="Skip vLLM and use Transformers+INT4 directly (useful if vLLM has version conflicts)")
    parser.add_argument("--hf_batch_size",  type=int, default=4,
                        help="Batch size for Transformers fallback (default: 4)")
    args = parser.parse_args()

    # ── --make_subset mode: create shared ID list and exit ──────────────────
    if args.make_subset:
        if not args.subset_ids:
            args.subset_ids = "data/eval_subset.json"
        make_and_save_subset(args.data, args.subset_size, args.subset_ids, args.subset_seed)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ── Load data ────────────────────────────────────────────────────────────
    subset_ids = None
    if args.subset_ids:
        if not Path(args.subset_ids).exists():
            print(f"ERROR: subset file not found: {args.subset_ids}")
            print("Run with --make_subset first to create it.")
            sys.exit(1)
        subset_ids = json.load(open(args.subset_ids))
        print(f"Using fixed subset: {args.subset_ids} ({len(subset_ids)} IDs)")

    data = load_jsonl(args.data, ids=subset_ids)
    print(f"Dataset: {args.data} | Questions to run: {len(data)}")

    # ── Output path ──────────────────────────────────────────────────────────
    out_dir  = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "model0_results.jsonl"

    # ── Resume: skip already-done IDs ────────────────────────────────────────
    done = set()
    if out_path.exists():
        for line in open(out_path):
            done.add(json.loads(line)["id"])
    remaining = [d for d in data if d.get("id") not in done]
    print(f"Resume: {len(done)} done, {len(remaining)} remaining")
    if not remaining:
        print("Nothing to do. Summary:", summarize_results(out_path))
        return

    # ── Load tokenizer ────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Build all prompts ─────────────────────────────────────────────────────
    prompts = []
    for item in remaining:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(make_chat_prompt(tokenizer, system, user))

    # ── Try vLLM, fall back to Transformers if incompatible ───────────────────
    use_vllm = not args.no_vllm
    responses = None

    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
            print("Loading model via vLLM...")
            llm_kwargs = dict(
                model=MODEL_ID,
                gpu_memory_utilization=args.gpu_mem_util,
                max_model_len=args.max_model_len,
                trust_remote_code=True,
            )
            if args.quantization:
                llm_kwargs["quantization"] = args.quantization
                llm_kwargs["load_format"]  = args.quantization
            llm = LLM(**llm_kwargs)
            sampling_params = SamplingParams(max_tokens=args.max_tokens, **SAMPLING_PARAMS)
            print(f"Generating {len(prompts)} responses (vLLM)...")
            outputs   = llm.generate(prompts, sampling_params=sampling_params)
            responses = [out.outputs[0].text.strip() for out in outputs]
            print("vLLM generation complete.")
        except Exception as e:
            print(f"vLLM failed ({e})\nFalling back to Transformers...")
            use_vllm = False

    if not use_vllm:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        print("Loading model via Transformers (INT4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
        llm.eval()
        print("Model ready. Generating in batches...")

        responses = []
        batch_size = args.hf_batch_size
        for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[start:start + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_model_len,
            ).to(llm.device)
            with torch.no_grad():
                output_ids = llm.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=SAMPLING_PARAMS["temperature"],
                    top_p=SAMPLING_PARAMS["top_p"],
                    top_k=SAMPLING_PARAMS["top_k"],
                    repetition_penalty=SAMPLING_PARAMS["repetition_penalty"],
                    do_sample=True,
                )
            prompt_len = inputs["input_ids"].shape[1]
            for out in output_ids:
                new_tokens = out[prompt_len:]
                responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    # ── Load judger ───────────────────────────────────────────────────────────
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    # ── Score and save ────────────────────────────────────────────────────────
    with open(out_path, "a") as f:
        for item, response in tqdm(zip(remaining, responses), total=len(remaining), desc="Scoring & saving"):
            rec = {"id": item.get("id"), "is_mcq": bool(item.get("options")), "response": response}
            if "answer" in item:
                rec["gold"]    = item["answer"]
                rec["correct"] = score_response(item, response, judger)
            f.write(json.dumps(rec) + "\n")
            f.flush()

    print("\nSaved to:", out_path)
    summary = summarize_results(out_path)
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:<12}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
