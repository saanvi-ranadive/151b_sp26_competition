"""
Model 2: Self-Consistency Decoding for CSE 151B Math Reasoning Competition

Contribution B: Majority voting over N independent samples per question.
  - Keeps the same zero-shot starter prompt as Model 0 (isolates self-consistency effect)
  - Generates N samples per question using vLLM batched inference
  - Extracts boxed answer from each sample, majority-votes for final answer
  - Supports N=8 and N=16 as described in the milestone report
  - --subset_ids for apples-to-apples comparison with Model 0 and Model 1

Key improvements over original:
  - vLLM batches ALL N samples for ALL questions in one call (not N sequential calls per item)
  - Transformers INT4 fallback with left-padding (matches Model 0/1 pattern)
  - MAX_TOKENS raised from 512 to 1024 (thinking model needs room for <think> tokens)
  - Normalized answer key handles multi-part answers and MCQ uniformly
  - --subset_ids flag for apples-to-apples comparison
  - Removed common_utils dependency (self-contained)

Usage:
  # Quick smoke-test (5 questions, N=8)
  python model2_self_consistency.py --limit 5 --num_samples 8 --no_vllm

  # Run on shared eval subset
  python model2_self_consistency.py --subset_ids data/eval_subset.json --num_samples 8
  python model2_self_consistency.py --subset_ids data/eval_subset.json --num_samples 16

  # Full dataset
  python model2_self_consistency.py --data data/public.jsonl --num_samples 8
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH  = "data/public.jsonl"
MAX_TOKENS = 1024  # 512 truncates chain-of-thought; raise further if answers are cut off

# Slightly higher temperature than baseline encourages diverse samples for voting
SAMPLING_PARAMS = dict(temperature=0.7, top_p=0.95, top_k=20, repetition_penalty=1.0)


# ── Prompt (identical to Model 0 starter — isolates self-consistency effect) ──

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


def build_prompt(item: dict) -> tuple[str, str]:
    """Returns (system, user) using the same starter prompt as Model 0."""
    if item.get("options"):
        user = f'{item["question"]}\n\nOptions:\n{format_options(item["options"])}'
        return SYSTEM_PROMPT_MCQ, user
    return SYSTEM_PROMPT_MATH, item["question"]


def make_chat_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Answer extraction & voting ────────────────────────────────────────────────

def extract_boxed(text: str) -> str:
    """Extract the content of the last \\boxed{} in text, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len(r"\boxed{")
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            break
    return results[-1].strip() if results else ""


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def vote_key(item: dict, response: str) -> str:
    """
    Normalised key used for majority voting.
    MCQ  → single uppercase letter (or empty string if not found)
    Free → boxed answer string, lowercased and stripped for robustness
    """
    if item.get("options"):
        return extract_letter(response)
    boxed = extract_boxed(response)
    # Normalize: lowercase, collapse whitespace so e.g. "2, 3" == "2,3" don't split votes
    if boxed:
        return re.sub(r"\s+", "", boxed.lower())
    # Fallback: last 200 chars of response (avoids empty-key collisions)
    return response.strip()[-200:]


def majority_vote(items_and_responses: list[tuple[dict, list[str]]]) -> list[dict]:
    """
    Given a list of (item, [response_1, ..., response_N]),
    return a list of vote-result dicts — one per item.
    """
    results = []
    for item, responses in items_and_responses:
        keys        = [vote_key(item, r) for r in responses]
        vote_counts = Counter(keys)
        winning_key, winning_count = vote_counts.most_common(1)[0]
        # Use first response that produced the winning answer as the canonical response
        winning_response = responses[keys.index(winning_key)]
        results.append({
            "votes":            keys,
            "vote_counts":      dict(vote_counts),
            "winning_answer":   winning_key,
            "winning_count":    winning_count,
            "winning_response": winning_response,
            "all_responses":    responses,
        })
    return results


# ── Scoring ───────────────────────────────────────────────────────────────────

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
        return {"note": "no scored results yet"}
    mcq  = [r for r in results if r["is_mcq"]]
    free = [r for r in results if not r["is_mcq"]]
    def acc(s): return round(sum(r["correct"] for r in s) / len(s) * 100, 2) if s else 0.0
    return {
        "total":    len(results),
        "mcq_acc":  f"{acc(mcq)}%  ({sum(r['correct'] for r in mcq)}/{len(mcq)})",
        "free_acc": f"{acc(free)}%  ({sum(r['correct'] for r in free)}/{len(free)})",
        "overall":  f"{acc(results)}%  ({sum(r['correct'] for r in results)}/{len(results)})",
    }


# ── Generation helpers ────────────────────────────────────────────────────────

def generate_vllm(llm, prompts_repeated: list[str], max_tokens: int) -> list[str]:
    """
    Submit all N*Q prompts to vLLM in one call.
    vLLM's continuous batching handles this optimally.
    """
    from vllm import SamplingParams
    sampling_params = SamplingParams(max_tokens=max_tokens, **SAMPLING_PARAMS)
    outputs = llm.generate(prompts_repeated, sampling_params=sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]


def generate_hf(llm, tokenizer, prompts_repeated: list[str],
                max_tokens: int, batch_size: int, max_model_len: int) -> list[str]:
    """Batched HF Transformers generation with left-padding."""
    import torch
    responses = []
    for start in tqdm(range(0, len(prompts_repeated), batch_size), desc="HF batches"):
        batch = prompts_repeated[start:start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_model_len,
        ).to(llm.device)
        with torch.no_grad():
            out_ids = llm.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=SAMPLING_PARAMS["temperature"],
                top_p=SAMPLING_PARAMS["top_p"],
                top_k=SAMPLING_PARAMS["top_k"],
                repetition_penalty=SAMPLING_PARAMS["repetition_penalty"],
                do_sample=True,
            )
        plen = inputs["input_ids"].shape[1]
        for o in out_ids:
            responses.append(tokenizer.decode(o[plen:], skip_special_tokens=True).strip())
    return responses


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model 2: Self-Consistency Decoding")
    parser.add_argument("--data",        default=DATA_PATH)
    parser.add_argument("--limit",       type=int, default=None,
                        help="Cap questions (smoke-test)")
    parser.add_argument("--gpu",         default="0")
    parser.add_argument("--out_dir",     default="results")
    parser.add_argument("--max_tokens",  type=int, default=MAX_TOKENS)
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of independent samples per question (8 or 16)")

    # Apples-to-apples
    parser.add_argument("--subset_ids",  default=None,
                        help="Path to eval_subset.json (created by model0 --make_subset)")

    # vLLM / HF fallback
    parser.add_argument("--no_vllm",       action="store_true",
                        help="Skip vLLM, use Transformers+INT4")
    parser.add_argument("--hf_batch_size", type=int, default=4,
                        help="Batch size for Transformers fallback (across N*Q prompts)")
    parser.add_argument("--gpu_mem_util",  type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int,   default=16384)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    Path(args.out_dir).mkdir(exist_ok=True)
    out_path = Path(args.out_dir) / f"model2_self_consistency_N{args.num_samples}_results.jsonl"

    # Load data
    subset_ids = None
    if args.subset_ids:
        if not Path(args.subset_ids).exists():
            print(f"ERROR: subset file not found: {args.subset_ids}")
            sys.exit(1)
        subset_ids = json.load(open(args.subset_ids))
        print(f"Using fixed subset: {args.subset_ids} ({len(subset_ids)} IDs)")

    data = load_jsonl(args.data, limit=args.limit, ids=subset_ids)
    print(f"Dataset: {args.data} | Questions: {len(data)} | N={args.num_samples}")

    # Resume
    done = set()
    if out_path.exists():
        for line in open(out_path):
            done.add(json.loads(line)["id"])
    remaining = [d for d in data if d.get("id") not in done]
    print(f"Resume: {len(done)} done, {len(remaining)} remaining")
    if not remaining:
        print("Nothing to do. Summary:", summarize_results(out_path))
        return

    # Load tokenizer
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched decoder-only generation

    # Build prompts — each question repeated N times so one generation call covers all samples
    # Layout: [q0_s0, q0_s1, ..., q0_sN, q1_s0, ..., qK_sN]
    base_prompts = [make_chat_prompt(tokenizer, *build_prompt(item)) for item in remaining]
    prompts_repeated = [p for p in base_prompts for _ in range(args.num_samples)]
    total = len(prompts_repeated)
    print(f"Total prompts to generate: {len(remaining)} questions × {args.num_samples} samples = {total}")

    # Load model
    llm = None
    use_vllm = False
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
            use_vllm = True
            print("vLLM model ready.")
        except Exception as e:
            print(f"vLLM failed ({e})\nFalling back to Transformers...")

    if not use_vllm:
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

    # Generate all samples in one batched call
    print("Generating samples...")
    if use_vllm:
        all_responses = generate_vllm(llm, prompts_repeated, args.max_tokens)
    else:
        all_responses = generate_hf(
            llm, tokenizer, prompts_repeated,
            args.max_tokens, args.hf_batch_size, args.max_model_len,
        )

    # Reshape: [Q * N] → [(item, [r0..rN]), ...]
    grouped = []
    for i, item in enumerate(remaining):
        start = i * args.num_samples
        grouped.append((item, all_responses[start:start + args.num_samples]))

    # Vote and score
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    vote_results = majority_vote(grouped)

    with open(out_path, "a") as f:
        for item, vr in tqdm(zip(remaining, vote_results), total=len(remaining), desc="Scoring & saving"):
            rec = {
                "id":             item.get("id"),
                "model":          f"self_consistency_N{args.num_samples}",
                "num_samples":    args.num_samples,
                "is_mcq":         bool(item.get("options")),
                "votes":          vr["votes"],
                "vote_counts":    vr["vote_counts"],
                "winning_answer": vr["winning_answer"],
                "winning_count":  vr["winning_count"],
                "response":       vr["winning_response"],
                "all_responses":  vr["all_responses"],
            }
            if "answer" in item:
                rec["gold"]    = item["answer"]
                rec["correct"] = score_response(item, vr["winning_response"], judger)
            f.write(json.dumps(rec) + "\n")
            f.flush()

    print("\nSaved to:", out_path)
    summary = summarize_results(out_path)
    print("\n" + "=" * 55)
    print(f"SELF-CONSISTENCY RESULTS  (N={args.num_samples})")
    print("=" * 55)
    for k, v in summary.items():
        print(f"  {k:<12}: {v}")
    print("=" * 55)


if __name__ == "__main__":
    main()
