#!/usr/bin/env python3
"""
Unified Inference Harness for CSE 151B Math Reasoning Competition.
Combines Prompt Engineering, LoRA Adapters, and Self-Consistency Decoding.
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
# Prompt Configurations (Consolidated Strategies)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "structured_cot": {
        "sys_math": (
            "You are an expert mathematician. Solve the problem step by step:\n"
            "1. Identify the problem type.\n"
            "2. State relevant theorems or formulas.\n"
            "3. Compute carefully, showing all steps.\n"
            "4. Verify your result against boundary conditions.\n"
            "5. Place the final answer inside \\boxed{}."
        ),
        "sys_mcq": (
            "You are an expert mathematician. Solve the problem completely first, "
            "then match your result to the options. End your response with ONLY "
            "the chosen letter inside \\boxed{}, e.g., \\boxed{C}."
        )
    },
    "verification": {
        "sys_math": (
            "You are an expert mathematician. Solve the problem step by step, then independently "
            "verify your answer by substitution or recalculation. If verification fails, revise. "
            "Put the final verified answer inside \\boxed{}."
        ),
        "sys_mcq": (
            "You are an expert mathematician. Solve independently, verify your answer matches "
            "the correct option, and rule out the others. End with ONLY the option letter inside \\boxed{}."
        )
    },
    "vanilla": {
        "sys_math": "You are an expert mathematician. Solve the problem and box your answer.",
        "sys_mcq": "You are an expert mathematician. Choose the correct option and box the letter."
    }
}

# ---------------------------------------------------------------------------
# Core Logic & Normalization
# ---------------------------------------------------------------------------
def build_prompt(item: dict, template_name: str) -> tuple[str, str]:
    cfg = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["vanilla"])
    if item.get("options"):
        user = f'{item["question"]}\n\nOptions:\n{format_options(item["options"])}'
        return cfg["sys_mcq"], user
    return cfg["sys_math"], item["question"]

def normalize_vote(item: dict, response: str) -> str:
    if item.get("options"):
        letter = extract_letter(response)
        return letter.upper() if letter else ""
    boxed = extract_boxed(response)
    if boxed:
        return re.sub(r"\s+", "", boxed).strip().lower()
    return re.sub(r"\s+", "", response.strip()[-200:]).strip().lower()

def majority_vote(item: dict, responses: list[str]) -> tuple[str, str, dict, list[str]]:
    votes = [normalize_vote(item, r) for r in responses]
    counts = collections.Counter(v for v in votes if v)

    if not counts:
        return "", responses[0] if responses else "", {}, votes

    # Tie-break by average character length of supporting paths (favors detailed reasoning)
    def tie_breaker(pair):
        vote, count = pair
        supporters = [r for r, v in zip(responses, votes) if v == vote]
        avg_len = sum(len(r) for r in supporters) / max(len(supporters), 1)
        return (count, avg_len)

    best_vote = max(counts.items(), key=tie_breaker)[0]
    best_response = max([r for r, v in zip(responses, votes) if v == best_vote], key=len)
    return best_vote, best_response, dict(counts), votes

# ---------------------------------------------------------------------------
# Unified Model Loader
# ---------------------------------------------------------------------------
def load_engine(adapter_path: str | None, load_in_4bit: bool):
    if adapter_path:
        print(f"Loading base model [{MODEL_ID}] + LoRA adapter [{adapter_path}]...")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        
        kwargs = {"trust_remote_code": True, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        return tokenizer, model
    else:
        print(f"Loading zero-shot base model [{MODEL_ID}]...")
        return load_model_and_tokenizer(MODEL_ID, load_in_4bit=load_in_4bit)

# ---------------------------------------------------------------------------
# Executable Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Unified Math Reasoning Pipeline Engine")
    p.add_argument("--data", default="data/public.jsonl")
    p.add_argument("--subset_ids", default=None, help="JSON file with filtered test-set sample IDs.")
    p.add_argument("--adapter", default=None, help="Path to trained LoRA checkpoint directory.")
    p.add_argument("--prompt_strategy", default="structured_cot", choices=list(PROMPT_TEMPLATES))
    p.add_argument("--num_samples", type=int, default=1, help="Set >1 to trigger Self-Consistency ensembling.")
    p.add_argument("--inner_batch", type=int, default=8, help="Sub-batch size for sampling chunks to avoid VRAM OOM.")
    p.add_argument("--max_tokens", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for single-sample inference passes.")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--gpu", default="0")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output naming scheme maps strategy complexity automatically
    adapter_tag = "lora" if args.adapter else "base"
    sc_tag = f"scN{args.num_samples}" if args.num_samples > 1 else "greedy"
    out_path = out_dir / f"run_{adapter_tag}_{args.prompt_strategy}_{sc_tag}.jsonl"

    # Data ingestion
    data = load_jsonl(args.data)
    if args.subset_ids:
        with open(args.subset_ids) as f:
            raw = json.load(f)
        subset = {x["id"] if isinstance(x, dict) else x for x in raw}
        data = [d for d in data if d.get("id") in subset]

    tokenizer, llm = load_engine(args.adapter, args.load_in_4bit)
    judger = load_judger()

    # Self-Consistency sets temperature-driven sampling; Single-sample falls back to stable Greedy decoding
    if args.num_samples > 1:
        sampling_params = dict(temperature=0.7, top_p=0.95, top_k=50, repetition_penalty=1.05, do_sample=True)
    else:
        sampling_params = dict(temperature=1.0, top_p=1.0, top_k=1, repetition_penalty=1.05, do_sample=False)

    print(f"Starting execution loop -> Saving targets to: {out_path}")
    
    with open(out_path, "a") as fh:
        for item in tqdm(data, desc="Evaluating"):
            system, user = build_prompt(item, args.prompt_strategy)
            prompt = make_chat_prompt(tokenizer, system, user)

            responses = []
            if args.num_samples > 1:
                # Sub-batching sampling rounds to protect hardware buffers
                for start in range(0, args.num_samples, args.inner_batch):
                    chunk = min(args.inner_batch, args.num_samples - start)
                    responses.extend(generate_batch(llm, tokenizer, [prompt] * chunk, max_new_tokens=args.max_tokens, **sampling_params))
                best_vote, final_response, counts, votes = majority_vote(item, responses)
            else:
                final_response = generate_batch(llm, tokenizer, [prompt], max_new_tokens=args.max_tokens, **sampling_params)[0]
                best_vote = normalize_vote(item, final_response)
                counts, votes = {}, [best_vote]

            rec = {
                "id": item.get("id"),
                "is_mcq": bool(item.get("options")),
                "prediction": best_vote,
                "response": final_response,
                "metadata": {"strategy": args.prompt_strategy, "samples": args.num_samples, "vote_spread": counts}
            }
            if "answer" in item:
                rec["gold"] = item["answer"]
                rec["correct"] = score_response(item, final_response, judger)

            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    print("\nRun Complete. Summary Results:")
    print(summarize_results(out_path))

if __name__ == "__main__":
    main()