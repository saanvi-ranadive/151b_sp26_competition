"""
Model 6: Inference with Fine-Tuned LoRA Adapter  (Contribution C)

Run this after model5_qlora_finetune.py has saved an adapter. Loads the base
Qwen model plus the LoRA adapter and evaluates on the eval subset. Isolates the
benefit of fine-tuning from prompt engineering / self-consistency.

Usage:
  python model6_lora_inference.py \
      --adapter adapters/qwen3_4b_math_lora \
      --subset_ids data/eval_subset.json \
      --no_vllm
"""

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

from common_utils import (
    MODEL_ID,
    format_options,
    generate_batch,
    load_judger,
    load_jsonl,
    make_chat_prompt,
    score_response,
    starter_build_prompt,
    summarize_results,
)

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
DATA_PATH = "data/public.jsonl"
MAX_TOKENS = 768        # raised to give fine-tuned model room to show its work
DEFAULT_BATCH = 8

# Greedy decoding to isolate fine-tuning contribution cleanly
SAMPLING_PARAMS = dict(
    temperature=1.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.05,
    do_sample=False,
)

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_base_plus_adapter(adapter_path: str, load_in_4bit: bool = False):
    """Load base model + LoRA adapter, return (tokenizer, model)."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for batched inference

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    kwargs: dict = dict(trust_remote_code=True, device_map="auto")
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else torch.float16

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model


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
    p = argparse.ArgumentParser(description="Model 6: LoRA adapter inference (Contribution C)")
    p.add_argument("--adapter", required=True,
                   help="Path to saved LoRA adapter directory (from model5_qlora_finetune.py).")
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

    tokenizer, llm = load_base_plus_adapter(args.adapter, args.load_in_4bit)
    judger = load_judger()

    out_path = Path(args.out_dir) / "model6_lora_inference_results.jsonl"
    done = load_done_ids(out_path)
    remaining = [d for d in data if d.get("id") not in done]
    print(f"LoRA inference | todo={len(remaining)}  (skipping {len(done)} already done)")

    with open(out_path, "a") as fh:
        for batch_start in tqdm(range(0, len(remaining), args.batch_size),
                                desc="lora_inference", unit="batch"):
            batch = remaining[batch_start : batch_start + args.batch_size]
            prompts = []
            for item in batch:
                system, user = starter_build_prompt(item["question"], item.get("options"))
                prompts.append(make_chat_prompt(tokenizer, system, user))

            responses = generate_batch(
                llm, tokenizer, prompts,
                max_new_tokens=args.max_tokens, **SAMPLING_PARAMS,
            )

            for item, response in zip(batch, responses):
                rec = {
                    "id": item.get("id"),
                    "model": "lora_finetuned",
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
