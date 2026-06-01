"""
Model 5: QLoRA Fine-Tuning for CSE 151B Math Reasoning Competition

Contribution C: Supervised fine-tuning of Qwen3-4B-Thinking-2507 with LoRA adapters
on MetaMathQA, NuminaMath, or any compatible competition JSONL dataset.
Saves only the adapter weights (not a full model copy).

Expected training JSONL format (one example per line):
  {"question": "...", "answer": "7", "solution": "..."}          # free-response
  {"question": "...", "options": [...], "answer": "C", "solution": "..."}  # MCQ

If `solution` is missing, a minimal target using the answer field is constructed.

Key improvements over the original:
  - gradient_checkpointing enabled to cut VRAM ~30-40% at the cost of ~20% speed.
  - fp16 auto-detected as fallback when bf16 is not supported by the hardware.
  - Target modules expanded to include gate_proj/up_proj/down_proj (MLP layers),
    which consistently improves math fine-tuning results on Qwen-family models.
  - LoRA rank raised to r=32 for more expressive adapters on a hard reasoning task.
  - dataloader_num_workers=2 for faster data feeding during training.
  - Explicit padding side set to 'right' to avoid left-pad issues with causal LM.

Usage smoke test (5 steps, no 4-bit):
  python model5_qlora_finetune.py --train data/sft_train.jsonl --limit 20 --max_steps 5

Full run on DSMLP A100 (no 4-bit needed, bf16 native):
  python model5_qlora_finetune.py --train data/sft_train.jsonl --epochs 1

On older GPUs without bf16 support:
  python model5_qlora_finetune.py --train data/sft_train.jsonl --load_in_4bit --epochs 1
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from common_utils import (
    MODEL_ID,
    STARTER_SYSTEM_PROMPT_MATH,
    STARTER_SYSTEM_PROMPT_MCQ,
    format_options,
    make_chat_prompt,
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    with open(path) as fh:
        rows = [json.loads(line) for line in fh]
    return rows[:limit] if limit else rows


def target_text(ex: dict) -> str:
    """Build the assistant-side target string from a training example."""
    if ex.get("solution"):
        sol = ex["solution"].strip()
        # Ensure a boxed answer is present so the model learns the format
        if "\\boxed" not in sol and ex.get("answer") is not None:
            sol += f"\nFinal answer: \\boxed{{{ex['answer']}}}"
        return sol
    return f"Final answer: \\boxed{{{ex.get('answer', '')}}}"


def build_training_text(tokenizer, ex: dict) -> str:
    """Concatenate chat prompt + target + EOS into one training string."""
    if ex.get("options"):
        system = STARTER_SYSTEM_PROMPT_MCQ
        user = f'{ex["question"]}\n\nOptions:\n{format_options(ex["options"])}'
    else:
        system = STARTER_SYSTEM_PROMPT_MATH
        user = ex["question"]
    prompt = make_chat_prompt(tokenizer, system, user)
    return prompt + target_text(ex) + tokenizer.eos_token


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Model 5: QLoRA/LoRA supervised fine-tuning (Contribution C)")
    p.add_argument("--train", required=True, help="Training JSONL file")
    p.add_argument("--output_dir", default="adapters/qwen3_4b_math_lora")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap training examples (useful for smoke tests).")
    p.add_argument("--gpu", default="0")
    p.add_argument("--max_length", type=int, default=2048,
                   help="Max token length per training example. 2048 fits most math problems.")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Override epochs with a fixed step count (useful for smoke tests).")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="Effective batch = per_device * accumulation. Default=8 → batch of 8.")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=32,
                   help="LoRA rank. Higher = more parameters, better capacity for hard tasks.")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Use 4-bit quantisation. Only needed on GPUs with limited VRAM.")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Auto-detect bf16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16 and torch.cuda.is_available()
    print(f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for causal LM training

    # Base model
    kwargs: dict = dict(trust_remote_code=True, device_map="auto")
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.config.use_cache = False   # required for gradient checkpointing

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    # LoRA config — includes MLP projection layers for better math fine-tuning
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,   # standard heuristic: alpha = 2 * r
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP — important for math
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    rows = read_jsonl(args.train, args.limit)
    print(f"Training on {len(rows)} examples from {args.train}")
    texts = [build_training_text(tokenizer, ex) for ex in rows]
    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,     # saves ~30-40% VRAM
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=1.0,
        dataloader_num_workers=2,        # faster data feeding
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
