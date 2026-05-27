"""Shared utilities for CSE 151B math reasoning experiments."""
import json, re, sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

STARTER_SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
STARTER_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

def load_jsonl(path: str, limit: Optional[int] = None) -> list[dict]:
    data = [json.loads(line) for line in open(path)]
    return data[:limit] if limit else data

def format_options(options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    return "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))

def starter_build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        return STARTER_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{format_options(options)}"
    return STARTER_SYSTEM_PROMPT_MATH, question

def make_chat_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )

def load_model_and_tokenizer(model_id: str = MODEL_ID, load_in_4bit: bool = False, dtype: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(trust_remote_code=True, device_map="auto")
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, dtype)
    llm = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    llm.eval()
    return tokenizer, llm

def generate_batch(llm, tokenizer, prompts: list[str], max_new_tokens: int, temperature: float, top_p: float, top_k: int, do_sample: bool, repetition_penalty: float = 1.0, max_length: int = 8192) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(llm.device)
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    responses = []
    prompt_len = inputs["input_ids"].shape[1]
    for out in output_ids:
        new_tokens = out[prompt_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return responses

def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1].strip()
    return ""

def extract_letter(text: str) -> str:
    boxed = extract_boxed(text)
    m = re.search(r"[A-Za-z]", boxed)
    if m:
        return m.group(0).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def score_response(item: dict, response: str, judger=None) -> bool | None:
    if "answer" not in item:
        return None
    if item.get("options"):
        return extract_letter(response) == str(item["answer"]).strip().upper()
    if judger is None:
        return None
    gold = item["answer"]
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return bool(judger.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list)))
    except Exception:
        return False

def load_judger():
    sys.path.insert(0, ".")
    try:
        from judger import Judger
        return Judger(strict_extract=False)
    except Exception as e:
        print(f"Warning: could not load judger.py ({e}). Will save predictions without free-form scoring.")
        return None

def summarize_results(path: Path) -> dict:
    results = [json.loads(l) for l in open(path)] if path.exists() else []
    scored = [r for r in results if r.get("correct") is not None]
    mcq = [r for r in scored if r.get("is_mcq")]
    free = [r for r in scored if not r.get("is_mcq")]
    def acc(xs):
        return 100 * sum(bool(x.get("correct")) for x in xs) / len(xs) if xs else 0.0
    return {"n_total": len(results), "n_scored": len(scored), "mcq_acc": acc(mcq), "free_acc": acc(free), "total_acc": acc(scored)}
