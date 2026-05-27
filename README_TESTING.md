# CSE 151B Math Reasoning Models — Testing Instructions

These scripts are aligned to the project contributions in the milestone report:

- **Control / starter baseline**: `model0_starter_baseline.py`
- **Contribution A — Prompt engineering**: `model3_prompt_engineering.py`, `model4_fewshot_prompting.py`
- **Contribution B — Self-consistency decoding**: `model2_self_consistency.py`
- **Contribution C — QLoRA fine-tuning**: `model5_qlora_finetune.py`, then `model6_lora_inference.py`

The baseline intentionally follows the starter-code logic: starter prompt, chat template, one generated response, boxed answer extraction, MCQ exact letter scoring, and free-response `judger.auto_judge` when available.

## 0. Setup

Put these files in the same folder as `judger.py`, and make sure your data is at:

```bash
data/public.jsonl
```

Install likely dependencies:

```bash
pip install torch transformers accelerate bitsandbytes peft datasets tqdm
```

If your GPU gave the Blackwell `sm_120` / `no kernel image` error, do **not** pass `--load_in_4bit`. The scripts default to non-4-bit loading for that reason. Use `--load_in_4bit` only on compatible machines such as A100/DSMLP if the installed PyTorch + bitsandbytes supports it.

## 1. Smoke test the starter baseline

```bash
python model0_starter_baseline.py --data data/public.jsonl --limit 2 --max_tokens 256 --out_dir results_smoke
```

Expected: creates `results_smoke/model0_starter_baseline_results.jsonl` and prints a summary.

## 2. Run the full starter baseline/control

```bash
python model0_starter_baseline.py --data data/public.jsonl --max_tokens 512 --out_dir results
```

This is the control to compare against all improvements.

## 3. Contribution A: structured prompt engineering

Run all prompt variants:

```bash
python model3_prompt_engineering.py --data data/public.jsonl --variant all --max_tokens 512 --out_dir results
```

Or run one variant:

```bash
python model3_prompt_engineering.py --data data/public.jsonl --variant structured_cot --max_tokens 512 --out_dir results
```

Use this to evaluate whether structured decomposition, verification, or stricter answer formatting improves over the baseline.

## 4. Contribution A: few-shot prompting

```bash
python model4_fewshot_prompting.py --data data/public.jsonl --max_tokens 512 --out_dir results
```

This tests the report claim that few-shot examples help the model learn competition-style answer formatting.

## 5. Contribution B: self-consistency decoding

Smoke test with small N:

```bash
python model2_self_consistency.py --data data/public.jsonl --limit 2 --num_samples 4 --max_tokens 256 --out_dir results_smoke
```

Report-style runs:

```bash
python model2_self_consistency.py --data data/public.jsonl --num_samples 8 --max_tokens 512 --out_dir results
python model2_self_consistency.py --data data/public.jsonl --num_samples 16 --max_tokens 512 --out_dir results
```

This tests majority voting over multiple sampled outputs using the starter prompt.

## 6. Contribution C: QLoRA fine-tuning

Prepare a supervised fine-tuning JSONL such as:

```json
{"question":"Solve 3x - 7 = 14. x = [ANS]","answer":"7","solution":"Add 7 to both sides: 3x=21. Divide by 3: x=7. Final answer: \\boxed{7}"}
```

Train a tiny smoke-test adapter:

```bash
python model5_qlora_finetune.py --train data/sft_train.jsonl --limit 20 --max_steps 5 --output_dir adapters/smoke_lora
```

Train a real adapter, on compatible compute:

```bash
python model5_qlora_finetune.py --train data/sft_train.jsonl --epochs 1 --output_dir adapters/qwen3_4b_math_lora
```

On compatible A100-style machines you may add:

```bash
--load_in_4bit
```

Then evaluate the adapter:

```bash
python model6_lora_inference.py --adapter adapters/qwen3_4b_math_lora --data data/public.jsonl --max_tokens 512 --out_dir results
```

## 7. Compare results

Each script prints a summary and saves JSONL files in `results/`. Compare `total_acc`, `mcq_acc`, and `free_acc` across:

- `model0_starter_baseline_results.jsonl`
- `model3_prompt_structured_cot_results.jsonl`
- `model4_fewshot_prompting_results.jsonl`
- `model2_self_consistency_N8_results.jsonl`
- `model2_self_consistency_N16_results.jsonl`
- `model6_lora_inference_results.jsonl`

Use the first successful full run as your baseline, then report the delta for each contribution.
