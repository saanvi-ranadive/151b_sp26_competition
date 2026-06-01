# CSE 151B Math Reasoning Competition Final Submission

This repository contains the final reproducible inference pipeline for the CSE 151B Kaggle Math Reasoning Competition. The pipeline uses the competition-required base model:

`Qwen/Qwen3-4B-Thinking-2507`

The code supports prompt engineering, optional LoRA/QLoRA adapters, and optional self-consistency voting. The final entry point is `run_inference()`, which loads the model, runs inference, applies answer extraction/post-processing, and writes the final Kaggle submission CSV.

---

## Team Members

- Saanvi Ranadive
- Ritvik Chand

---

## Final Method Summary

Our final pipeline uses model-intrinsic inference methods only. It does not use external APIs, calculators, code interpreters, retrieval systems, or tool-augmented generation at test time.

Implemented strategies include:

1. **Structured chain-of-thought prompting**
   - Free-response questions ask the model to reason step by step and place the final answer in `\boxed{}`.
   - Multiple-choice questions ask the model to solve first, then output only the option letter in `\boxed{}`.

2. **Optional verification prompting**
   - The `verification` prompt strategy asks the model to solve, independently check the result, and revise if needed.

3. **Optional self-consistency decoding**
   - When `--num_samples > 1`, the script generates multiple reasoning paths with sampling and majority-votes over normalized final answers.
   - Voting is deterministic: highest vote count wins; ties are broken by earliest occurrence.

4. **Optional LoRA/QLoRA adapter loading**
   - If `--adapter` is provided, the script loads the required Qwen base model and applies the provided adapter.
   - The adapter path can be a local directory or a HuggingFace Hub repository ID.

5. **Answer normalization and CSV generation**
   - Multiple-choice answers are normalized to option letters.
   - Free-response answers are extracted from `\boxed{}` when available.
   - The final output is a Kaggle-style CSV.

---

## Repository Structure

Recommended final repo structure:

```text
.
├── README.md
├── requirements.txt
├── constraints.txt
├── run_inference.py
├── common_utils.py
├── data/
│   └── private.jsonl        
├── results/
│   └── debug_generations.jsonl         # generated after inference
└── submission.csv                      # generated after inference
```

Required files:

- `README.md`: This file.
- `run_inference.py`: Final reproducible inference entry point.
- `common_utils.py`: Shared model loading, prompting, extraction, generation, and evaluation helpers.
- `requirements.txt`: Python dependencies needed to run the pipeline.
- `constraints.txt`: Python dependency package versions needed to run the pipeline.

---

## Hardware and Runtime

Final run hardware:

- GPU type used: `NVIDIA H200`
- Number of GPUs used: 1
- Approximate total inference/generation time: 8 hours

---

## Installation

Create and activate an environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

Install torch separately first:
`uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121`

Then install the rest:
`uv pip install -r requirements.txt -c constraints.txt`

---

## Model Weights Setup

### Base model

The final submission uses the competition-required base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

No fine-tuned LoRA/QLoRA adapter was used for the final submission.
The model weights are downloaded automatically from Hugging Face when run_inference() is executed for the first time.

### Downloading the Model

Log in to Hugging Face if required:
huggingface-cli login
Then run:

```
from run_inference import run_inference

run_inference()
```

or

```
python run_inference.py \
  --data data/private.jsonl \
  --output_csv submission.csv \
  --debug_jsonl results/debug_generations_1024.jsonl \
  --max_tokens 1024
```

### Local Model Cache Location

By default, Hugging Face stores downloaded model weights in:
~/.cache/huggingface/hub/
No manual placement of model files is required.
If the model has already been downloaded, the pipeline will automatically reuse the cached weights.

---

## Final Reproducible Entry Point

The required entry point is:

```python
from run_inference import run_inference

run_inference()
```

By default, this expects:

```text
Input data: data/public.jsonl by default, or pass --data data/private.jsonl for the Kaggle private set.
Output CSV: submission.csv
Debug output: results/debug_generations.jsonl
```

Calling `run_inference()` performs the full pipeline end-to-end:

1. Loads the input dataset.
2. Loads `Qwen/Qwen3-4B-Thinking-2507`.
3. Builds prompts.
4. Runs generation.
5. Applies answer extraction and normalization.
6. Applies self-consistency voting if enabled.
7. Writes the final Kaggle CSV.
8. Writes a debug JSONL file with generations, votes, and metadata.

---

## Command-Line Usage

### Final Kaggle submission run

The final `submission.csv` was generated using the required Qwen base model with structured chain-of-thought prompting, one sample per problem, and a 1024-token generation limit:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --debug_jsonl results/debug_generations_1024.jsonl \
  --output_csv submission.csv \
  --max_tokens 1024
```

This command writes generation records to `results/debug_generations_1024.jsonl` and rebuilds the final Kaggle submission file as `submission.csv`.

### Resume from checkpoint

`run_inference.py` treats the debug JSONL file as a checkpoint. If `--debug_jsonl` already exists, the script loads completed records from that file, skips examples with completed nonblank predictions, and appends new generations to the same file.

To resume the final run, use the same command without `--overwrite`:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --debug_jsonl results/debug_generations_1024.jsonl \
  --output_csv submission.csv \
  --max_tokens 1024
```

Do not pass `--overwrite` unless you want to delete the existing checkpoint and start from scratch.

### Local subset testing

For local testing on a subset of examples:

```bash
python run_inference.py \
  --data data/public.jsonl \
  --subset_ids data/eval_subset.json \
  --output_csv results/subset_submission.csv \
  --debug_jsonl results/subset_debug.jsonl \
  --max_tokens 1024
```

### Optional self-consistency run

The script also supports self-consistency decoding, although this was not the final submitted configuration:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output_csv submission_self_consistency.csv \
  --debug_jsonl results/debug_generations_self_consistency.jsonl \
  --num_samples 16 \
  --inner_batch 4 \
  --max_tokens 1024
```

### Optional LoRA/QLoRA adapter run

If using a fine-tuned adapter:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output_csv submission_adapter.csv \
  --debug_jsonl results/debug_generations_adapter.jsonl \
  --adapter your-username/your-model-name \
  --max_tokens 1024
```

### Optional memory-saving 4-bit run

If GPU memory is limited:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output_csv submission_4bit.csv \
  --debug_jsonl results/debug_generations_4bit.jsonl \
  --load_in_4bit \
  --max_tokens 1024
```

---

## Main Configuration Options

| Argument | Meaning | Final value used |
|---|---|---|
| `--data` | Input JSONL dataset path | `data/private.jsonl` |
| `--output_csv` | Kaggle CSV output path | `submission.csv` |
| `--debug_jsonl` | Debug JSONL checkpoint path | `results/debug_generations_1024.jsonl` |
| `--adapter` | Optional LoRA/QLoRA adapter path or Hub ID | `None` |
| `--prompt_strategy` | Prompt template: `structured_cot`, `verification`, or `vanilla` | `structured_cot` |
| `--num_samples` | Number of samples per question | `1` |
| `--inner_batch` | Sub-batch size for self-consistency generation | `4` |
| `--max_tokens` | Maximum new tokens per generation | `1024` |
| `--load_in_4bit` | Whether to use 4-bit loading | `False` |
| `--seed` | Random seed | `42` |
| `--id_column` | ID column name in output CSV | `id` |
| `--answer_column` | Answer column name in output CSV | `answer` |
| `--overwrite` | Delete existing checkpoint/output files and start from scratch | `False` |
| `--allow_model_id_override` | Bypass required model guardrail | `False` |
---
## Checkpointing and Resume Behavior

`run_inference.py` treats the debug JSONL file as a checkpoint.

By default, if `--debug_jsonl` already exists, the script loads completed records from that file, skips examples with completed nonblank predictions, and appends new generations to the same file.

To resume from a specific checkpoint, pass it as `--debug_jsonl`:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --debug_jsonl results/debug_generations_1024.jsonl \
  --output_csv submission.csv \
  --max_tokens 1024
---

## Main Configuration Options

| Argument | Meaning | Final value used |
|---|---|---|
| `--data` | Input JSONL dataset path | `data/private.jsonl` |
| `--output_csv` | Kaggle CSV output path | `submission.csv` |
| `--debug_jsonl` | Debug JSONL checkpoint path | `results/debug_generations_1024.jsonl` |
| `--adapter` | Optional LoRA/QLoRA adapter path or Hub ID | `None` |
| `--prompt_strategy` | Prompt template: `structured_cot`, `verification`, or `vanilla` | `structured_cot` |
| `--num_samples` | Number of samples per question | `1` |
| `--inner_batch` | Sub-batch size for self-consistency generation | `4` |
| `--max_tokens` | Maximum new tokens per generation | `1024` |
| `--load_in_4bit` | Whether to use 4-bit loading | `False` |
| `--seed` | Random seed | `42` |
| `--id_column` | ID column name in output CSV | `id` |
| `--answer_column` | Answer column name in output CSV | `answer` |
| `--overwrite` | Delete existing checkpoint/output files and start from scratch | `False` |
| `--allow_model_id_override` | Bypass required model guardrail | `False` |

The final Kaggle submission used the required competition model `Qwen/Qwen3-4B-Thinking-2507` with structured chain-of-thought prompting and a maximum generation length of 1024 tokens.

## Output Files

The script writes two files:

### 1. Kaggle submission CSV

Default path:

```text
submission.csv
```

Default columns:

```csv
id,answer
```

If Kaggle requires a different answer column name, pass:

```bash
--answer_column YOUR_COLUMN_NAME
```

### 2. Debug generations JSONL

Default path:

```text
results/debug_generations.jsonl
```

This file contains one JSON record per example, including:

- problem ID
- whether it was multiple-choice
- normalized prediction
- final selected response
- all votes
- all sampled responses when using self-consistency
- prompt strategy
- sampling parameters
- error information, if any

This file is useful for debugging but is not the Kaggle submission.

---

## Reproducibility Notes

The script sets a best-effort random seed with:

```text
seed = 42
```

However, exact string-identical outputs are not guaranteed because GPU kernels and sampling can introduce nondeterminism. The expected goal is consistent overall performance, not identical generations.

For reproducibility:

- Keep final hyperparameters fixed in this README.
- Use the same adapter checkpoint or HuggingFace Hub path.
- Use the same input file.
- Use the same code commit.
- Do not manually edit `submission.csv`.
- Do not manually replace individual answers.
- Do not use external tools or API calls at inference time.

---

## Academic Integrity Statement

This pipeline uses only the competition-required Qwen3-4B model and model-intrinsic methods at inference time. It does not use external APIs, larger models, calculators, code execution tools, retrieval systems, or manual answer editing during test-time inference.
