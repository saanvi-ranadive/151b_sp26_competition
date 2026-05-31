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
├── constraints.txt
├── run_inference.py
├── common_utils.py
├── data/
│   └── private.jsonl                  # Not committed if competition rules prohibit sharing
├── results/
│   └── debug_generations.jsonl         # Generated after inference
└── submission.csv                      # Generated after inference
```

Required files:

- `README.md`: This file.
- `run_inference.py`: Final reproducible inference entry point.
- `common_utils.py`: Shared model loading, prompting, extraction, generation, and evaluation helpers.
- `constraints.txt`: Python dependencies needed to run the pipeline.

Do **not** commit large model checkpoints, `.venv/`, HuggingFace cache folders, API keys, or private competition data unless course rules explicitly allow it.

---

## Hardware and Runtime

Final run hardware:

- GPU type used: TODO after testing, e.g. `A100 80GB`, `RTX 4090`, `T4`, etc.
- Number of GPUs used: TODO
- Approximate total inference/generation time: TODO
- Approximate runtime for final settings: TODO

Example format after testing:

```text
GPU type: NVIDIA A100 80GB
Number of GPUs: 1
Full private-set inference time: approximately 2.5 hours
Final configuration: structured_cot, num_samples=16, max_tokens=1536, load_in_4bit=False
```

---

## Installation

Create and activate an environment:

```bash
conda create -n cse151b-kaggle python=3.10 -y
conda activate cse151b-kaggle
```

Install dependencies:

```bash
pip install -r constraints.txt
```

Example `constraints.txt`:

```text
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
```

If using vLLM inside `common_utils.py`, also include:

```text
vllm
```

---

## Model Weights Setup

### Base model

The pipeline is designed to use:

```text
Qwen/Qwen3-4B-Thinking-2507
```

The script includes a guardrail that checks `common_utils.MODEL_ID` against the required model ID. Make sure `common_utils.py` contains:

```python
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
```

### If using no fine-tuned adapter

No additional model files are required beyond access to the base model through HuggingFace.

### If using a fine-tuned LoRA/QLoRA adapter

Upload the adapter to HuggingFace Hub, then pass the Hub path to `--adapter`.

Example:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output_csv submission.csv \
  --adapter your-username/your-model-name
```

The adapter repository should contain the required PEFT adapter files, such as:

```text
adapter_config.json
adapter_model.safetensors
```

If the tokenizer was modified during training, upload tokenizer files as well.

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
3. Optionally loads a LoRA/QLoRA adapter.
4. Builds prompts.
5. Runs generation.
6. Applies answer extraction and normalization.
7. Applies self-consistency voting if enabled.
8. Writes the final Kaggle CSV.
9. Writes a debug JSONL file with generations, votes, and metadata.

---

## Command-Line Usage

### Basic final run

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output_csv submission.csv
```

### Self-consistency run

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output_csv submission.csv \
  --num_samples 16 \
  --inner_batch 4 \
  --max_tokens 1536
```

### LoRA/QLoRA adapter run

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output_csv submission.csv \
  --adapter your-username/your-model-name
```

### Memory-saving 4-bit run

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output_csv submission.csv \
  --load_in_4bit
```

### Local subset testing

```bash
python run_inference.py \
  --data data/public.jsonl \
  --subset_ids data/eval_subset.json \
  --output_csv results/subset_submission.csv \
  --debug_jsonl results/subset_debug.jsonl
```
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
  --max_tokens 2048
---

## Main Configuration Options

| Argument | Meaning | Final value used |
|---|---|---|
| `--data` | Input JSONL dataset path | TODO |
| `--output_csv` | Kaggle CSV output path | `submission.csv` |
| `--debug_jsonl` | Debug JSONL path | `results/debug_generations.jsonl` |
| `--adapter` | Optional LoRA/QLoRA adapter path or Hub ID | TODO or `None` |
| `--prompt_strategy` | `structured_cot`, `verification`, or `vanilla` | TODO |
| `--num_samples` | Number of samples per question | TODO |
| `--inner_batch` | Sub-batch size for self-consistency generation | TODO |
| `--max_tokens` | Maximum new tokens per generation | TODO |
| `--load_in_4bit` | Whether to use 4-bit loading | TODO |
| `--seed` | Random seed | `42` |

Before final submission, replace every `TODO` in the rightmost column with the exact settings used for the final Kaggle CSV.

---

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

## Competition Compliance Checklist

### GitHub repo

- [ ] Repo is public by the deadline.
- [ ] `README.md` is complete.
- [ ] `run_inference.py` exists.
- [ ] `run_inference()` can be imported and called.
- [ ] `common_utils.py` is included.
- [ ] `constraints.txt` is included.
- [ ] `common_utils.MODEL_ID` is `Qwen/Qwen3-4B-Thinking-2507`.
- [ ] Final hyperparameters are listed in the README.
- [ ] GPU type and approximate runtime are listed in the README.
- [ ] Model weight setup instructions are listed in the README.
- [ ] The exact reproduction command is listed in the README.
- [ ] No secrets, API keys, `.venv/`, cache folders, or unauthorized private data are committed.

### HuggingFace Hub, if fine-tuned

- [ ] Adapter/model is uploaded to HuggingFace Hub.
- [ ] The Hub repo is accessible to course staff.
- [ ] The adapter path in the README matches the adapter path used by `run_inference.py`.
- [ ] The uploaded checkpoint is the same one used for the final Kaggle submission.

### Kaggle

- [ ] `submission.csv` has the correct columns.
- [ ] `submission.csv` has one row per test example.
- [ ] No IDs are missing.
- [ ] No duplicate IDs are present.
- [ ] The submitted CSV was generated by `run_inference.py`.
- [ ] Final selected Kaggle submission matches the CSV generated by this repo.

### Gradescope

- [ ] Submit the public GitHub repo link.
- [ ] Add all group members to the Gradescope submission.
- [ ] Make sure the submitted repo link points to the final version of the code.

---

## What to Update After GPU Testing

After you successfully test the pipeline, update the following items before final submission:

1. **GPU and runtime section**
   - GPU type
   - number of GPUs
   - total generation time
   - whether 4-bit loading was used

2. **Final configuration table**
   - final `--adapter` value
   - final `--prompt_strategy`
   - final `--num_samples`
   - final `--inner_batch`
   - final `--max_tokens`
   - final `--load_in_4bit` setting

3. **Final reproduction command**

Replace this with the exact command used to generate the final Kaggle CSV:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --debug_jsonl results/debug_generations_1024.jsonl \
  --output_csv submission.csv \
  --max_tokens 1024
```

If using an adapter, include:

```bash
--adapter your-username/your-model-name
```

4. **Kaggle CSV validation results**

Before uploading to Kaggle, verify:

```python
import pandas as pd

df = pd.read_csv("submission.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df["id"].duplicated().sum())
print(df["answer"].isna().sum())
```

Expected:
- correct row count
- columns match Kaggle format
- duplicate IDs = 0
- missing answers = 0

5. **README TODO cleanup**

Search the README for:

```text
TODO
```

and replace every remaining placeholder.

---

## Academic Integrity Statement

This pipeline uses only the competition-required Qwen3-4B model and model-intrinsic methods at inference time. It does not use external APIs, larger models, calculators, code execution tools, retrieval systems, or manual answer editing during test-time inference.
