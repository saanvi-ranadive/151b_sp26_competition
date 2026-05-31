import json
import re
import csv
from pathlib import Path

DEBUG_PATH = Path("results/debug_generations_2048.jsonl")
OUT_PATH = Path("submission_postprocessed.csv")

DEFAULT_LETTERS = list("ABCDEFGH")

def clean_text(x):
    return str(x or "").strip()

def get_valid_letters(item):
    opts = item.get("options")
    if isinstance(opts, dict):
        return [str(k).upper() for k in opts.keys()]

    meta = item.get("metadata", {})
    opts = meta.get("options") if isinstance(meta, dict) else None
    if isinstance(opts, dict):
        return [str(k).upper() for k in opts.keys()]

    return DEFAULT_LETTERS

def get_qid(item):
    if "id" in item:
        return item["id"]
    if "question_id" in item:
        return item["question_id"]
    if "qid" in item:
        return item["qid"]
    raise ValueError(f"Missing id field in row: {item.keys()}")

def extract_mcq(response, old_pred, valid_letters):
    text = clean_text(response)[-1500:]
    valid_letters = [v.upper() for v in valid_letters]
    valid = "".join(valid_letters)

    patterns = [
        rf"\\boxed\{{\s*([{valid}])\s*\}}",
        rf"final answer(?: is|:)?\s*\(?([{valid}])\)?",
        rf"correct answer(?: is|:)?\s*\(?([{valid}])\)?",
        rf"answer(?: is|:)?\s*\(?([{valid}])\)?",
        rf"choice\s*\(?([{valid}])\)?",
        rf"option\s*\(?([{valid}])\)?",
        rf"Option\s+([{valid}])\s+is",
        rf"\(([{valid}])\)",
    ]

    matches = []
    for pat in patterns:
        matches.extend(re.findall(pat, text, flags=re.IGNORECASE))

    if matches:
        return matches[-1].upper()

    standalone = re.findall(rf"\b([{valid}])\b", text[-600:], flags=re.IGNORECASE)
    if standalone:
        return standalone[-1].upper()

    old_pred = clean_text(old_pred).upper()
    if old_pred in valid_letters:
        return old_pred

    return valid_letters[0]

def rebuild():
    rows = []

    with open(DEBUG_PATH, "r") as f:
        for line in f:
            item = json.loads(line)

            qid = get_qid(item)
            old_pred = item.get("prediction", "")

            if item.get("is_mcq"):
                pred = extract_mcq(
                    item.get("response", ""),
                    old_pred,
                    get_valid_letters(item)
                )
            else:
                # safest: do not alter FRQ predictions
                pred = old_pred

            rows.append({"id": qid, "prediction": pred})

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {OUT_PATH}")

if __name__ == "__main__":
    rebuild()