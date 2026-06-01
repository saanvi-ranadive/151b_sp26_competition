import re
import csv
from pathlib import Path

INPUT_PATH = Path("submission_fixed.csv")
OUT_PATH = Path("submission_postprocessed.csv")

DEFAULT_LETTERS = list("ABCDEFGH")

def clean_text(x):
    return str(x or "").strip()

def extract_mcq(response, old_pred, valid_letters=DEFAULT_LETTERS):
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

    old_pred = clean_text(old_pred).upper()
    if old_pred in valid_letters:
        return old_pred

    # If it is not clearly an MCQ answer, keep original response.
    return clean_text(response)

def rebuild():
    rows = []

    with open(INPUT_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if "id" not in fieldnames:
            raise ValueError(f"CSV must contain an 'id' column. Found: {fieldnames}")

        if "response" not in fieldnames:
            raise ValueError(f"CSV must contain a 'response' column. Found: {fieldnames}")

        for row in reader:
            qid = row["id"]
            old_response = row["response"]

            pred = extract_mcq(old_response, old_response)

            rows.append({"id": qid, "response": pred})

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {OUT_PATH}")

if __name__ == "__main__":
    rebuild()