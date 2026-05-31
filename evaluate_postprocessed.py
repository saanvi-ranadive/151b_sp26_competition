import json
from rebuild_submission import extract_mcq, get_valid_letters

path = "results/debug_generations_2048.jsonl"

total = correct = 0
mcq_total = mcq_correct = 0
frq_total = frq_correct = 0

for line in open(path):
    item = json.loads(line)

    if item.get("is_mcq"):
        old_pred = str(item.get("prediction", "")).strip().upper()
        gold = str(item.get("gold", "")).strip().upper()

        pred = extract_mcq(
            item.get("response", ""),
            old_pred,
            get_valid_letters(item)
        ).strip().upper()

        is_correct = pred == gold

        mcq_total += 1
        mcq_correct += int(is_correct)

    else:
        # Do not exact-match FRQ strings; use original correctness flag
        is_correct = bool(item.get("correct"))

        frq_total += 1
        frq_correct += int(is_correct)

    total += 1
    correct += int(is_correct)

print("=" * 60)
print(f"Overall Accuracy : {correct/total:.4%}")
print(f"MCQ Accuracy     : {mcq_correct/mcq_total:.4%}")
print(f"FRQ Accuracy     : {frq_correct/frq_total:.4%}")
print("=" * 60)
print(f"Total Correct    : {correct}/{total}")