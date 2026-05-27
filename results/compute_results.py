import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

FILES = [
    "model0_results.jsonl",
    "model1_v1_enhanced_cot_results.jsonl",
    "model1_v2_fewshot_results.jsonl",
    "model1_v3_verification_results.jsonl",
    "model2_self_consistency_N8_results.jsonl",
    "model3_fewshot_self_consistency_N8_results.jsonl",
    "model4_fewshot_prompting_results.jsonl",
    "model4_verification_self_consistency_N8_results.jsonl"
]

MODEL_NAMES = [
    "Baseline",
    "Enhanced CoT",
    "Few-Shot v1",
    "Verification v1",
    "SC N=8",
    "FewShot + SC",
    "Few-Shot Optimized",
    "Verification + SC"
]

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def accuracy(rows):
    if not rows:
        return 0, 0, 0.0
    correct = sum(bool(r["correct"]) for r in rows)
    total = len(rows)
    return correct, total, 100 * correct / total

def summarize_file(path):
    results = load_jsonl(path)

    mcq = [r for r in results if r["is_mcq"]]
    free = [r for r in results if not r["is_mcq"]]

    mcq_correct, mcq_total, mcq_acc = accuracy(mcq)
    free_correct, free_total, free_acc = accuracy(free)
    total_correct, total_total, total_acc = accuracy(results)

    return {
        "file": path.name,
        "mcq_acc": mcq_acc,
        "free_acc": free_acc,
        "overall_acc": total_acc,
    }

def plot_results(models, mcq_accs, free_accs, overall_accs):
    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(16, 6))

    plt.bar(x - width, mcq_accs, width, label="MCQ")
    plt.bar(x, free_accs, width, label="Free Response")
    plt.bar(x + width, overall_accs, width, label="Overall")

    plt.xticks(x, models, rotation=30, ha='right')
    plt.ylabel("Accuracy (%)")
    plt.title("Model Performance Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig("grouped_bar_chart.png", dpi=300)
    plt.show()

def main():
    print(f"{'Model/File':<55} {'MCQ %':>8} {'Free %':>8} {'Overall %':>10}")
    print("-" * 100)

    mcq_accs = []
    free_accs = []
    overall_accs = []

    valid_models = []

    for filename, model_name in zip(FILES, MODEL_NAMES):
        path = Path(filename)

        if not path.exists():
            print(f"{filename:<55} MISSING")
            continue

        row = summarize_file(path)

        valid_models.append(model_name)

        mcq_accs.append(row["mcq_acc"])
        free_accs.append(row["free_acc"])
        overall_accs.append(row["overall_acc"])

        print(
            f"{model_name:<55} "
            f"{row['mcq_acc']:>7.2f}% "
            f"{row['free_acc']:>7.2f}% "
            f"{row['overall_acc']:>9.2f}%"
        )

    plot_results(valid_models, mcq_accs, free_accs, overall_accs)

if __name__ == "__main__":
    main()