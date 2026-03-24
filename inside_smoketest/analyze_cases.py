import os
import json
import math
from collections import Counter, defaultdict

OUTPUT_DIR = "outputs/case_analysis"
TOP_N = 15

DATASETS = {
    "qa820": [
        "outputs/stability_k20_results_seed41.json",
        "outputs/stability_k20_results_seed42.json",
        "outputs/stability_k20_results_seed43.json",
    ],
    "triviaqa": [
        "outputs/triviaqa_inside_k20_results_seed41.json",
        "outputs/triviaqa_inside_k20_results_seed42.json",
        "outputs/triviaqa_inside_k20_results_seed43.json",
    ],
}


def safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def safe_std(values):
    if not values:
        return None
    m = safe_mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def load_results(paths):
    all_runs = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        all_runs.append(rows)

    return all_runs


def aggregate_runs(all_runs):
    by_id = defaultdict(list)

    for run_idx, rows in enumerate(all_runs):
        for row in rows:
            by_id[row["id"]].append(row)

    aggregated = []

    for qid, rows in by_id.items():
        rows = sorted(rows, key=lambda x: x["id"])

        question = rows[0]["question"]
        gold_answer = rows[0]["gold_answer"]

        eigenscores = [float(r["eigenscore"]) for r in rows]
        correctness_scores = [float(r["correctness_score_vs_gold"]) for r in rows]
        correct_flags = [bool(r["majority_is_correct"]) for r in rows]
        unique_generations = [int(r["unique_generations"]) for r in rows]
        majority_answers = [r["majority_answer"] for r in rows]

        answer_counter = Counter(majority_answers)
        most_common_answer, most_common_answer_count = answer_counter.most_common(1)[0]

        aggregated_row = {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "mean_eigenscore": safe_mean(eigenscores),
            "std_eigenscore": safe_std(eigenscores),
            "min_eigenscore": min(eigenscores),
            "max_eigenscore": max(eigenscores),
            "mean_correctness_score_vs_gold": safe_mean(correctness_scores),
            "std_correctness_score_vs_gold": safe_std(correctness_scores),
            "correct_count": int(sum(correct_flags)),
            "incorrect_count": int(len(correct_flags) - sum(correct_flags)),
            "correct_rate": float(sum(correct_flags) / len(correct_flags)),
            "seed_level_majority_is_correct": correct_flags,
            "seed_level_eigenscores": eigenscores,
            "seed_level_correctness_scores": correctness_scores,
            "seed_level_unique_generations": unique_generations,
            "seed_level_majority_answers": majority_answers,
            "mean_unique_generations": safe_mean(unique_generations),
            "most_common_majority_answer": most_common_answer,
            "most_common_majority_answer_count": int(most_common_answer_count),
        }

        # clase agregada robusta por mayoría de seeds
        if aggregated_row["correct_count"] >= 2:
            aggregated_row["aggregate_label"] = "correct"
        else:
            aggregated_row["aggregate_label"] = "incorrect"

        aggregated.append(aggregated_row)

    return aggregated


def sort_desc(rows, key):
    return sorted(rows, key=lambda x: x[key], reverse=True)


def sort_asc(rows, key):
    return sorted(rows, key=lambda x: x[key])


def build_case_splits(aggregated_rows, top_n=15):
    incorrect_rows = [r for r in aggregated_rows if r["aggregate_label"] == "incorrect"]
    correct_rows = [r for r in aggregated_rows if r["aggregate_label"] == "correct"]

    incorrect_high_risk = sort_desc(incorrect_rows, "mean_eigenscore")[:top_n]
    incorrect_low_risk = sort_asc(incorrect_rows, "mean_eigenscore")[:top_n]
    correct_high_risk = sort_desc(correct_rows, "mean_eigenscore")[:top_n]
    correct_low_risk = sort_asc(correct_rows, "mean_eigenscore")[:top_n]

    return {
        "incorrect_high_risk": incorrect_high_risk,
        "incorrect_low_risk": incorrect_low_risk,
        "correct_high_risk": correct_high_risk,
        "correct_low_risk": correct_low_risk,
    }


def summarize_dataset(aggregated_rows):
    incorrect_rows = [r for r in aggregated_rows if r["aggregate_label"] == "incorrect"]
    correct_rows = [r for r in aggregated_rows if r["aggregate_label"] == "correct"]

    return {
        "n_items": len(aggregated_rows),
        "n_correct_aggregate": len(correct_rows),
        "n_incorrect_aggregate": len(incorrect_rows),
        "mean_eigenscore_all": safe_mean([r["mean_eigenscore"] for r in aggregated_rows]),
        "mean_eigenscore_correct": safe_mean([r["mean_eigenscore"] for r in correct_rows]),
        "mean_eigenscore_incorrect": safe_mean([r["mean_eigenscore"] for r in incorrect_rows]),
        "mean_correct_rate_all": safe_mean([r["correct_rate"] for r in aggregated_rows]),
    }


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset_name, paths in DATASETS.items():
        print("\n====================")
        print(f"processing dataset: {dataset_name}")
        print("====================")

        all_runs = load_results(paths)
        aggregated_rows = aggregate_runs(all_runs)
        case_splits = build_case_splits(aggregated_rows, top_n=TOP_N)
        dataset_summary = summarize_dataset(aggregated_rows)

        dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        save_json(
            os.path.join(dataset_dir, "aggregated_cases.json"),
            aggregated_rows
        )
        save_json(
            os.path.join(dataset_dir, "dataset_summary.json"),
            dataset_summary
        )
        save_json(
            os.path.join(dataset_dir, "incorrect_high_risk.json"),
            case_splits["incorrect_high_risk"]
        )
        save_json(
            os.path.join(dataset_dir, "incorrect_low_risk.json"),
            case_splits["incorrect_low_risk"]
        )
        save_json(
            os.path.join(dataset_dir, "correct_high_risk.json"),
            case_splits["correct_high_risk"]
        )
        save_json(
            os.path.join(dataset_dir, "correct_low_risk.json"),
            case_splits["correct_low_risk"]
        )

        print("summary:")
        print(dataset_summary)
        print("saved in:", dataset_dir)


if __name__ == "__main__":
    main()
