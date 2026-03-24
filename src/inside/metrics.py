from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc_incorrect_vs_eigenscore(
    majority_is_correct_list: List[int | bool],
    eigenscores: List[float],
) -> Optional[float]:
    correct = np.array([1 if bool(x) else 0 for x in majority_is_correct_list], dtype=np.int32)
    incorrect = 1 - correct
    scores = np.array(eigenscores, dtype=np.float64)

    if len(correct) == 0:
        return None

    if incorrect.min() == incorrect.max():
        return None

    return float(roc_auc_score(incorrect, scores))


def compute_aurc(
    majority_is_correct_list: List[int | bool],
    eigenscores: List[float],
) -> Optional[float]:
    correct = np.array([1 if bool(x) else 0 for x in majority_is_correct_list], dtype=np.int32)
    scores = np.array(eigenscores, dtype=np.float64)

    if len(correct) == 0:
        return None

    order = np.argsort(scores)
    correct_sorted = correct[order]

    risks = []
    cum_correct = 0
    n = len(correct_sorted)

    for i in range(1, n + 1):
        cum_correct += int(correct_sorted[i - 1])
        accuracy = cum_correct / i
        risk = 1.0 - accuracy
        risks.append(risk)

    return float(np.mean(risks))


def compute_pcc(
    correctness_scores: List[float],
    eigenscores: List[float],
) -> Optional[float]:
    x = np.array(correctness_scores, dtype=np.float64)
    y = np.array(eigenscores, dtype=np.float64)

    if len(x) < 2:
        return None

    x_std = x.std()
    y_std = y.std()

    if x_std == 0.0 or y_std == 0.0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def safe_std(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    mean_value = sum(valid) / len(valid)
    variance = sum((v - mean_value) ** 2 for v in valid) / len(valid)
    return float(variance ** 0.5)


def summarize_example_level_results(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    majority_is_correct = [bool(row["majority_is_correct"]) for row in rows]
    correctness_scores = [float(row["correctness_score_vs_gold"]) for row in rows]
    eigenscores = [float(row["eigenscore"]) for row in rows]

    num_correct = int(sum(majority_is_correct))
    num_incorrect = int(len(rows) - num_correct)

    mean_eigenscore_correct = None
    mean_eigenscore_incorrect = None

    if num_correct > 0:
        mean_eigenscore_correct = float(
            sum(r["eigenscore"] for r in rows if bool(r["majority_is_correct"])) / num_correct
        )

    if num_incorrect > 0:
        mean_eigenscore_incorrect = float(
            sum(r["eigenscore"] for r in rows if not bool(r["majority_is_correct"])) / num_incorrect
        )

    clip_values = [
        float(row["clip_fraction"])
        for row in rows
        if row.get("clip_fraction") is not None
    ]

    summary = {
        "experiment_name": config.get("experiment_name"),
        "method": config.get("method"),
        "model_name": config.get("model_name"),
        "dataset_name": config.get("dataset_name"),
        "dataset_path": config.get("dataset_path"),
        "seed_dataset": config.get("seed_dataset"),
        "generation_seed": config.get("generation_seed"),
        "generation_seeds": config.get("generation_seeds"),
        "n_samples": config.get("n_samples"),
        "k": config.get("k"),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "top_k": config.get("top_k"),
        "max_new_tokens": config.get("max_new_tokens"),
        "alpha": config.get("alpha"),
        "embedding_layer": config.get("embedding_layer"),
        "embedding_token": config.get("embedding_token"),
        "eigenscore_mode": config.get("eigenscore_mode"),
        "correctness_threshold": config.get("correctness_threshold"),
        "use_feature_clipping": bool(config.get("use_feature_clipping", False)),
        "fc_hook_location": config.get("fc_hook_location"),
        "fc_lower_percentile": config.get("fc_lower_percentile"),
        "fc_upper_percentile": config.get("fc_upper_percentile"),
        "fc_threshold_source": config.get("fc_threshold_source"),
        "calibration_n": config.get("calibration_n"),
        "calibration_k": config.get("calibration_k"),
        "num_examples": len(rows),
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "mean_eigenscore_correct": mean_eigenscore_correct,
        "mean_eigenscore_incorrect": mean_eigenscore_incorrect,
        "auroc_incorrect_vs_eigenscore": compute_auroc_incorrect_vs_eigenscore(
            majority_is_correct,
            eigenscores,
        ),
        "aurc_incorrect_vs_eigenscore": compute_aurc(
            majority_is_correct,
            eigenscores,
        ),
        "pcc_correctness_vs_eigenscore": compute_pcc(
            correctness_scores,
            eigenscores,
        ),
        "mean_clip_fraction": safe_mean(clip_values),
        "std_clip_fraction": safe_std(clip_values),
    }

    return summary


def aggregate_seed_summaries(seed_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not seed_summaries:
        raise ValueError("seed_summaries is empty")

    first = seed_summaries[0]

    aggregated = {
        "experiment_name": first.get("experiment_name"),
        "method": first.get("method"),
        "model_name": first.get("model_name"),
        "dataset_name": first.get("dataset_name"),
        "dataset_path": first.get("dataset_path"),
        "seed_dataset": first.get("seed_dataset"),
        "generation_seeds": [s.get("generation_seed") for s in seed_summaries],
        "n_samples": first.get("n_samples"),
        "k": first.get("k"),
        "temperature": first.get("temperature"),
        "top_p": first.get("top_p"),
        "top_k": first.get("top_k"),
        "max_new_tokens": first.get("max_new_tokens"),
        "alpha": first.get("alpha"),
        "embedding_layer": first.get("embedding_layer"),
        "embedding_token": first.get("embedding_token"),
        "eigenscore_mode": first.get("eigenscore_mode"),
        "correctness_threshold": first.get("correctness_threshold"),
        "use_feature_clipping": bool(first.get("use_feature_clipping", False)),
        "fc_hook_location": first.get("fc_hook_location"),
        "fc_lower_percentile": first.get("fc_lower_percentile"),
        "fc_upper_percentile": first.get("fc_upper_percentile"),
        "fc_threshold_source": first.get("fc_threshold_source"),
        "calibration_n": first.get("calibration_n"),
        "calibration_k": first.get("calibration_k"),
        "mean_num_correct": safe_mean([s.get("num_correct") for s in seed_summaries]),
        "mean_num_incorrect": safe_mean([s.get("num_incorrect") for s in seed_summaries]),
        "mean_eigenscore_correct": safe_mean([s.get("mean_eigenscore_correct") for s in seed_summaries]),
        "mean_eigenscore_incorrect": safe_mean([s.get("mean_eigenscore_incorrect") for s in seed_summaries]),
        "mean_auroc": safe_mean([s.get("auroc_incorrect_vs_eigenscore") for s in seed_summaries]),
        "std_auroc": safe_std([s.get("auroc_incorrect_vs_eigenscore") for s in seed_summaries]),
        "mean_aurc": safe_mean([s.get("aurc_incorrect_vs_eigenscore") for s in seed_summaries]),
        "std_aurc": safe_std([s.get("aurc_incorrect_vs_eigenscore") for s in seed_summaries]),
        "mean_pcc": safe_mean([s.get("pcc_correctness_vs_eigenscore") for s in seed_summaries]),
        "std_pcc": safe_std([s.get("pcc_correctness_vs_eigenscore") for s in seed_summaries]),
        "mean_clip_fraction": safe_mean([s.get("mean_clip_fraction") for s in seed_summaries]),
        "std_clip_fraction": safe_std([s.get("mean_clip_fraction") for s in seed_summaries]),
        "runs": seed_summaries,
    }

    return aggregated
