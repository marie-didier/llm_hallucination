from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inside.utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs feature clipping experiment outputs.")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to aggregated baseline run directory inside results/local_runs, or directly to results.json.",
    )
    parser.add_argument(
        "--fc",
        type=str,
        required=True,
        help="Path to aggregated FC run directory inside results/local_runs, or directly to results.json.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="comparison_baseline_vs_fc",
        help="Name of output directory under results/final/cases and results/local_runs.",
    )
    return parser.parse_args()


def resolve_run_dir_or_results(path_str: str) -> Tuple[Path, bool]:
    path = Path(path_str)
    if path.is_file():
        return path, True
    return path, False


def load_single_results(path_str: str) -> List[Dict[str, Any]]:
    path, is_file = resolve_run_dir_or_results(path_str)
    if is_file:
        return load_json(path)

    results_path = path / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {path}")
    return load_json(results_path)


def load_aggregated_summary(path_str: str) -> Dict[str, Any]:
    path, is_file = resolve_run_dir_or_results(path_str)
    if is_file:
        raise ValueError("expected run directory for aggregated comparison, not direct file path")

    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {path}")

    return load_json(summary_path)


def load_seed_runs_from_aggregated_dir(path_str: str) -> List[Tuple[int, List[Dict[str, Any]]]]:
    run_dir = Path(path_str)
    aggregated_summary = load_aggregated_summary(path_str)

    if "runs" not in aggregated_summary:
        raise ValueError(f"{run_dir} does not look like an aggregated multi-seed run")

    seed_runs = []
    for run_summary in aggregated_summary["runs"]:
        generation_seed = run_summary.get("generation_seed")
        if generation_seed is None:
            raise ValueError("missing generation_seed in aggregated summary run")

        seed_experiment_name = run_summary.get("experiment_name")
        if not seed_experiment_name:
            raise ValueError("missing experiment_name in aggregated summary run")

        seed_results_path = REPO_ROOT / "results" / "local_runs" / seed_experiment_name / "results.json"
        if not seed_results_path.exists():
            raise FileNotFoundError(f"seed results not found at {seed_results_path}")

        seed_rows = load_json(seed_results_path)
        seed_runs.append((int(generation_seed), seed_rows))

    return seed_runs


def build_keyed_rows(rows: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    keyed = {}
    for row in rows:
        row_id = row.get("id")
        if row_id is None:
            raise ValueError("row is missing 'id', cannot compare by example id")
        keyed[row_id] = row
    return keyed


def compare_seed_rows(
    baseline_rows: List[Dict[str, Any]],
    fc_rows: List[Dict[str, Any]],
    generation_seed: int,
) -> List[Dict[str, Any]]:
    baseline_by_id = build_keyed_rows(baseline_rows)
    fc_by_id = build_keyed_rows(fc_rows)

    common_ids = sorted(set(baseline_by_id.keys()) & set(fc_by_id.keys()))
    compared_rows = []

    for row_id in common_ids:
        b = baseline_by_id[row_id]
        f = fc_by_id[row_id]

        baseline_eigenscore = float(b["eigenscore"])
        fc_eigenscore = float(f["eigenscore"])
        delta_eigenscore = fc_eigenscore - baseline_eigenscore

        baseline_correct = bool(b["is_correct"])
        fc_correct = bool(f["is_correct"])

        compared_rows.append(
            {
                "id": row_id,
                "generation_seed": generation_seed,
                "question": b.get("question"),
                "gold_candidates": b.get("gold_candidates"),
                "baseline_eigenscore": baseline_eigenscore,
                "fc_eigenscore": fc_eigenscore,
                "delta_eigenscore": delta_eigenscore,
                "baseline_is_correct": baseline_correct,
                "fc_is_correct": fc_correct,
                "baseline_majority_answer": b.get("majority_answer"),
                "fc_majority_answer": f.get("majority_answer"),
                "baseline_unique_generations": b.get("unique_generations"),
                "fc_unique_generations": f.get("unique_generations"),
                "fc_clip_fraction": f.get("clip_fraction"),
                "correctness_flip": classify_correctness_flip(baseline_correct, fc_correct),
            }
        )

    return compared_rows


def classify_correctness_flip(baseline_correct: bool, fc_correct: bool) -> str:
    if baseline_correct == fc_correct:
        return "same"
    if (not baseline_correct) and fc_correct:
        return "incorrect_to_correct"
    return "correct_to_incorrect"


def aggregate_rows_by_id(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)

    aggregated = []
    for row_id, group in grouped.items():
        first = group[0]

        baseline_scores = [float(r["baseline_eigenscore"]) for r in group]
        fc_scores = [float(r["fc_eigenscore"]) for r in group]
        delta_scores = [float(r["delta_eigenscore"]) for r in group]
        baseline_unique = [float(r["baseline_unique_generations"]) for r in group]
        fc_unique = [float(r["fc_unique_generations"]) for r in group]
        fc_clip = [float(r["fc_clip_fraction"]) for r in group if r.get("fc_clip_fraction") is not None]

        baseline_correct_votes = [int(bool(r["baseline_is_correct"])) for r in group]
        fc_correct_votes = [int(bool(r["fc_is_correct"])) for r in group]

        aggregated.append(
            {
                "id": row_id,
                "question": first.get("question"),
                "gold_candidates": first.get("gold_candidates"),
                "n_seeds": len(group),
                "mean_baseline_eigenscore": float(np.mean(baseline_scores)),
                "mean_fc_eigenscore": float(np.mean(fc_scores)),
                "mean_delta_eigenscore": float(np.mean(delta_scores)),
                "mean_baseline_unique_generations": float(np.mean(baseline_unique)),
                "mean_fc_unique_generations": float(np.mean(fc_unique)),
                "mean_fc_clip_fraction": float(np.mean(fc_clip)) if fc_clip else None,
                "baseline_correct_fraction": float(np.mean(baseline_correct_votes)),
                "fc_correct_fraction": float(np.mean(fc_correct_votes)),
                "baseline_majority_correct": bool(round(np.mean(baseline_correct_votes))),
                "fc_majority_correct": bool(round(np.mean(fc_correct_votes))),
                "seed_rows": group,
            }
        )

    aggregated.sort(key=lambda x: x["mean_delta_eigenscore"], reverse=True)
    return aggregated


def build_summary(seed_rows: List[Dict[str, Any]], aggregated_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    flip_counter = defaultdict(int)
    for row in seed_rows:
        flip_counter[row["correctness_flip"]] += 1

    summary = {
        "n_seed_rows": len(seed_rows),
        "n_aggregated_ids": len(aggregated_rows),
        "mean_delta_eigenscore_seed_level": float(np.mean([r["delta_eigenscore"] for r in seed_rows])),
        "mean_delta_eigenscore_aggregated": float(np.mean([r["mean_delta_eigenscore"] for r in aggregated_rows])),
        "mean_baseline_eigenscore_seed_level": float(np.mean([r["baseline_eigenscore"] for r in seed_rows])),
        "mean_fc_eigenscore_seed_level": float(np.mean([r["fc_eigenscore"] for r in seed_rows])),
        "mean_baseline_unique_generations_seed_level": float(np.mean([r["baseline_unique_generations"] for r in seed_rows])),
        "mean_fc_unique_generations_seed_level": float(np.mean([r["fc_unique_generations"] for r in seed_rows])),
        "mean_fc_clip_fraction_seed_level": float(
            np.mean([r["fc_clip_fraction"] for r in seed_rows if r.get("fc_clip_fraction") is not None])
        ),
        "seed_level_correctness_flips": dict(sorted(flip_counter.items())),
    }

    return summary


def select_incorrect_low_risk_helped(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        if (not row["baseline_majority_correct"]) and (not row["fc_majority_correct"]):
            if row["mean_delta_eigenscore"] > 0:
                selected.append(row)
    selected.sort(key=lambda x: x["mean_delta_eigenscore"], reverse=True)
    return selected


def select_correct_low_risk_worsened(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        if row["baseline_majority_correct"] and row["fc_majority_correct"]:
            if row["mean_delta_eigenscore"] > 0:
                selected.append(row)
    selected.sort(key=lambda x: x["mean_delta_eigenscore"], reverse=True)
    return selected


def select_top_helped(rows: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda x: x["mean_delta_eigenscore"], reverse=True)
    return ranked[:top_k]


def select_top_hurt(rows: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda x: x["mean_delta_eigenscore"])
    return ranked[:top_k]


def main() -> None:
    args = parse_args()

    baseline_seed_runs = load_seed_runs_from_aggregated_dir(args.baseline)
    fc_seed_runs = load_seed_runs_from_aggregated_dir(args.fc)

    baseline_by_seed = {seed: rows for seed, rows in baseline_seed_runs}
    fc_by_seed = {seed: rows for seed, rows in fc_seed_runs}

    common_seeds = sorted(set(baseline_by_seed.keys()) & set(fc_by_seed.keys()))
    if not common_seeds:
        raise ValueError("no common generation seeds found between baseline and fc runs")

    all_seed_rows = []
    for seed in common_seeds:
        seed_rows = compare_seed_rows(
            baseline_rows=baseline_by_seed[seed],
            fc_rows=fc_by_seed[seed],
            generation_seed=seed,
        )
        all_seed_rows.extend(seed_rows)

    aggregated_rows = aggregate_rows_by_id(all_seed_rows)
    summary = build_summary(all_seed_rows, aggregated_rows)

    incorrect_low_risk_helped = select_incorrect_low_risk_helped(aggregated_rows)
    correct_low_risk_worsened = select_correct_low_risk_worsened(aggregated_rows)
    top_helped = select_top_helped(aggregated_rows)
    top_hurt = select_top_hurt(aggregated_rows)

    local_output_dir = REPO_ROOT / "results" / "local_runs" / args.output_name
    final_output_dir = REPO_ROOT / "results" / "final" / "cases"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    save_json(all_seed_rows, local_output_dir / "all_rows_seed_level.json")
    save_json(aggregated_rows, local_output_dir / "all_rows_aggregated_by_id.json")
    save_json(summary, local_output_dir / "summary.json")
    save_json(incorrect_low_risk_helped, local_output_dir / "aggregated_incorrect_low_risk_helped.json")
    save_json(correct_low_risk_worsened, local_output_dir / "aggregated_correct_low_risk_worsened.json")
    save_json(top_helped, local_output_dir / "aggregated_top_helped_by_delta_eigenscore.json")
    save_json(top_hurt, local_output_dir / "aggregated_top_hurt_by_delta_eigenscore.json")

    save_json(summary, final_output_dir / "fc_vs_baseline_summary.json")
    save_json(incorrect_low_risk_helped, final_output_dir / "incorrect_low_risk_helped.json")
    save_json(correct_low_risk_worsened, final_output_dir / "correct_low_risk_worsened.json")

    print(f"common_seeds: {common_seeds}")
    print(f"seed_rows: {len(all_seed_rows)}")
    print(f"aggregated_ids: {len(aggregated_rows)}")
    print(f"local_output: {local_output_dir}")
    print(f"final_cases_output: {final_output_dir}")


if __name__ == "__main__":
    main()
