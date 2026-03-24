#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS_DIR = ROOT / "results" / "local_runs"
FINAL_DIR = ROOT / "results" / "final"
TABLES_DIR = FINAL_DIR / "tables"
SUMMARIES_DIR = FINAL_DIR / "summaries"
CASES_DIR = FINAL_DIR / "cases"
CONFIGS_DIR = ROOT / "configs"


OFFICIAL_CONFIGS = [
    "baseline_smoke.yaml",
    "baseline_ablation_k_k5.yaml",
    "baseline_ablation_k_k10.yaml",
    "baseline_ablation_k_k20.yaml",
    "baseline_qa820_n100.yaml",
    "baseline_qa820_n300.yaml",
    "baseline_qa820_n500.yaml",
    "baseline_qa820_n820.yaml",
    "baseline_triviaqa_n100.yaml",
    "fc_qa820_n100_p01.yaml",
    "fc_qa820_n100_p02.yaml",
    "fc_qa820_n100_p05.yaml",
    "fc_qa820_n300_p05.yaml",
    "fc_qa820_nall_p05.yaml",
    "fc_triviaqa_nall_p05.yaml",
]


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def first(d: dict, *keys: str, default=None):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def maybe_float(x: Any):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def maybe_int(x: Any):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def infer_family(run_name: str, cfg: dict, summary: dict) -> str:
    value = str(first(cfg, "method", default="") or first(summary, "method", default="")).lower()
    name = run_name.lower()
    if value == "fc" or name.startswith("fc_"):
        return "fc"
    return "baseline"


def infer_dataset(run_name: str, cfg: dict, summary: dict) -> str:
    for raw in [
        first(cfg, "dataset_name", default=None),
        first(summary, "dataset_name", default=None),
        first(cfg, "input_json", default=None),
        first(summary, "input_json", default=None),
        run_name,
    ]:
        text = str(raw).lower()
        if "trivia" in text:
            return "triviaqa"
        if "qa820" in text or "qa_generations_820" in text:
            return "qa820"
    return "unknown"


def infer_fc_percentiles(run_name: str, cfg: dict, summary: dict) -> tuple[float | None, float | None]:
    low = maybe_float(first(cfg, "fc_lower_percentile", default=None))
    high = maybe_float(first(cfg, "fc_upper_percentile", default=None))
    if low is not None or high is not None:
        return low, high

    name = run_name.lower()
    if "fc01" in name or "p01" in name:
        return 0.1, 99.9
    if "fc02" in name or "p02" in name:
        return 0.2, 99.8
    if "fc05" in name or "p05" in name:
        return 0.5, 99.5
    return None, None


def extract_rows_from_results(results_obj: Any) -> list[dict]:
    if isinstance(results_obj, list):
        return [row for row in results_obj if isinstance(row, dict)]

    if isinstance(results_obj, dict):
        for key in ["results", "rows", "items", "examples", "data"]:
            value = results_obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]

    return []


def iter_run_dirs():
    if not LOCAL_RUNS_DIR.exists():
        return

    for run_dir in sorted(LOCAL_RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("_"):
            continue
        if run_dir.name == "comparison_baseline_vs_fc_n300":
            continue

        summary_path = run_dir / "summary.json"
        results_path = run_dir / "results.json"
        config_path = run_dir / "config.json"

        if summary_path.exists() and config_path.exists():
            yield run_dir, config_path, summary_path, results_path


def normalize_summary(run_dir: Path, config_path: Path, summary_path: Path) -> dict:
    cfg = load_json(config_path)
    summary = load_json(summary_path)
    run_name = str(first(summary, "experiment_name", default=run_dir.name))
    family = infer_family(run_name, cfg, summary)
    dataset = infer_dataset(run_name, cfg, summary)
    fc_low, fc_high = infer_fc_percentiles(run_name, cfg, summary)

    row = {
        "run_dir": run_dir.name,
        "experiment_name": run_name,
        "family": family,
        "dataset": dataset,
        "n_samples": maybe_int(first(summary, "n_samples", default=first(cfg, "n_samples", default=None))),
        "k": maybe_int(first(summary, "k", default=first(cfg, "k", default=None))),
        "seed_dataset": maybe_int(first(summary, "seed_dataset", default=first(cfg, "seed_dataset", default=None))),
        "generation_seed": maybe_int(first(summary, "generation_seed", default=first(cfg, "generation_seed", default=None))),
        "fc_enabled": bool(first(cfg, "use_feature_clipping", default=(family == "fc"))),
        "fc_lower_percentile": fc_low,
        "fc_upper_percentile": fc_high,
        "mean_clip_fraction": maybe_float(
            first(summary, "mean_clip_fraction", default=first(summary, "clip_fraction", default=None))
        ),
        "auroc": maybe_float(
            first(
                summary,
                "auroc_incorrect_vs_eigenscore",
                "AUROC",
                "auroc",
                "mean_auroc_incorrect_vs_eigenscore",
                default=None,
            )
        ),
        "aurc": maybe_float(
            first(
                summary,
                "aurc_incorrect_vs_eigenscore",
                "AURC",
                "aurc",
                "mean_aurc_incorrect_vs_eigenscore",
                default=None,
            )
        ),
        "pcc": maybe_float(
            first(
                summary,
                "pcc_correctness_vs_eigenscore",
                "PCC",
                "pcc",
                "mean_pcc_correctness_vs_eigenscore",
                default=None,
            )
        ),
        "num_correct": maybe_int(first(summary, "num_correct", default=None)),
        "num_incorrect": maybe_int(first(summary, "num_incorrect", default=None)),
        "input_json": first(summary, "input_json", default=first(cfg, "input_json", default=None)),
    }

    return row


def normalize_sample_rows(run_dir: Path, config_path: Path, summary_path: Path, results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []

    cfg = load_json(config_path)
    summary = load_json(summary_path)
    results_obj = load_json(results_path)
    raw_rows = extract_rows_from_results(results_obj)

    run_name = str(first(summary, "experiment_name", default=run_dir.name))
    family = infer_family(run_name, cfg, summary)
    dataset = infer_dataset(run_name, cfg, summary)
    fc_low, fc_high = infer_fc_percentiles(run_name, cfg, summary)
    generation_seed = maybe_int(first(summary, "generation_seed", default=first(cfg, "generation_seed", default=None)))
    n_samples = maybe_int(first(summary, "n_samples", default=first(cfg, "n_samples", default=None)))
    k_value = maybe_int(first(summary, "k", default=first(cfg, "k", default=None)))

    rows = []
    for i, raw in enumerate(raw_rows):
        eigenscore = maybe_float(first(raw, "eigenscore", "score", "risk_score", default=None))
        correctness_score = maybe_float(first(raw, "correctness_score", "answer_score", "match_score", default=None))
        is_correct = first(raw, "is_correct", "correct", "majority_correct", "is_majority_correct", default=None)

        if is_correct is None and correctness_score is not None:
            is_correct = correctness_score >= 0.5

        if eigenscore is None or is_correct is None:
            continue

        row = {
            "run_dir": run_dir.name,
            "experiment_name": run_name,
            "family": family,
            "dataset": dataset,
            "n_samples": n_samples,
            "k": k_value,
            "generation_seed": generation_seed,
            "seed_dataset": maybe_int(first(summary, "seed_dataset", default=first(cfg, "seed_dataset", default=None))),
            "fc_enabled": bool(first(cfg, "use_feature_clipping", default=(family == "fc"))),
            "fc_lower_percentile": fc_low,
            "fc_upper_percentile": fc_high,
            "question_id": first(raw, "id", "question_id", "sample_id", "index", default=i),
            "question": first(raw, "question", default=None),
            "gold_answer": first(raw, "gold_answer", "gold", default=None),
            "majority_answer": first(raw, "majority_answer", "predicted_answer", "answer", default=None),
            "eigenscore": eigenscore,
            "correctness_score": correctness_score,
            "is_correct": bool(is_correct),
            "unique_generations": maybe_int(first(raw, "unique_generations", "num_unique_generations", default=None)),
            "clip_fraction": maybe_float(first(raw, "clip_fraction", default=None)),
        }
        rows.append(row)

    return rows


def aggregate_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_aurc=("aurc", "mean"),
        std_aurc=("aurc", "std"),
        mean_pcc=("pcc", "mean"),
        std_pcc=("pcc", "std"),
        mean_num_correct=("num_correct", "mean"),
        mean_num_incorrect=("num_incorrect", "mean"),
        mean_clip_fraction=("mean_clip_fraction", "mean"),
        n_runs=("experiment_name", "count"),
    ).reset_index()

    for col in ["std_auroc", "std_aurc", "std_pcc", "mean_clip_fraction"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    return out


def build_main_tables(metrics_df: pd.DataFrame) -> None:
    baseline_df = metrics_df[metrics_df["family"] == "baseline"].copy()
    fc_df = metrics_df[metrics_df["family"] == "fc"].copy()

    scaling = aggregate_metrics(
        baseline_df[
            (baseline_df["dataset"] == "qa820")
            & (baseline_df["k"] == 20)
            & (baseline_df["n_samples"].isin([100, 300, 500, 820]))
        ],
        ["dataset", "n_samples", "k"],
    ).sort_values("n_samples")
    scaling.to_csv(TABLES_DIR / "table_baseline_scaling.csv", index=False)
    save_json(SUMMARIES_DIR / "baseline_qa820_scaling_summary.json", scaling.to_dict(orient="records"))

    cross_dataset = aggregate_metrics(
        baseline_df[
            ((baseline_df["dataset"] == "qa820") & (baseline_df["n_samples"] == 820))
            | ((baseline_df["dataset"] == "triviaqa") & (baseline_df["n_samples"] == 100))
        ],
        ["dataset", "n_samples", "k"],
    )
    cross_dataset.to_csv(TABLES_DIR / "table_cross_dataset.csv", index=False)
    save_json(SUMMARIES_DIR / "baseline_triviaqa_summary.json", cross_dataset.to_dict(orient="records"))

    ablation = baseline_df[
        (baseline_df["dataset"] == "qa820")
        & (baseline_df["n_samples"] == 100)
        & (baseline_df["k"].isin([5, 10, 20]))
        & (baseline_df["generation_seed"] == 42)
    ][["dataset", "n_samples", "k", "generation_seed", "auroc", "aurc", "pcc", "num_correct", "num_incorrect"]].sort_values("k")
    ablation.to_csv(TABLES_DIR / "table_ablation_k.csv", index=False)
    save_json(SUMMARIES_DIR / "baseline_ablation_k_summary.json", ablation.to_dict(orient="records"))

    fc_sweep = metrics_df[
        (
            (metrics_df["family"] == "baseline")
            & (metrics_df["dataset"] == "qa820")
            & (metrics_df["n_samples"] == 100)
            & (metrics_df["generation_seed"] == 42)
            & (metrics_df["k"] == 20)
        )
        | (
            (metrics_df["family"] == "fc")
            & (metrics_df["dataset"] == "qa820")
            & (metrics_df["n_samples"] == 100)
            & (metrics_df["generation_seed"] == 42)
            & (metrics_df["k"] == 20)
        )
    ].copy()

    def fc_label(row):
        if row["family"] == "baseline":
            return "baseline"
        low = row["fc_lower_percentile"]
        high = row["fc_upper_percentile"]
        return f"fc_{low:.1f}_{high:.1f}"

    fc_sweep["method_label"] = fc_sweep.apply(fc_label, axis=1)
    fc_sweep = fc_sweep[
        ["method_label", "family", "dataset", "n_samples", "k", "generation_seed", "auroc", "aurc", "pcc", "mean_clip_fraction"]
    ].sort_values("method_label")
    fc_sweep.to_csv(TABLES_DIR / "table_fc_sweep_n100.csv", index=False)
    save_json(SUMMARIES_DIR / "fc_sweep_n100_summary.json", fc_sweep.to_dict(orient="records"))

    fc_vs_baseline_n300 = metrics_df[
        (
            (metrics_df["family"] == "baseline")
            & (metrics_df["dataset"] == "qa820")
            & (metrics_df["n_samples"] == 300)
            & (metrics_df["k"] == 20)
        )
        | (
            (metrics_df["family"] == "fc")
            & (metrics_df["dataset"] == "qa820")
            & (metrics_df["n_samples"] == 300)
            & (metrics_df["k"] == 20)
            & (metrics_df["fc_lower_percentile"] == 0.5)
            & (metrics_df["fc_upper_percentile"] == 99.5)
        )
    ].copy()

    fc_vs_baseline_n300["method_label"] = np.where(
        fc_vs_baseline_n300["family"] == "baseline",
        "baseline",
        "fc_0.5_99.5",
    )

    fc_vs_baseline_n300 = aggregate_metrics(fc_vs_baseline_n300, ["method_label", "family", "dataset", "n_samples", "k"])
    fc_vs_baseline_n300.to_csv(TABLES_DIR / "table_fc_vs_baseline_n300.csv", index=False)
    save_json(SUMMARIES_DIR / "fc_vs_baseline_n300_summary.json", fc_vs_baseline_n300.to_dict(orient="records"))

    main_rows = pd.concat(
        [
            aggregate_metrics(
                baseline_df[
                    (baseline_df["dataset"] == "qa820")
                    & (baseline_df["n_samples"] == 820)
                    & (baseline_df["k"] == 20)
                ],
                ["family", "dataset", "n_samples", "k"],
            ).assign(result_label="baseline_qa820_n820"),
            aggregate_metrics(
                baseline_df[
                    (baseline_df["dataset"] == "triviaqa")
                    & (baseline_df["n_samples"] == 100)
                    & (baseline_df["k"] == 20)
                ],
                ["family", "dataset", "n_samples", "k"],
            ).assign(result_label="baseline_triviaqa_n100"),
            aggregate_metrics(
                fc_df[
                    (fc_df["dataset"] == "qa820")
                    & (fc_df["n_samples"] == 300)
                    & (fc_df["k"] == 20)
                    & (fc_df["fc_lower_percentile"] == 0.5)
                    & (fc_df["fc_upper_percentile"] == 99.5)
                ],
                ["family", "dataset", "n_samples", "k"],
            ).assign(result_label="fc_qa820_n300_p05"),
        ],
        ignore_index=True,
    )

    main_rows.to_csv(TABLES_DIR / "table_main_results.csv", index=False)


def build_manifest() -> None:
    rows = []
    for name in OFFICIAL_CONFIGS:
        rows.append(
            {
                "config_name": name,
                "exists_in_configs": (CONFIGS_DIR / name).exists(),
                "official_status": "official",
            }
        )

    pd.DataFrame(rows).to_csv(TABLES_DIR / "official_runs_manifest.csv", index=False)

def export_fc_case_summary() -> None:
    comparison_dir = LOCAL_RUNS_DIR / "comparison_baseline_vs_fc_n300"
    if not comparison_dir.exists():
        print("warning: no existe comparison_baseline_vs_fc_n300; no se exporta fc_case_summary")
        return

    summary_path = comparison_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        save_json(CASES_DIR / "fc_vs_baseline_summary.json", summary)

        flips = first(summary, "correctness_flips", default={})
        if not isinstance(flips, dict):
            flips = {}

        compact = {
            "mean_delta_eigenscore_seed_level": first(summary, "mean_delta_eigenscore_seed_level", default=None),
            "mean_baseline_unique_generations_seed_level": first(summary, "mean_baseline_unique_generations_seed_level", default=None),
            "mean_fc_unique_generations_seed_level": first(summary, "mean_fc_unique_generations_seed_level", default=None),
            "mean_fc_clip_fraction_seed_level": first(summary, "mean_fc_clip_fraction_seed_level", default=None),
            "same": first(flips, "same", default=first(summary, "same", default=0)),
            "incorrect_to_correct": first(flips, "incorrect_to_correct", default=first(summary, "incorrect_to_correct", default=0)),
            "correct_to_incorrect": first(flips, "correct_to_incorrect", default=first(summary, "correct_to_incorrect", default=0)),
        }

        for key in ["same", "incorrect_to_correct", "correct_to_incorrect"]:
            if compact[key] is None:
                compact[key] = 0

        pd.DataFrame([compact]).to_csv(TABLES_DIR / "fc_case_summary.csv", index=False)

    for src_name, dst_name in [
        ("aggregated_incorrect_low_risk_helped.json", "incorrect_low_risk_helped.json"),
        ("aggregated_correct_low_risk_worsened.json", "correct_low_risk_worsened.json"),
    ]:
        src = comparison_dir / src_name
        dst = CASES_DIR / dst_name
        if src.exists():
            shutil.copyfile(src, dst)


def main() -> None:
    ensure_dirs()

    metric_rows = []
    sample_rows = []

    for run_dir, config_path, summary_path, results_path in iter_run_dirs():
        metric_rows.append(normalize_summary(run_dir, config_path, summary_path))
        sample_rows.extend(normalize_sample_rows(run_dir, config_path, summary_path, results_path))

    metrics_df = pd.DataFrame(metric_rows)
    samples_df = pd.DataFrame(sample_rows)

    if metrics_df.empty:
        raise RuntimeError("No se encontraron summaries válidos en results/local_runs/.")

    metrics_df = metrics_df.sort_values(["family", "dataset", "n_samples", "k", "generation_seed"], na_position="last")
    metrics_df.to_csv(TABLES_DIR / "metrics_master.csv", index=False)

    if not samples_df.empty:
        samples_df = samples_df.sort_values(["family", "dataset", "n_samples", "k", "generation_seed", "question_id"])
        samples_df.to_csv(TABLES_DIR / "sample_scores_master.csv", index=False)

    build_main_tables(metrics_df)
    build_manifest()
    export_fc_case_summary()

    print("OK - tablas finales generadas en results/final/tables y summaries.")


if __name__ == "__main__":
    main()
