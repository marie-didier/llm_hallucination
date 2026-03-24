#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "results" / "final"
TABLES_DIR = FINAL_DIR / "tables"
FIGURES_DIR = FINAL_DIR / "figures"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "font.size": 12,
        }
    )


def load_csv(name: str) -> pd.DataFrame:
    path = TABLES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    return pd.read_csv(path)


def finalize_axis(ax, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_panels(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    filename: str,
    x_label: str,
    x_tick_labels: list[str] | None = None,
    std_prefix: str = "std_",
    mean_prefix: str = "mean_",
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    metric_specs = [
        ("auroc", "AUROC", False),
        ("aurc", "AURC", False),
        ("pcc", "PCC", True),
    ]

    x_values = np.arange(len(df)) if x_col == "__index__" else df[x_col].to_numpy()

    for ax, (metric, label, allow_negative) in zip(axes, metric_specs):
        mean_col = f"{mean_prefix}{metric}" if f"{mean_prefix}{metric}" in df.columns else metric
        std_col = f"{std_prefix}{metric}" if f"{std_prefix}{metric}" in df.columns else None

        y = df[mean_col].to_numpy(dtype=float)

        if std_col and std_col in df.columns:
            yerr = df[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(x_values, y, yerr=yerr, marker="o", linewidth=2.0, capsize=4)
        else:
            ax.plot(x_values, y, marker="o", linewidth=2.0)

        ax.set_ylabel(label)
        finalize_axis(ax)

        if metric == "auroc":
            ax.set_ylim(0.0, 1.0)
        elif metric == "aurc":
            upper = max(0.35, float(np.nanmax(y)) * 1.2)
            ax.set_ylim(0.0, upper)
        elif metric == "pcc":
            if allow_negative:
                low = min(-0.7, float(np.nanmin(y)) * 1.15)
                high = max(0.05, float(np.nanmax(y)) * 1.15)
                ax.set_ylim(low, high)

    axes[0].set_title(title)
    axes[-1].set_xlabel(x_label)

    if x_tick_labels is not None:
        axes[-1].set_xticks(x_values)
        axes[-1].set_xticklabels(x_tick_labels, rotation=15)
    elif x_col == "__index__":
        axes[-1].set_xticks(x_values)

    savefig(filename)


def plot_final_roc_curve(samples_df: pd.DataFrame) -> None:
    df = samples_df[
        (samples_df["family"] == "baseline")
        & (samples_df["dataset"] == "qa820")
        & (samples_df["n_samples"] == 820)
        & (samples_df["k"] == 20)
    ].copy()

    if df.empty:
        return

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    base_fpr = np.linspace(0.0, 1.0, 400)
    tprs = []
    aucs = []

    for seed, group in df.groupby("generation_seed"):
        y_true = (~group["is_correct"].astype(bool)).astype(int).to_numpy()
        scores = group["eigenscore"].astype(float).to_numpy()

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, scores)
        seed_auc = auc(fpr, tpr)
        aucs.append(seed_auc)

        interp_tpr = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs.append(interp_tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=1.5,
            alpha=0.85,
            label=f"seed {seed} (AUROC={seed_auc:.3f})",
        )

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        ax.plot(
            base_fpr,
            mean_tpr,
            linewidth=3.0,
            label=f"mean ROC (AUROC={mean_auc:.3f}±{std_auc:.3f})",
        )
        ax.fill_between(
            base_fpr,
            np.maximum(mean_tpr - std_tpr, 0.0),
            np.minimum(mean_tpr + std_tpr, 1.0),
            alpha=0.15,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray", alpha=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve - INSIDE baseline on QA820 (n=820)")
    ax.legend(loc="upper left", frameon=True)
    finalize_axis(ax, grid_axis="both")
    savefig("final_roc_curve_inside_baseline_qa820_n820.png")


def plot_score_histogram(samples_df: pd.DataFrame) -> None:
    df = samples_df[
        (samples_df["family"] == "baseline")
        & (samples_df["dataset"] == "qa820")
        & (samples_df["n_samples"] == 820)
        & (samples_df["k"] == 20)
    ].copy()

    if df.empty:
        return

    correct = df[df["is_correct"] == True]["eigenscore"].astype(float).to_numpy()
    incorrect = df[df["is_correct"] == False]["eigenscore"].astype(float).to_numpy()

    if len(correct) == 0 or len(incorrect) == 0:
        return

    all_scores = np.concatenate([correct, incorrect])
    bins = np.histogram_bin_edges(all_scores, bins=35)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.hist(correct, bins=bins, density=True, alpha=0.6, label="correct")
    ax.hist(incorrect, bins=bins, density=True, alpha=0.6, label="incorrect")

    ax.axvline(np.mean(correct), linestyle="--", linewidth=1.5, alpha=0.9, label="mean correct")
    ax.axvline(np.mean(incorrect), linestyle="--", linewidth=1.5, alpha=0.9, label="mean incorrect")

    ax.set_xlabel("EigenScore")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution - INSIDE baseline on QA820 (n=820)")
    ax.legend()
    finalize_axis(ax)
    savefig("score_distribution_histogram_inside_qa820_n820.png")


def plot_scaling(table_scaling: pd.DataFrame) -> None:
    if table_scaling.empty:
        return

    df = table_scaling.sort_values("n_samples").copy()
    plot_metric_panels(
        df=df,
        x_col="n_samples",
        title="Scaling behaviour of INSIDE baseline on QA820",
        filename="inside_scaling_qa820.png",
        x_label="Number of questions",
    )


def plot_dataset_comparison(table_cross_dataset: pd.DataFrame) -> None:
    if table_cross_dataset.empty:
        return

    df = table_cross_dataset.copy()
    df["label"] = df.apply(lambda r: f"{r['dataset']} (n={int(r['n_samples'])})", axis=1)
    df = df.reset_index(drop=True)
    df["__index__"] = np.arange(len(df))

    plot_metric_panels(
        df=df,
        x_col="__index__",
        title="Cross-dataset performance of INSIDE baseline",
        filename="inside_dataset_comparison.png",
        x_label="Dataset",
        x_tick_labels=df["label"].tolist(),
    )


def plot_score_vs_correctness(samples_df: pd.DataFrame) -> None:
    df = samples_df[
        (samples_df["family"] == "baseline")
        & (samples_df["dataset"] == "qa820")
        & (samples_df["n_samples"] == 820)
        & (samples_df["k"] == 20)
    ].copy()

    if df.empty:
        return

    df["correct_num"] = df["is_correct"].astype(int)

    try:
        df["score_bin"] = pd.qcut(df["eigenscore"], q=12, duplicates="drop")
    except Exception:
        return

    agg = (
        df.groupby("score_bin", observed=False)
        .agg(
            mean_eigenscore=("eigenscore", "mean"),
            mean_correctness=("correct_num", "mean"),
            count=("correct_num", "count"),
        )
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(agg["mean_eigenscore"], agg["mean_correctness"], marker="o", linewidth=2.2)
    axes[0].set_ylabel("Mean correctness")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Correctness as a function of EigenScore bins")
    finalize_axis(axes[0])

    axes[1].bar(agg["mean_eigenscore"], agg["count"], width=np.diff(agg["mean_eigenscore"]).mean() * 0.7 if len(agg) > 1 else 0.1)
    axes[1].set_xlabel("Mean EigenScore per bin")
    axes[1].set_ylabel("Count")
    finalize_axis(axes[1])

    savefig("visualisation_eigenscore_correctness_moyenne.png")


def plot_fc_vs_baseline(table_fc_vs_baseline: pd.DataFrame) -> None:
    if table_fc_vs_baseline.empty:
        return

    df = table_fc_vs_baseline.copy().reset_index(drop=True)
    df["__index__"] = np.arange(len(df))

    plot_metric_panels(
        df=df,
        x_col="__index__",
        title="Baseline vs feature clipping on QA820 (n=300)",
        filename="inside_fc_vs_baseline_n300.png",
        x_label="Method",
        x_tick_labels=df["method_label"].tolist(),
    )


def plot_k_ablation(table_ablation: pd.DataFrame) -> None:
    if table_ablation.empty:
        return

    df = table_ablation.sort_values("k").copy()

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(df["k"], df["auroc"], marker="o", linewidth=2.0)
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Ablation over the number of generations K")
    finalize_axis(axes[0])

    axes[1].plot(df["k"], df["aurc"], marker="o", linewidth=2.0)
    axes[1].set_ylabel("AURC")
    finalize_axis(axes[1])

    axes[2].plot(df["k"], df["pcc"], marker="o", linewidth=2.0)
    axes[2].set_ylabel("PCC")
    axes[2].set_xlabel("K")
    finalize_axis(axes[2])

    savefig("inside_k_ablation.png")


def plot_fc_sweep(table_fc_sweep: pd.DataFrame) -> None:
    if table_fc_sweep.empty:
        return

    df = table_fc_sweep.copy().reset_index(drop=True)
    df["__index__"] = np.arange(len(df))

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].bar(df["__index__"], df["auroc"])
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Feature clipping sweep on QA820 (n=100)")
    finalize_axis(axes[0])

    axes[1].bar(df["__index__"], df["aurc"])
    axes[1].set_ylabel("AURC")
    finalize_axis(axes[1])

    axes[2].bar(df["__index__"], df["pcc"])
    axes[2].set_ylabel("PCC")
    axes[2].set_xlabel("Method")
    axes[2].set_xticks(df["__index__"])
    axes[2].set_xticklabels(df["method_label"], rotation=15)
    finalize_axis(axes[2])

    savefig("inside_fc_percentile_sweep_n100.png")


def plot_fc_flips(table_fc_case_summary: pd.DataFrame) -> None:
    if table_fc_case_summary.empty:
        print("warning: fc_case_summary.csv está vacío; se omite inside_fc_correctness_flips.png")
        return

    row = table_fc_case_summary.iloc[0]

    def safe_value(x) -> float:
        try:
            if pd.isna(x):
                return 0.0
            return float(x)
        except Exception:
            return 0.0

    labels = ["same", "incorrect_to_correct", "correct_to_incorrect"]
    values = [
        safe_value(row.get("same", 0)),
        safe_value(row.get("incorrect_to_correct", 0)),
        safe_value(row.get("correct_to_incorrect", 0)),
    ]

    if all(v == 0.0 for v in values):
        print("warning: fc_case_summary.csv no contiene flips válidos; se omite inside_fc_correctness_flips.png")
        return

    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()
    bars = ax.bar(labels, values)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(round(value))}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Number of cases")
    ax.set_title("Correctness flips after feature clipping on QA820 (n=300)")
    finalize_axis(ax)
    savefig("inside_fc_correctness_flips.png")


def main() -> None:
    ensure_dirs()
    configure_style()

    sample_scores_path = TABLES_DIR / "sample_scores_master.csv"
    sample_scores_master = pd.read_csv(sample_scores_path) if sample_scores_path.exists() else pd.DataFrame()

    table_scaling = load_csv("table_baseline_scaling.csv")
    table_cross_dataset = load_csv("table_cross_dataset.csv")
    table_fc_vs_baseline = load_csv("table_fc_vs_baseline_n300.csv")
    table_ablation = load_csv("table_ablation_k.csv")
    table_fc_sweep = load_csv("table_fc_sweep_n100.csv")

    fc_case_summary_path = TABLES_DIR / "fc_case_summary.csv"
    table_fc_case_summary = pd.read_csv(fc_case_summary_path) if fc_case_summary_path.exists() else pd.DataFrame()

    if not sample_scores_master.empty:
        plot_final_roc_curve(sample_scores_master)
        plot_score_histogram(sample_scores_master)
        plot_score_vs_correctness(sample_scores_master)

    plot_scaling(table_scaling)
    plot_dataset_comparison(table_cross_dataset)
    plot_fc_vs_baseline(table_fc_vs_baseline)
    plot_k_ablation(table_ablation)
    plot_fc_sweep(table_fc_sweep)
    plot_fc_flips(table_fc_case_summary)

    print("OK - figuras refinadas generadas en results/final/figures.")


if __name__ == "__main__":
    main()
