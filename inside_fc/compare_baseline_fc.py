import os
import json
import math
from statistics import mean

# ============================================================
# config
# ============================================================

OUTPUT_DIR = "inside_fc/outputs"
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "comparison_baseline_vs_fc_n300")

BASELINE_EXPERIMENT = "baseline03_n300_s41_42_43"
FC_EXPERIMENT = "fc03_n300_s41_42_43"

SEEDS = [41, 42, 43]

# how many examples to keep in "top" files
TOP_K = 30

# quantile used to define "low-risk" inside each class
# 0.25 means bottom 25% of eigenscore within that class
LOW_RISK_QUANTILE = 0.25


# ============================================================
# io helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_results_path(experiment_name, seed):
    return os.path.join(
        OUTPUT_DIR,
        f"stability_k20_{experiment_name}_results_seed{seed}.json"
    )


# ============================================================
# numeric helpers
# ============================================================

def quantile(values, q):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def safe_mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(mean(vals))


# ============================================================
# loading and matching
# ============================================================

def load_seed_results_as_dict(experiment_name, seed):
    path = get_results_path(experiment_name, seed)
    rows = load_json(path)

    by_id = {}
    for row in rows:
        qid = row["id"]
        by_id[qid] = row
    return by_id


def compare_one_seed(seed):
    base = load_seed_results_as_dict(BASELINE_EXPERIMENT, seed)
    fc = load_seed_results_as_dict(FC_EXPERIMENT, seed)

    common_ids = sorted(set(base.keys()) & set(fc.keys()))
    rows = []

    for qid in common_ids:
        b = base[qid]
        f = fc[qid]

        delta_eigenscore = float(f["eigenscore"] - b["eigenscore"])
        delta_unique = int(f["unique_generations"] - b["unique_generations"])

        row = {
            "seed": seed,
            "id": qid,
            "question": b["question"],
            "gold_answer": b["gold_answer"],

            "baseline_majority_answer": b["majority_answer"],
            "baseline_majority_is_correct": bool(b["majority_is_correct"]),
            "baseline_correctness_score_vs_gold": float(b["correctness_score_vs_gold"]),
            "baseline_unique_generations": int(b["unique_generations"]),
            "baseline_eigenscore": float(b["eigenscore"]),

            "fc_majority_answer": f["majority_answer"],
            "fc_majority_is_correct": bool(f["majority_is_correct"]),
            "fc_correctness_score_vs_gold": float(f["correctness_score_vs_gold"]),
            "fc_unique_generations": int(f["unique_generations"]),
            "fc_eigenscore": float(f["eigenscore"]),
            "fc_clip_fraction": f.get("clip_fraction", None),

            "delta_eigenscore": delta_eigenscore,
            "delta_unique_generations": delta_unique,
            "correctness_flip": (
                "same"
                if b["majority_is_correct"] == f["majority_is_correct"]
                else ("incorrect_to_correct" if (not b["majority_is_correct"] and f["majority_is_correct"]) else "correct_to_incorrect")
            )
        }
        rows.append(row)

    return rows


# ============================================================
# aggregate across seeds
# ============================================================

def aggregate_by_id(all_rows):
    grouped = {}

    for row in all_rows:
        qid = row["id"]
        if qid not in grouped:
            grouped[qid] = []
        grouped[qid].append(row)

    aggregated = []

    for qid, rows in grouped.items():
        first = rows[0]

        agg = {
            "id": qid,
            "question": first["question"],
            "gold_answer": first["gold_answer"],

            "n_seeds": len(rows),

            "baseline_correct_rate": float(mean([1.0 if r["baseline_majority_is_correct"] else 0.0 for r in rows])),
            "fc_correct_rate": float(mean([1.0 if r["fc_majority_is_correct"] else 0.0 for r in rows])),

            "baseline_mean_correctness_score_vs_gold": safe_mean([r["baseline_correctness_score_vs_gold"] for r in rows]),
            "fc_mean_correctness_score_vs_gold": safe_mean([r["fc_correctness_score_vs_gold"] for r in rows]),

            "baseline_mean_unique_generations": safe_mean([r["baseline_unique_generations"] for r in rows]),
            "fc_mean_unique_generations": safe_mean([r["fc_unique_generations"] for r in rows]),

            "baseline_mean_eigenscore": safe_mean([r["baseline_eigenscore"] for r in rows]),
            "fc_mean_eigenscore": safe_mean([r["fc_eigenscore"] for r in rows]),
            "mean_delta_eigenscore": safe_mean([r["delta_eigenscore"] for r in rows]),
            "mean_delta_unique_generations": safe_mean([r["delta_unique_generations"] for r in rows]),

            "mean_fc_clip_fraction": safe_mean([r["fc_clip_fraction"] for r in rows if r["fc_clip_fraction"] is not None]),

            "baseline_answers_by_seed": [
                {
                    "seed": r["seed"],
                    "answer": r["baseline_majority_answer"],
                    "is_correct": r["baseline_majority_is_correct"],
                    "eigenscore": r["baseline_eigenscore"],
                }
                for r in rows
            ],
            "fc_answers_by_seed": [
                {
                    "seed": r["seed"],
                    "answer": r["fc_majority_answer"],
                    "is_correct": r["fc_majority_is_correct"],
                    "eigenscore": r["fc_eigenscore"],
                }
                for r in rows
            ],
        }

        aggregated.append(agg)

    return aggregated


# ============================================================
# slices
# ============================================================

def build_low_risk_slices_seed_level(all_rows):
    baseline_incorrect_scores = [
        r["baseline_eigenscore"]
        for r in all_rows
        if not r["baseline_majority_is_correct"]
    ]
    baseline_correct_scores = [
        r["baseline_eigenscore"]
        for r in all_rows
        if r["baseline_majority_is_correct"]
    ]

    incorrect_threshold = quantile(baseline_incorrect_scores, LOW_RISK_QUANTILE)
    correct_threshold = quantile(baseline_correct_scores, LOW_RISK_QUANTILE)

    incorrect_low_risk_helped = []
    correct_low_risk_worsened = []

    for r in all_rows:
        if (not r["baseline_majority_is_correct"]) and r["baseline_eigenscore"] <= incorrect_threshold:
            if r["delta_eigenscore"] > 0:
                incorrect_low_risk_helped.append(r)

        if r["baseline_majority_is_correct"] and r["baseline_eigenscore"] <= correct_threshold:
            if r["delta_eigenscore"] > 0:
                correct_low_risk_worsened.append(r)

    incorrect_low_risk_helped.sort(key=lambda x: x["delta_eigenscore"], reverse=True)
    correct_low_risk_worsened.sort(key=lambda x: x["delta_eigenscore"], reverse=True)

    meta = {
        "low_risk_quantile": LOW_RISK_QUANTILE,
        "baseline_incorrect_low_risk_threshold": incorrect_threshold,
        "baseline_correct_low_risk_threshold": correct_threshold,
        "num_seed_level_rows": len(all_rows),
        "num_incorrect_low_risk_helped": len(incorrect_low_risk_helped),
        "num_correct_low_risk_worsened": len(correct_low_risk_worsened),
    }

    return meta, incorrect_low_risk_helped, correct_low_risk_worsened


def build_low_risk_slices_aggregated(aggregated_rows):
    baseline_incorrect_scores = [
        r["baseline_mean_eigenscore"]
        for r in aggregated_rows
        if r["baseline_correct_rate"] < 0.5
    ]
    baseline_correct_scores = [
        r["baseline_mean_eigenscore"]
        for r in aggregated_rows
        if r["baseline_correct_rate"] >= 0.5
    ]

    incorrect_threshold = quantile(baseline_incorrect_scores, LOW_RISK_QUANTILE)
    correct_threshold = quantile(baseline_correct_scores, LOW_RISK_QUANTILE)

    incorrect_low_risk_helped = []
    correct_low_risk_worsened = []

    for r in aggregated_rows:
        if (r["baseline_correct_rate"] < 0.5) and r["baseline_mean_eigenscore"] <= incorrect_threshold:
            if r["mean_delta_eigenscore"] > 0:
                incorrect_low_risk_helped.append(r)

        if (r["baseline_correct_rate"] >= 0.5) and r["baseline_mean_eigenscore"] <= correct_threshold:
            if r["mean_delta_eigenscore"] > 0:
                correct_low_risk_worsened.append(r)

    incorrect_low_risk_helped.sort(key=lambda x: x["mean_delta_eigenscore"], reverse=True)
    correct_low_risk_worsened.sort(key=lambda x: x["mean_delta_eigenscore"], reverse=True)

    meta = {
        "low_risk_quantile": LOW_RISK_QUANTILE,
        "baseline_incorrect_low_risk_threshold": incorrect_threshold,
        "baseline_correct_low_risk_threshold": correct_threshold,
        "num_aggregated_rows": len(aggregated_rows),
        "num_incorrect_low_risk_helped": len(incorrect_low_risk_helped),
        "num_correct_low_risk_worsened": len(correct_low_risk_worsened),
    }

    return meta, incorrect_low_risk_helped, correct_low_risk_worsened


# ============================================================
# tops
# ============================================================

def build_top_lists_seed_level(all_rows):
    top_helped = sorted(all_rows, key=lambda x: x["delta_eigenscore"], reverse=True)[:TOP_K]
    top_hurt = sorted(all_rows, key=lambda x: x["delta_eigenscore"])[:TOP_K]
    return top_helped, top_hurt


def build_top_lists_aggregated(aggregated_rows):
    top_helped = sorted(aggregated_rows, key=lambda x: x["mean_delta_eigenscore"], reverse=True)[:TOP_K]
    top_hurt = sorted(aggregated_rows, key=lambda x: x["mean_delta_eigenscore"])[:TOP_K]
    return top_helped, top_hurt


# ============================================================
# summary
# ============================================================

def build_summary(all_rows, aggregated_rows):
    seed_level_flips = {
        "same": 0,
        "incorrect_to_correct": 0,
        "correct_to_incorrect": 0,
    }
    for r in all_rows:
        seed_level_flips[r["correctness_flip"]] += 1

    baseline_mean_delta = safe_mean([r["delta_eigenscore"] for r in all_rows])
    aggregated_mean_delta = safe_mean([r["mean_delta_eigenscore"] for r in aggregated_rows])

    summary = {
        "baseline_experiment": BASELINE_EXPERIMENT,
        "fc_experiment": FC_EXPERIMENT,
        "seeds": SEEDS,
        "num_seed_level_rows": len(all_rows),
        "num_unique_questions": len(aggregated_rows),
        "low_risk_quantile": LOW_RISK_QUANTILE,
        "top_k": TOP_K,
        "seed_level_correctness_flips": seed_level_flips,
        "mean_delta_eigenscore_seed_level": baseline_mean_delta,
        "mean_delta_eigenscore_aggregated": aggregated_mean_delta,
        "mean_baseline_eigenscore_seed_level": safe_mean([r["baseline_eigenscore"] for r in all_rows]),
        "mean_fc_eigenscore_seed_level": safe_mean([r["fc_eigenscore"] for r in all_rows]),
        "mean_baseline_unique_generations_seed_level": safe_mean([r["baseline_unique_generations"] for r in all_rows]),
        "mean_fc_unique_generations_seed_level": safe_mean([r["fc_unique_generations"] for r in all_rows]),
        "mean_fc_clip_fraction_seed_level": safe_mean([r["fc_clip_fraction"] for r in all_rows if r["fc_clip_fraction"] is not None]),
    }
    return summary


# ============================================================
# main
# ============================================================

def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    all_rows = []
    for seed in SEEDS:
        rows = compare_one_seed(seed)
        all_rows.extend(rows)

    aggregated_rows = aggregate_by_id(all_rows)

    seed_meta, seed_incorrect_helped, seed_correct_worsened = build_low_risk_slices_seed_level(all_rows)
    agg_meta, agg_incorrect_helped, agg_correct_worsened = build_low_risk_slices_aggregated(aggregated_rows)

    seed_top_helped, seed_top_hurt = build_top_lists_seed_level(all_rows)
    agg_top_helped, agg_top_hurt = build_top_lists_aggregated(aggregated_rows)

    summary = build_summary(all_rows, aggregated_rows)

    save_json(os.path.join(ANALYSIS_DIR, "summary.json"), summary)

    save_json(os.path.join(ANALYSIS_DIR, "all_rows_seed_level.json"), all_rows)
    save_json(os.path.join(ANALYSIS_DIR, "all_rows_aggregated_by_id.json"), aggregated_rows)

    save_json(
        os.path.join(ANALYSIS_DIR, "seed_level_incorrect_low_risk_helped.json"),
        {"meta": seed_meta, "rows": seed_incorrect_helped}
    )
    save_json(
        os.path.join(ANALYSIS_DIR, "seed_level_correct_low_risk_worsened.json"),
        {"meta": seed_meta, "rows": seed_correct_worsened}
    )

    save_json(
        os.path.join(ANALYSIS_DIR, "aggregated_incorrect_low_risk_helped.json"),
        {"meta": agg_meta, "rows": agg_incorrect_helped}
    )
    save_json(
        os.path.join(ANALYSIS_DIR, "aggregated_correct_low_risk_worsened.json"),
        {"meta": agg_meta, "rows": agg_correct_worsened}
    )

    save_json(os.path.join(ANALYSIS_DIR, "seed_level_top_helped_by_delta_eigenscore.json"), seed_top_helped)
    save_json(os.path.join(ANALYSIS_DIR, "seed_level_top_hurt_by_delta_eigenscore.json"), seed_top_hurt)

    save_json(os.path.join(ANALYSIS_DIR, "aggregated_top_helped_by_delta_eigenscore.json"), agg_top_helped)
    save_json(os.path.join(ANALYSIS_DIR, "aggregated_top_hurt_by_delta_eigenscore.json"), agg_top_hurt)

    print("\n====================")
    print("comparison saved")
    print("====================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nfiles:")
    for name in [
        "summary.json",
        "all_rows_seed_level.json",
        "all_rows_aggregated_by_id.json",
        "seed_level_incorrect_low_risk_helped.json",
        "seed_level_correct_low_risk_worsened.json",
        "aggregated_incorrect_low_risk_helped.json",
        "aggregated_correct_low_risk_worsened.json",
        "seed_level_top_helped_by_delta_eigenscore.json",
        "seed_level_top_hurt_by_delta_eigenscore.json",
        "aggregated_top_helped_by_delta_eigenscore.json",
        "aggregated_top_hurt_by_delta_eigenscore.json",
    ]:
        print("-", os.path.join(ANALYSIS_DIR, name))


if __name__ == "__main__":
    main()
