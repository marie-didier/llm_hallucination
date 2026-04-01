"""
BERTScore Evaluation Pipeline
-----------------------------
This script handles the evaluation of semantic centroids using BERTScore.
It ensures all outputs (JSON, CSV, Plots) are centralized in 'outputs/stats/'.
"""

import os
import sys
import json
import torch
import shutil
from pathlib import Path
from tqdm import tqdm
from bert_score import score as bert_scorer

# --- 1. Path Management ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]

# Force the working directory to the project root
os.chdir(PROJECT_ROOT)

# Standardize the stats directory path
STATS_DIR = os.path.join("outputs", "stats")
os.makedirs(STATS_DIR, exist_ok=True)

# Add 'src' to sys.path
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Internal imports
from calculate_eval_metrics import EvalMethods

from utils.centroids_selection import run_selection_pipeline 

def evaluate_bertscore():
    """
    Computes BERTScores and saves metrics/plots inside outputs/stats/
    """
    path_to_search = Path(STATS_DIR)
    files_to_process = list(path_to_search.glob("*_centroid.json"))

    if not files_to_process:
        print(f"Error: No centroid files found in {STATS_DIR}.")
        return

    for file_path in files_to_process:
        ds_name = file_path.stem.replace("_centroid", "")

        print(f"\n[Running BERTScore Evaluation: {ds_name}]")

        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)

        if not items:
            continue

        y_true = [item['is_hallucination'] for item in items]
        selected_texts = [item['selected_response'] for item in items]
        ground_truths = [item['ground_truth'] for item in items]

        print("-> Computing BERTScores...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, f1_scores = bert_scorer(
            selected_texts, 
            ground_truths, 
            lang="en", 
            device=device,
            verbose=False
        )
        
        y_scores = [1 - s for s in f1_scores.tolist()]

        # Initialize EvalMethods
        evaluator = EvalMethods(y_true, print_logs=True)
        
        # --- FIX: THE PATH HACK ---
        # The library tries to write to: outputs/stats/bertscore_plots/stats/bertscore_plots_...
        # We must create this nested "stats" folder to prevent Errno 2
        method_name = "stats/bertscore_plots"
        nested_dir = os.path.join("outputs", method_name, "stats")
        os.makedirs(nested_dir, exist_ok=True)
        
        evaluator.plot_roc(
            y_scores=y_scores, 
            name_method=method_name, 
            name_score=f"bertscore_{ds_name}"
        )

        # Save metrics CSV using the 'bertscore' naming convention
        output_csv = os.path.join(STATS_DIR, f"{ds_name}_bertscore_metrics.csv")
        evaluator.save_eval_metrics(output_csv)
        print(f"Metrics saved to: {output_csv}")

if __name__ == "__main__":
    # Ensure data paths are correct
    TRIVIA_PATH = os.path.join("data", "triviaqa", "qa_generations_820_12_annotated.json")
    TRUTHFUL_PATH = os.path.join("data", "truthfulqa", "truthfulqa_with_hallucination_truth.json")
    
    # Check if centroids need to be generated
    expected_files = [
        os.path.join(STATS_DIR, "triviaqa_centroid.json"), 
        os.path.join(STATS_DIR, "truthfulqa_centroid.json")
    ]
    missing_files = [f for f in expected_files if not os.path.exists(f)]

    if missing_files:
        print(f"Centroids missing. Running selection pipeline...")
        try:
            run_selection_pipeline(TRIVIA_PATH, TRUTHFUL_PATH)
        except Exception as e:
            print(f"Centroid selection failed: {e}")
            sys.exit(1)

    try:
        evaluate_bertscore()
        print(f"\nPipeline finished. Results are in: {os.path.abspath(STATS_DIR)}")
    except Exception as e:
        print(f"Evaluation crashed: {e}")