"""
Advanced Evaluation Script
--------------------------
This script integrates BERTScore calculation with the EvalMethods class 
to compute professional metrics: AUROC, AURC, and ECE.

It evaluates the model's ability to detect hallucinations at the 
individual response level.
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from bert_score import score as bert_scorer

# Allow running this script directly from any working directory.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import your custom modules
from utils.load_datasets import load_all_datasets
from calculate_eval_metrics import EvalMethods

def run_evaluation_pipeline(trivia_path, truthful_path):
    """
    Loads data, computes BERTScores, and runs advanced metrics.
    """
    # 1. Load standardized data
    all_data = load_all_datasets(trivia_path, truthful_path)
    for ds_name, items in all_data.items():
        print(f"\n{'='*30}")
        print(f"Evaluating Dataset: {ds_name}")
        print(f"{'='*30}")

        y_true = []
        raw_bert_scores = []

        # 2. Collect labels and compute BERTScores for every response
        print(f"Computing BERTScores for {ds_name} responses...")
        for item in tqdm(items):
            ground_truth = item['ground_truth']
            
            # Extract text and labels for all responses in this question
            responses_text = [r['text'] for r in item['responses']]
            responses_labels = [r['is_hallucination'] for r in item['responses']]
            
            # Compute BERTScore (Similarity to ground truth)
            # Higher score means MORE factual
            _, _, f1_scores = bert_scorer(
                responses_text, 
                [ground_truth] * len(responses_text), 
                lang="en", 
                verbose=False
            )
            
            raw_bert_scores.extend(f1_scores.tolist())
            y_true.extend(responses_labels)

        # 3. Prepare scores for EvalMethods
        # Note: EvalMethods expects y_scores where HIGH value = Hallucination.
        # Since BERTScore is HIGH for Factual, we use (1 - score).
        y_scores = [1 - s for s in raw_bert_scores]

        # 4. Initialize your EvalMethods class
        evaluator = EvalMethods(y_true, print_logs=True)

        # 5. Compute Metrics and Generate Plots
        # method_name: folder name, score_name: specific test name
        evaluator.plot_roc(
            y_scores=y_scores, 
            name_method="bertscore_analysis", 
            name_score=f"{ds_name}_factuality"
        )

        # 6. Save results
        output_csv = f"outputs/stats/{ds_name}_metrics.csv"
        evaluator.save_eval_metrics(output_csv)
        print(f"Detailed metrics saved to: {output_csv}")

if __name__ == "__main__":
    # Define file paths
    TRIVIA_PATH = "../../data/triviaqa/qa_generations_820_12_annotated.json"
    TRUTHFUL_PATH = "../../data/truthfulqa/truthfulqa_with_hallucination_truth.json"

    # Ensure output directory exists
    os.makedirs("outputs/stats", exist_ok=True)

    try:
        run_evaluation_pipeline(TRIVIA_PATH, TRUTHFUL_PATH)
    except Exception as e:
        print(f"An error occurred during the pipeline: {e}")