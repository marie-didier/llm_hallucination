"""
NLI-Based Centroid Selection
---------------------------
This script processes TriviaQA and TruthfulQA to extract one "best" answer per question.

Logic:
1. For each question, we compare all generated responses using an NLI model.
2. We build an agreement graph: if Response A entails Response B, A gets a point.
3. The response with the highest centrality (the semantic center) is selected.
4. Outputs are saved as individual files in the project's 'outputs/stats/' directory.
"""

import json
import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Path Management ---
# Locate the project root based on this script's position (src/scores/centroids_selection.py)
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_SCRIPT_PATH)))

# Add 'src' to the system path to allow importing from the utils folder
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.load_datasets import load_all_datasets

def get_nli_centroid(responses, model, tokenizer, device, threshold=0.5):
    """
    Find the response that best represents the semantic consensus.
    Uses an NLI model to determine which answer is logically implied by the most others.
    """
    if not responses:
        return None
    if len(responses) == 1:
        return responses[0]

    texts = [r['text'] for r in responses]
    n = len(texts)
    scores = [0] * n
    
    # Generate all directed pairs for the NLI model (A entails B)
    pairs = []
    pair_indices = []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            pairs.append((texts[i], texts[j]))
            pair_indices.append((i, j))

    # Batching to manage VRAM usage
    batch_size = 32
    entailment_probs = []
    
    with torch.no_grad():
        for k in range(0, len(pairs), batch_size):
            batch = pairs[k : k + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            # cross-encoder/nli-deberta-v3-large: Index 0 is 'entailment'
            probs = torch.softmax(logits, dim=1)[:, 0].cpu().tolist()
            entailment_probs.extend(probs)

    # Calculate centrality scores based on logical agreement
    for idx, prob in enumerate(entailment_probs):
        if prob > threshold:
            i, j = pair_indices[idx]
            scores[i] += 1 

    # Select the response with the highest consensus score
    best_idx = scores.index(max(scores))
    return responses[best_idx]

def run_selection_pipeline(trivia_path, truthful_path):
    """
    Runs the full selection process and saves results to the project root's output folder.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Absolute path for outputs to ensure consistency
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "stats")
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Initializing NLI model on {device} ---")
    model_name = "cross-encoder/nli-deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Load and standardize data using absolute paths
    all_data = load_all_datasets(trivia_path, truthful_path)

    for ds_name, items in all_data.items():
        print(f"\n[Processing {ds_name}] Extracting semantic centroids...")
        processed_data = []
        
        for item in tqdm(items):
            best_response = get_nli_centroid(item['responses'], model, tokenizer, device)
            
            if best_response:
                processed_data.append({
                    "question": item['question'],
                    "ground_truth": item['ground_truth'],
                    "selected_response": best_response['text'],
                    "is_hallucination": best_response['is_hallucination']
                })

        # Save results to the centralized output directory
        output_file = os.path.join(output_dir, f"{ds_name}_centroid.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Define absolute data paths relative to the project root
    TRIVIA_PATH = os.path.join(PROJECT_ROOT, "data", "triviaqa", "qa_generations_820_12_annotated.json")
    TRUTHFUL_PATH = os.path.join(PROJECT_ROOT, "data", "truthfulqa", "truthfulqa_with_hallucination_truth.json")
    
    try:
        run_selection_pipeline(TRIVIA_PATH, TRUTHFUL_PATH)
        print("\nSelection pipeline completed successfully.")
    except Exception as e:
        print(f"Critical error in pipeline: {e}")