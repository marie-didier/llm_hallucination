"""
NLI-Based Centroid Selection
---------------------------
This script processes TriviaQA and TruthfulQA to extract one "best" answer per question.

Logic:
1. For each question, we compare all generated responses using an NLI model.
2. We build an agreement graph: if Response A entails Response B, A gets a point.
3. The response with the highest centrality (the semantic center) is selected.
4. Outputs are saved as individual files in 'outputs/stats/' for each dataset.
"""

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Grabbing the loader we built
from load_datasets import load_all_datasets

def get_nli_centroid(responses, model, tokenizer, device, threshold=0.5):
    """
    Find the response that best represents the consensus.
    We use NLI to see which answer is logically implied by the most others.
    """
    if not responses:
        return None
    if len(responses) == 1:
        return responses[0]

    texts = [r['text'] for r in responses]
    n = len(texts)
    scores = [0] * n
    
    # Generate all pairs for the NLI model
    pairs = []
    pair_indices = []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            pairs.append((texts[i], texts[j]))
            pair_indices.append((i, j))

    # Batching to avoid killing the VRAM
    batch_size = 32
    entailment_probs = []
    
    with torch.no_grad():
        for k in range(0, len(pairs), batch_size):
            batch = pairs[k : k + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            # Standard DeBERTa NLI: Label 0 is 'entailment'
            probs = torch.softmax(logits, dim=1)[:, 0].cpu().tolist()
            entailment_probs.extend(probs)

    # Scoring based on logical agreement
    for idx, prob in enumerate(entailment_probs):
        if prob > threshold:
            i, j = pair_indices[idx]
            scores[i] += 1 

    # Return the 'winner' of the semantic consensus
    best_idx = scores.index(max(scores))
    return responses[best_idx]

def run_selection_pipeline(trivia_path, truthful_path):
    """
    Clean the data and dump one JSON file per input dataset in outputs/stats.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup output folder
    output_dir = "outputs/stats"
    os.makedirs(output_dir, exist_ok=True)

    # Load the NLI heavy-lifter
    print(f"--- Firing up NLI model on {device} ---")
    model_name = "cross-encoder/nli-deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Get the standardized data
    all_data = load_all_datasets(trivia_path, truthful_path)
    all_data = {name: items[:10] for name, items in all_data.items()}
    # Process each dataset individually
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

        # Save specifically for this dataset
        output_file = os.path.join(output_dir, f"{ds_name}_centroid.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        
        print(f"Saved {len(processed_data)} entries to: {output_file}")

if __name__ == "__main__":
    # Check these paths before running
    TRIVIA_PATH = "../../data/triviaqa/qa_generations_820_12_annotated.json"
    TRUTHFUL_PATH = "../../data/truthfulqa/truthfulqa_with_hallucination_truth.json"
    
    try:
        run_selection_pipeline(TRIVIA_PATH, TRUTHFUL_PATH)
        print("\nAll datasets processed successfully.")
    except Exception as e:
        print(f"Pipeline crashed: {e}")