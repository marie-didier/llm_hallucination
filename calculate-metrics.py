import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =================CONFIGURATION=================
INPUT_FILE = "truthfulqa_samples.json"   # The file from the previous step
OUTPUT_FILE = "truthfulqa_with_metrics.json"
NLI_MODEL = "microsoft/deberta-large-mnli"
BATCH_SIZE = 16 # Optimization for speed
# ===============================================

def main():
    # 1. Setup Device (GPU is mandatory for speed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device} (If this says 'cpu', it will be very slow!)")

    # 2. Load NLI Model
    print(f"Loading NLI model: {NLI_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
    model.eval()

    # 3. Load Data
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Run generate_answers.py first!")
        return

    # --- HELPER FUNCTIONS ---
    def check_entailment_batch(premises, hypotheses):
        """
        Runs a batch of NLI checks. Returns a list of booleans (True if Entailment).
        """
        inputs = tokenizer(premises, hypotheses, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # DeBERTa-MNLI: 0=Contradiction, 1=Neutral, 2=Entailment
        # We check if index 2 is the maximum score
        predicted_classes = torch.argmax(logits, dim=1)
        return (predicted_classes == 2).cpu().numpy()

    def semantic_clustering(answers):
        """
        Greedy clustering of answers based on bidirectional entailment.
        """
        clusters = [] # List of lists (each inner list is a cluster of equivalent answers)
        
        # We process answers one by one
        for i, answer in enumerate(answers):
            # Skip empty answers
            if not answer.strip():
                continue
                
            matched = False
            
            # Compare against the first answer (representative) of each existing cluster
            if clusters:
                # Prepare batch inputs: Does New_Answer <-> Cluster_Rep ?
                reps = [c[0] for c in clusters]
                
                # Check 1: Answer -> Rep
                entails_rep = check_entailment_batch([answer] * len(reps), reps)
                
                # Check 2: Rep -> Answer (Only check those that passed step 1 to save time)
                # But for simplicity/batching, we can just run all reversed too
                rep_entails = check_entailment_batch(reps, [answer] * len(reps))
                
                # Find first match where BOTH are true
                for idx, (e1, e2) in enumerate(zip(entails_rep, rep_entails)):
                    if e1 and e2:
                        clusters[idx].append(answer)
                        matched = True
                        break
            
            if not matched:
                clusters.append([answer])
                
        return clusters

    def calculate_entropy(clusters, total_n):
        if total_n == 0: return 0.0
        probs = [len(c) / total_n for c in clusters]
        entropy = -sum([p * np.log(p) for p in probs]) # Natural log
        return entropy

    # --- MAIN LOOP ---
    print("Calculating Semantic Entropy...")
    
    for item in tqdm(data, desc="Processing"):
        samples = item.get("samples", [])
        
        if not samples:
            item["semantic_entropy"] = 0.0
            continue
            
        # 1. Cluster the samples
        clusters = semantic_clustering(samples)
        
        # 2. Calculate Entropy
        entropy = calculate_entropy(clusters, len(samples))
        
        # 3. Save Metrics
        item["semantic_entropy"] = entropy
        item["num_semantic_clusters"] = len(clusters)

        # Optional: Save cluster distribution for debugging
        # item["cluster_counts"] = [len(c) for c in clusters]

    # 4. Save Final File
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSuccess! Download '{OUTPUT_FILE}' to your local machine for analysis.")

if __name__ == "__main__":
    main()