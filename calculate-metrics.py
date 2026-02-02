import json
import torch
import numpy as np
import argparse
import itertools
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =================CONFIGURATION=================
INPUT_FILE = "truthfulqa_samples.json"     # Output from generate_answers.py
OUTPUT_FILE = "truthfulqa_full_metrics.json"
NLI_MODEL = "microsoft/deberta-large-mnli"
BATCH_SIZE = 16 
# ===============================================

def main():
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # 2. Load NLI Model
    print(f"Loading NLI model: {NLI_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
    model.eval()

    # 3. Load Data
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions.")
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # --- HELPER: Run NLI on a batch of pairs ---
    def get_nli_scores(premises, hypotheses):
        """
        Returns raw probabilities for [Contradiction, Neutral, Entailment]
        """
        inputs = tokenizer(premises, hypotheses, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Convert logits to probabilities (Softmax)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # --- METRIC 1: SEMANTIC ENTROPY (Your Method) ---
    def calculate_semantic_entropy(answers):
        clusters = []
        for answer in answers:
            if not answer.strip(): continue
            
            matched = False
            # Compare against existing cluster representatives
            if clusters:
                reps = [c[0] for c in clusters]
                
                # Bi-directional Entailment Check
                # 1. Answer -> Rep
                probs_forward = get_nli_scores([answer] * len(reps), reps)
                # 2. Rep -> Answer
                probs_backward = get_nli_scores(reps, [answer] * len(reps))
                
                # Check if Entailment (Index 2) is the predicted class for BOTH
                # (We use argmax for hard clustering)
                entails_fwd = np.argmax(probs_forward, axis=1) == 2
                entails_bwd = np.argmax(probs_backward, axis=1) == 2
                
                for i, (f, b) in enumerate(zip(entails_fwd, entails_bwd)):
                    if f and b:
                        clusters[i].append(answer)
                        matched = True
                        break
            
            if not matched:
                clusters.append([answer])
        
        # Entropy Calculation
        total_n = sum(len(c) for c in clusters)
        if total_n == 0: return 0.0, 0
        
        probs = [len(c) / total_n for c in clusters]
        entropy = -sum([p * np.log(p) for p in probs]) # Natural log
        return entropy, len(clusters)

    # --- METRIC 2: COHERENCE (Marie's Method) ---
    def calculate_coherence(answers):
        """
        Coherence = 1 - Average(Probability of Contradiction)
        We check every pair of answers.
        """
        # Create all unique pairs (A, B) where A != B
        # For 50 samples, 50*49 = 2450 pairs. This is heavy but accurate.
        # OPTIMIZATION: If N > 10, we sample pairs to speed it up.
        pairs = list(itertools.combinations(answers, 2))
        
        if len(pairs) > 200:
            # Randomly sample 200 pairs to keep speed reasonable
            indices = np.random.choice(len(pairs), 200, replace=False)
            pairs = [pairs[i] for i in indices]
        
        if not pairs: return 0.0

        premises = [p[0] for p in pairs]
        hypotheses = [p[1] for p in pairs]
        
        # Process in batches
        contradiction_scores = []
        
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_p = premises[i:i+BATCH_SIZE]
            batch_h = hypotheses[i:i+BATCH_SIZE]
            
            probs = get_nli_scores(batch_p, batch_h)
            
            # Index 0 is Contradiction in DeBERTa-MNLI
            contradiction_scores.extend(probs[:, 0])
            
        avg_contradiction = np.mean(contradiction_scores)
        return 1.0 - avg_contradiction

    # --- MAIN LOOP ---
    print("Calculating metrics (This will take time on GPU)...")
    
    for item in tqdm(data, desc="Processing Questions"):
        samples = item.get("samples", [])
        
        # 1. Calculate Your Metric (Entropy)
        entropy, n_clusters = calculate_semantic_entropy(samples)
        item["semantic_entropy"] = float(entropy)
        item["num_clusters"] = int(n_clusters)
        
        # 2. Calculate Marie's Metric (Coherence)
        coherence = calculate_coherence(samples)
        item["coherence_score"] = float(coherence)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nDone! Download '{OUTPUT_FILE}' to analyze.")

if __name__ == "__main__":
    main()