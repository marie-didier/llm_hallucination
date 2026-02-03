import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

# =================CONFIGURATION=================
INPUT_FILE = "truthfulqa_with_metrics.json" 
OUTPUT_FILE = "truthfulqa_final_complete.json"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

BATCH_SIZE = 128
NUM_WORKERS = 4
# ===============================================

class NLIDataset(Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        return self.text_pairs[idx]

def collate_fn(batch, tokenizer):
    """
    Tokenizes the batch. 
    CRITICAL CHANGE: Returns tensors on CPU. Do not move to 'device' here.
    """
    premises = [p[0] for p in batch]
    hypotheses = [p[1] for p in batch]
    
    inputs = tokenizer(
        premises, 
        hypotheses, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=512
    )
    return inputs

def main():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Loading Model: {NLI_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
    model.eval()

    # 2. Load Data
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions.")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}.")
        return

    # 3. Pre-process
    print("Preparing inference batches...")
    
    inference_pairs = []
    metadata_map = [] 

    for idx, item in enumerate(data):
        samples = item.get("samples", [])
        ground_truth = item.get("ground_truth", "")
        
        if not samples: 
            continue

        # Coherence Pairs
        if len(samples) >= 2:
            anchor = samples[0]
            targets = samples[1:]
            for t in targets:
                inference_pairs.append((anchor, t))
                metadata_map.append((idx, "coherence"))
        
        # Factuality Pairs
        inference_pairs.append((ground_truth, samples[0]))
        metadata_map.append((idx, "factuality"))

    print(f"Total inference pairs to process: {len(inference_pairs)}")

    # 4. Create DataLoader
    dataset = NLIDataset(inference_pairs)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        # REMOVED 'device' argument from here
        collate_fn=lambda b: collate_fn(b, tokenizer),
        pin_memory=True if device == "cuda" else False
    )

    # 5. Massive Inference Loop
    all_probs = []
    
    print("Running Inference (Mixed Precision)...")
    
    with torch.no_grad():
        for batch_inputs in tqdm(dataloader):
            # MOVE TO GPU HERE (Main Process)
            batch_inputs = batch_inputs.to(device)

            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(**batch_inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

    # Flatten results
    if len(all_probs) > 0:
        flat_results = np.concatenate(all_probs, axis=0)
    else:
        flat_results = []

    # 6. Aggregate Results
    print("Aggregating results...")
    
    temp_storage = defaultdict(lambda: {"coherence_scores": [], "factuality": 0})
    
    for i, (data_idx, task_type) in enumerate(metadata_map):
        probs = flat_results[i]
        
        if task_type == "coherence":
            contradiction_score = probs[0] 
            temp_storage[data_idx]["coherence_scores"].append(contradiction_score)
            
        elif task_type == "factuality":
            predicted_label = np.argmax(probs)
            is_hallucination = 0 if predicted_label == 1 else 1
            temp_storage[data_idx]["factuality"] = is_hallucination

    # 7. Write back
    for idx, item in enumerate(data):
        samples = item.get("samples", [])
        if not samples: continue

        if len(samples) < 2:
            item["coherence_score"] = 1.0
        else:
            scores = temp_storage[idx]["coherence_scores"]
            if scores:
                item["coherence_score"] = float(1.0 - np.mean(scores))
            else:
                item["coherence_score"] = 1.0

        item["is_hallucination"] = int(temp_storage[idx]["factuality"])

    # 8. Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSuccess! Processed {len(data)} items. Saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()