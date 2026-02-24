import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from itertools import combinations

class NLICalculator:
    def __init__(self, dataset, model_name="cross-encoder/nli-deberta-v3-large", batch_size=32, device="cuda"):
        with open(dataset, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    """
    Calculate entailment and contradiction matrixes for all answers combinations.
    """ 
    def calculate_nli(self):
        all_entailment = []
        all_neutral = []
        all_contradiction = []
        all_labels = []

        print(f"Processing {len(self.dataset)} questions...")

        for entry in tqdm(self.dataset, desc="Calculating NLI"):
            premise = entry["ground_truth_reference"]
            
            for resp in entry["model_responses"]:
                hypothesis = resp["text"]
                label = resp["is_hallucination"] # 1 if hallucinated, 0 if factual
                
                # Prepare input for Cross-Encoder
                inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    # Label mapping for this model: 0: contradiction, 1: neutral, 2: entailment
                    probs = torch.softmax(logits, dim=1)
                    contradiction_score = probs[0][0].item()
                    neutral_score = probs[0][1].item()
                    entailment_score = probs[0][2].item()
                    
                all_contradiction.append(contradiction_score)
                all_neutral.append(neutral_score)
                all_entailment.append(entailment_score)
                all_labels.append(label)

        self.nli_scores = pd.DataFrame({
            "entailment": all_entailment,
            "neutral": all_neutral,
            "contradiction": all_contradiction,
            "label": all_labels
        })

    """
        Calculate entailment and contradiction matrices for all response combinations.
    """
    def calculate_matrices(self, responses):
        n = len(responses)
        matrix_entail = np.zeros((n, n))
        matrix_contra = np.zeros((n, n))
        
        # Create all pairs (i, j) where i < j
        indices = list(combinations(range(n), 2))
        pairs = [(responses[i], responses[j]) for i, j in indices]
        
        # Process in batches
        with torch.no_grad():
            for k in tqdm(range(0, len(pairs), self.batch_size), desc="Calculating NLI between responses"):
                batch_pairs = pairs[k:k+self.batch_size]
                batch_indices = indices[k:k+self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    [p[0] for p in batch_pairs],
                    [p[1] for p in batch_pairs],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get probabilities
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                
                # Mapping for DeBERTa-MNLI: [contradiction, neutral, entailment]
                for idx, (i, j) in enumerate(batch_indices):
                    matrix_entail[i, j] = probs[idx, 2]  # entailment
                    matrix_entail[j, i] = matrix_entail[i, j]  # symmetric
                    
                    matrix_contra[i, j] = probs[idx, 0]  # contradiction
                    matrix_contra[j, i] = matrix_contra[i, j]  # symmetric
        
        # Diagonal (response compared to itself) = full entailment
        np.fill_diagonal(matrix_entail, 1.0)
        np.fill_diagonal(matrix_contra, 0.0)
        
        return matrix_entail, matrix_contra
    
    """
    Calculate matrices for all questions in the dataset
    """
    def calculate_nli_matrices(self):
        self.all_matrices = []
        
        for entry in tqdm(self.dataset, desc="Processing questions"):
            responses = entry["samples"]
            
            if len(responses) < 2:
                print(f"Warning: Question '{entry['question'][:50]}...' has only {len(responses)} response(s). Skipping.")
                continue
            
            matrix_entail, matrix_contra = self.calculate_matrices(responses)
            
            self.all_matrices.append({
                'question': entry['question'],
                'ground_truth': entry['ground_truth'],
                'matrix_entail': matrix_entail.tolist(),
                'matrix_contra': matrix_contra.tolist(),
                'label': entry['is_hallucination'],
                'num_responses': len(responses),
                'responses': responses
            })
        
        return self.all_matrices

    '''
    Save NLI scores
    '''
    def save_nli_scores(self, file_name="data/nli_scores.csv"):
        self.nli_scores.to_csv(file_name, index=False)
        print(f"Saved NLI scores to {file_name}")

    '''
    Save NLI matrices scores
    '''
    def save_nli_matrices_scores(self, file_name="data/nli_matrices_scores.csv"):     
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.all_matrices, f, indent=2, ensure_ascii=False)

        print(f"Saved NLI matrices scores to {file_name}")

    '''
    Retrieve NLI scores
    '''
    def get_nli_scores(self, file_name="data/nli_scores.csv"):
        self.nli_scores = pd.read_csv(file_name)
        all_labels = self.nli_scores["label"].values
        all_contradiction = self.nli_scores["contradiction"].values
        all_neutral = self.nli_scores["neutral"].values
        all_entailment = self.nli_scores["entailment"].values
        y_true = self.nli_scores["label"].values

        return all_labels, all_contradiction, all_neutral, all_entailment, y_true
    
    