import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

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

        for entry in tqdm(range(0, len(self.dataset), self.dataset), desc="Calculating NLI"):
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
    
    '''
    Save NLI scores
    '''
    def save_nli_scores(self, file_name="data/nli_scores.csv"):
        self.nli_scores.to_csv(file_name, index=False)
        print(f"Saved NLI scores to {file_name}")

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
    
    