import json
import pandas as pd
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lm_polygraph.stat_calculators.semantic_matrix import SemanticMatrixCalculator

class NLIModelWrapper:
    def __init__(self, model, tokenizer, batch_size, device):
        self.deberta = model
        self.deberta_tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device

        if hasattr(model.config, 'label2id'):
            print("Label mapping:", model.config.label2id)
            print("Id2label mapping:", model.config.id2label)
            label2id = model.config.label2id
            # Adiciona as chaves em caixa alta se não existirem
            if "ENTAILMENT" not in label2id and "entailment" in label2id:
                label2id["ENTAILMENT"] = label2id["entailment"]
            if "CONTRADICTION" not in label2id and "contradiction" in label2id:
                label2id["CONTRADICTION"] = label2id["contradiction"]
            # Finding right IDs
            if "ENTAILMENT" in label2id:
                self.entail_id = label2id["ENTAILMENT"]
                self.contra_id = label2id["CONTRADICTION"]
            elif "entailment" in label2id:
                self.entail_id = label2id["entailment"]
                self.contra_id = label2id["contradiction"]
            else:
                # Fallback para o padrão mais comum
                print("WARNING: Using default mapping (0=contradiction, 1=neutral, 2=entailment)")
                self.entail_id = 2
                self.contra_id = 0
        else:
            self.entail_id = 2
            self.contra_id = 0

        self.output_path = 'outputs/stats/'
        os.makedirs(self.output_path, exist_ok=True)

class NLICalculator:
    def __init__(self, dataset, model_name="cross-encoder/nli-deberta-v3-large", batch_size=32, device="cuda"):
        with open(dataset, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        self.nli_model = NLIModelWrapper(
            self.model, 
            self.tokenizer, 
            self.batch_size, 
            self.device
        )

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
        dependencies = {
            "sample_texts": [responses]
        }
        
        # Input text, no need
        texts = [""] * len(responses)
        
        # Chamar o calculator
        with torch.no_grad():
            matrixes = self.semantic_calculator(
                dependencies=dependencies,
                texts=texts,
                model=None,  # Não precisamos do modelo aqui
                max_new_tokens=100  # valor padrão
            )
        
        # Extrair matrizes
        matrix_entail = matrixes["semantic_matrix_entail"][0]  # primeira (e única) batch
        matrix_contra = matrixes["semantic_matrix_contra"][0]
        
        return matrix_entail, matrix_contra
    
    """
    Calculate matrices for all questions in the dataset
    """
    def calculate_nli_matrices(self):
        self.semantic_calculator = SemanticMatrixCalculator(self.nli_model)

        self.all_matrices = []
        
        for entry in tqdm(self.dataset, desc="Processing questions"):
            responses_data = entry.get("generations", entry.get("model_responses", []))

            responses_texts = []
            responses_labels = []
            for i, r in enumerate(responses_data):
                if isinstance(r, dict):
                    responses_texts.append(r.get("text"))
                    responses_labels.append(r.get("is_hallucination"))
                else:
                    responses_texts.append(r)
                    responses_labels.append(entry.get("hallucination_labels")[i])
                        
            if len(responses_texts) < 2:
                print(f"Warning: Question '{entry['question'][:50]}...' has only {len(responses_texts)} response(s). Skipping.")
                continue

            question_label = 1 if sum(responses_labels) > len(responses_labels)/2 else 0
            
            matrix_entail, matrix_contra = self.calculate_matrices(responses_texts)
            
            self.all_matrices.append({
                'question': entry.get("question"),
                'ground_truth': entry.get("ground_truth_reference", entry.get("gold_answer", "")),
                'matrix_entail': matrix_entail.tolist(),
                'matrix_contra': matrix_contra.tolist(),
                'question_label': question_label,
                'num_responses': len(responses_texts),
                'num_hallucinations': sum(responses_labels),
                'num_factual': len(responses_labels) - sum(responses_labels)
            })
        
        return self.all_matrices

    '''
    Save NLI scores
    '''
    def save_nli_scores(self, file_name="nli_scores.csv"):
        self.nli_scores.to_csv(f"{self.output_path}{file_name}", index=False)
        print(f"Saved NLI scores to {self.output_path}{file_name}")

    '''
    Save NLI matrices scores
    '''
    def save_nli_matrices_scores(self, file_name="nli_matrices_scores.json"):     
        with open(f"{self.output_path}{file_name}", 'w', encoding='utf-8') as f:
            json.dump(self.all_matrices, f, indent=2, ensure_ascii=False)

        print(f"Saved NLI matrices scores to {self.output_path}{file_name}")

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
