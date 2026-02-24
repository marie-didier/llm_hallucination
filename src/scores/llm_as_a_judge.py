# # Hallucination Detection: Llama 3.2 as a Judge
# We use Llama 3.2 to grade model responses on a scale of 0-10.
# Requirements: Have Ollama running locally with `ollama pull llama3.2`
# !pip install ollama scikit-learn numpy matplotlib

import json
import numpy as np
import ollama
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class LlmAsAJudge:
    def __init__(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    # Define the Prompting Logic
    def get_llama_score(self, question, reference, response):
        prompt = f"""
        [INST] You are a critical factual evaluator.
        Compare the 'Model Response' against the 'Ground Truth Reference'.
        Assign a score from 0 to 10 based on the probability of hallucination.
        
        - 0: The response is perfectly factual and matches the reference.
        - 10: The response is a total hallucination or contradicts the reference.

        Output ONLY the integer score.

        Question: {question}
        Ground Truth Reference: {reference}
        Model Response: {response}
        [/INST]
        Score (0-10):"""

        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        # Extract the first number found in the output
        content = response['message']['content'].strip()
        try:
            # Simple extraction to handle cases where Llama adds "The score is 5"
            score = int(''.join(filter(str.isdigit, content))[0])
            return score / 10.0
        except:
            return 0.5 # Neutral fallback

    # Run Inference
    def run_ollama_inference(self):
        y_true = []
        y_scores = []

        print("Judging responses with Llama 3.2...")
        for entry in self.data[:100]: # Testing on first 100 samples
            ref = entry["ground_truth_reference"]
            for resp in entry["model_responses"]:
                score = self.get_llama_score(entry["question"], ref, resp["text"])
                y_scores.append(score)
                y_true.append(resp["is_hallucination"])

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
