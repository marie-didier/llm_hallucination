# %% [markdown]
# # Hallucination Detection Evaluation using NLI
# This notebook evaluates the NLI method on a labeled dataset using AUROC, AURC, and ECE.

# %%
# 1. Install and Import dependencies
# !pip install transformers torch scikit-learn numpy

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
# 2. Load the dataset generated previously
with open("data/truthfulqa_with_hallucination_truth.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# %%
# 3. Initialize the NLI Model (Cross-Encoder)
# We use DeBERTa-v3 as it is state-of-the-art for NLI tasks
model_name = "cross-encoder/nli-deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
# 4. Compute NLI Scores (Contradiction Probabilities)
all_entailment = []
all_neutral = []
all_contradiction = []
all_labels = []

print(f"Processing {len(data)} questions...")

for entry in data:
    premise = entry["ground_truth_reference"]
    
    for resp in entry["model_responses"]:
        hypothesis = resp["text"]
        label = resp["is_hallucination"] # 1 if hallucinated, 0 if factual
        
        # Prepare input for Cross-Encoder
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            # Label mapping for this model: 0: contradiction, 1: neutral, 2: entailment
            probs = torch.softmax(logits, dim=1)
            contradiction_score = probs[0][0].item()
            neutral_score = probs[0][1].item()
            entailment_score = probs[0][2].item()
            
        all_contradiction.append(contradiction_score)
        all_neutral.append(neutral_score)
        all_entailment.append(entailment_score)
        all_labels.append(label)

y_true = np.array(all_labels)


# %% 
# ## Save scores for later analysis

# Save entailment, neutral, contradiction scores with truth label
df_scores = pd.DataFrame({
    "entailment": all_entailment,
    "neutral": all_neutral,
    "contradiction": all_contradiction,
    "label": all_labels
})
df_scores.to_csv("data/nli_scores.csv", index=False)
print("Saved NLI scores to data/nli_scores.csv")

# %% [markdown]
# ## Run to avoid rerunning the NLI model every time
df_scores = pd.read_csv("data/nli_scores.csv")
all_labels = df_scores["label"].values
all_contradiction = df_scores["contradiction"].values
all_neutral = df_scores["neutral"].values
all_entailment = df_scores["entailment"].values
y_true = df_scores["label"].values

# %% [markdown]
# ## 5. Metrics Calculation

# %%
def calculate_aurc(y_true, y_scores):
    """
    Area Under the Risk-Coverage Curve.
    Risk = 1 - Accuracy at a specific coverage.
    """
    desc_score_indices = np.argsort(y_scores)[::-1] # Sort by confidence of hallucination
    y_true_sorted = y_true[desc_score_indices]
    
    # In our case, Risk is defined as the error rate (hallucinations not caught)
    # But usually AURC is for selective classification. 
    # Here we simplify: Coverage is % of samples kept, Risk is % of errors in kept samples.
    n = len(y_true)
    risks = []
    for i in range(1, n + 1):
        # Coverage i/n: we take the i most 'uncertain' samples
        risk = np.mean(y_true_sorted[:i] == 0) # Error if we thought it was hallucination but it wasn't
        risks.append(risk)
    
    return np.mean(risks)

def calculate_ece(y_true, y_scores, n_bins=10):
    """
    Expected Calibration Error.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        # Find indices in the current bin
        in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_scores[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return ece

# %%
# Compute contradiction metrics
metrics = []

y_scores = np.array(all_contradiction) # Using contradiction score as confidence of hallucination
auroc = roc_auc_score(y_true, y_scores)
aurc = calculate_aurc(y_true, y_scores)
ece = calculate_ece(y_true, y_scores)

print(f"--- Results ---")
print(f"AUROC: {auroc:.4f} (Higher is better)")
print(f"AURC:  {aurc:.4f} (Lower is better)")
print(f"ECE:   {ece:.4f} (Lower is better - measures calibration)")

metrics.append({
    "method": "Contradiction Score",
    "AUROC": auroc,
    "AURC": aurc,
    "ECE": ece
})

# %%
# Plot ROC Curve for contradiction score
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr, label=f'NLI Method (AUC = {auroc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (Factual flagged as Hallucination)')
plt.ylabel('True Positive Rate (Hallucination correctly caught)')
plt.title('ROC Curve for Hallucination Detection')
plt.legend()
plt.show()

# %%
# Compute 1 - entailment metrics
y_scores = 1 -np.array(all_entailment) 
auroc = roc_auc_score(y_true, y_scores)
aurc = calculate_aurc(y_true, y_scores)
ece = calculate_ece(y_true, y_scores)

print(f"--- Results ---")
print(f"AUROC: {auroc:.4f} (Higher is better)")
print(f"AURC:  {aurc:.4f} (Lower is better)")
print(f"ECE:   {ece:.4f} (Lower is better - measures calibration)")

metrics.append({
    "method": "1 - Entailment Score",
    "AUROC": auroc,
    "AURC": aurc,
    "ECE": ece
})

# %%
# Plot ROC Curve for 1 - entailment score
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr, label=f'NLI Method (AUC = {auroc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (Factual flagged as Hallucination)')
plt.ylabel('True Positive Rate (Hallucination correctly caught)')
plt.title('ROC Curve for Hallucination Detection')
plt.legend()
plt.show()

# %%
# Compute ratio metrics
eps = 1e-6 # To avoid log(0)
y_scores = np.array(all_contradiction) / (np.array(all_neutral) +  eps)
auroc = roc_auc_score(y_true, y_scores)
aurc = calculate_aurc(y_true, y_scores)
ece = calculate_ece(y_true, y_scores)

print(f"--- Results ---")
print(f"AUROC: {auroc:.4f} (Higher is better)")
print(f"AURC:  {aurc:.4f} (Lower is better)")
print(f"ECE:   {ece:.4f} (Lower is better - measures calibration)")

metrics.append({
    "method": "contradiction/neutral Score ",
    "AUROC": auroc,
    "AURC": aurc,
    "ECE": ece
})

# %%
# Plot ROC Curve for combined method
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr, label=f'NLI Method (AUC = {auroc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (Factual flagged as Hallucination)')
plt.ylabel('True Positive Rate (Hallucination correctly caught)')
plt.title('ROC Curve for Hallucination Detection')
plt.legend()
plt.show()

# %%
# Compute product metrics
eps = 1e-6 # To avoid log(0)
y_scores = (1-np.array(all_entailment))* np.array(all_contradiction)
auroc = roc_auc_score(y_true, y_scores)
aurc = calculate_aurc(y_true, y_scores)
ece = calculate_ece(y_true, y_scores)

print(f"--- Results ---")
print(f"AUROC: {auroc:.4f} (Higher is better)")
print(f"AURC:  {aurc:.4f} (Lower is better)")
print(f"ECE:   {ece:.4f} (Lower is better - measures calibration)")

metrics.append({
    "method": "(1-E)*C Score ",
    "AUROC": auroc,
    "AURC": aurc,
    "ECE": ece
})

# %%
# Plot ROC Curve for combined method
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr, label=f'NLI Method (AUC = {auroc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (Factual flagged as Hallucination)')
plt.ylabel('True Positive Rate (Hallucination correctly caught)')
plt.title('ROC Curve for Hallucination Detection')
plt.legend()
plt.show()

# %%
# Compute combinaison metrics
alpha = 1
beta = -0.5
y_scores = alpha*np.array(all_contradiction) +beta*np.array(all_neutral) 
auroc = roc_auc_score(y_true, y_scores)
aurc = calculate_aurc(y_true, y_scores)
ece = calculate_ece(y_true, y_scores)

print(f"--- Results ---")
print(f"AUROC: {auroc:.4f} (Higher is better)")
print(f"AURC:  {aurc:.4f} (Lower is better)")
print(f"ECE:   {ece:.4f} (Lower is better - measures calibration)")

metrics.append({
    "method": "C - 0.5N Score ",
    "AUROC": auroc,
    "AURC": aurc,
    "ECE": ece
})

# %%
# Plot ROC Curve for combined method
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr, label=f'NLI Method (AUC = {auroc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (Factual flagged as Hallucination)')
plt.ylabel('True Positive Rate (Hallucination correctly caught)')
plt.title('ROC Curve for Hallucination Detection')
plt.legend()
plt.show()

# %%
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("data/nli_comparison_metrics.csv", index=False)
# %%
