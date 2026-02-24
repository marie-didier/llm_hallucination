import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

class EvalMethods:
    def __init__(self, y_true):
        self.y_true = y_true
        pass

    """
    Area Under the Risk-Coverage Curve.
    Risk = 1 - Accuracy at a specific coverage.
    """
    def calculate_aurc(self, y_scores):
        desc_score_indices = np.argsort(y_scores)[::-1] # Sort by confidence of hallucination
        y_true_sorted = self.y_true[desc_score_indices]
        
        '''
        In our case, Risk is defined as the error rate (hallucinations not caught)
        But usually AURC is for selective classification. 
        Here we simplify: Coverage is % of samples kept, Risk is % of errors in kept samples.
        '''
        n = len(self.y_true)
        risks = []
        for i in range(1, n + 1):
            # Coverage i/n: we take the i most 'uncertain' samples
            risk = np.mean(y_true_sorted[:i] == 0) # Error if we thought it was hallucination but it wasn't
            risks.append(risk)
        
        return np.mean(risks)

    """
    Expected Calibration Error.
    """
    def calculate_ece(self, y_scores, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
            # Find indices in the current bin
            in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(self.y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_scores[in_bin])
                ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
        return ece
    
    def compute_metrics(self, y_scores, name_score):
        # Compute metrics on the score
        auroc = roc_auc_score(self.y_true, y_scores)
        aurc = self.calculate_aurc(self.y_true, y_scores)
        ece = self.calculate_ece(self.y_true, y_scores)

        print(f"--- Results ---")
        print(f"AUROC: {auroc:.4f} (Higher is better)")
        print(f"AURC:  {aurc:.4f} (Lower is better)")
        print(f"ECE:   {ece:.4f} (Lower is better - measures calibration)")

        self.metrics.append({
            "method": name_score,
            "AUROC": auroc,
            "AURC": aurc,
            "ECE": ece
        })

        return auroc, aurc, ece

    # Plot ROC Curve for given score
    def plot_roc(self, y_scores, name_score):
        auroc, _, _ = self.compute_metrics(y_scores, name_score)

        fpr, tpr, _ = roc_curve(self.y_true, y_scores)
        plt.plot(fpr, tpr, label=f'NLI Method (AUC = {auroc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate (Factual flagged as Hallucination)')
        plt.ylabel('True Positive Rate (Hallucination correctly caught)')
        plt.title('ROC Curve for Hallucination Detection')
        plt.legend()
        plt.savefig(f'kle_roc_curve_{name_score}.png', dpi=150)
        plt.show()

    def save_eval_metrics(self, file_name="data/nli_comparison_metrics.csv"):
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(file_name, index=False)