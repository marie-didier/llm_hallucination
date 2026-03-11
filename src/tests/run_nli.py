from src.compute_nli import NLICalculator
from src.calculate_eval_metrics import EvalMethods
import numpy as np

nli_calc = NLICalculator("data/truthfulqa/truthfulqa_with_hallucination_truth.json")

try:
    all_labels, all_contradiction, all_neutral, all_entailment, y_true = nli_calc.get_nli_scores()
except:
    print("Calculating NLI...")
    nli_calc.calculate_nli()

    nli_calc.save_nli_scores()

eval_methods = EvalMethods(y_true=y_true, print_logs=True)

# Using contradiction score as confidence of hallucination
y_scores = np.array(all_contradiction) 
eval_methods.plot_roc(y_scores, 'nli', 'contradiction_metric')

# Compute 1 - entailment metrics
y_scores = 1 - np.array(all_entailment) 
eval_methods.plot_roc(y_scores, 'nli', 'entailment_metric')

# Compute ratio metrics
eps = 1e-6 # To avoid log(0)
y_scores = np.array(all_contradiction) / (np.array(all_neutral) +  eps)
eval_methods.plot_roc(y_scores, 'nli', 'ratio_metric')

# Compute product metrics
eps = 1e-6 # To avoid log(0)
y_scores = (1-np.array(all_entailment))* np.array(all_contradiction)
eval_methods.plot_roc(y_scores, 'nli', 'product_metric')

# Compute combinaison metrics
alpha = 1
beta = -0.5
y_scores = alpha*np.array(all_contradiction) +beta*np.array(all_neutral)
eval_methods.plot_roc(y_scores, 'nli', 'combination_metric')

eval_methods.save_eval_metrics()