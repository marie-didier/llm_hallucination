from src.scores.kernel_laguage_entropy import KleScore
from src.calculate_eval_metrics import EvalMethods

kle_calc = KleScore(dataset_path="data/truthfulqa_with_hallucination_truth.json", nli_matrices_json="data/nli_matrices_scores.json")

kle_calc.compute_kle()

kle_calc.save_kle_scores()

y_scores, y_true = kle_calc.get_y_scores()

eval_methods = EvalMethods(y_true=y_true, print_logs=True)

# Using contradiction score as confidence of hallucination
eval_methods.plot_roc(y_scores, 'kle', 'kernal_language_entropy_metric')