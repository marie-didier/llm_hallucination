from src.scores.kernel_laguage_entropy import KleScore
from src.calculate_eval_metrics import EvalMethods

kle_calc = KleScore(dataset_path="data/truthfulqa/truthfulqa_with_hallucination_truth.json", nli_matrices_json="outputs/stats/nli_matrices_scores_truthfulqa.json")

# kle_calc = KleScore(dataset_path="data/triviaqa/triviaqa_annotated.json", nli_matrices_json="outputs/stats/nli_matrices_scores_triviaqa.json")

kle_calc.compute_kle()

kle_calc.save_kle_scores(file_name="nli_matrices_scores_truthfulqa.csv")
kle_calc.save_kle_scores(file_name="nli_matrices_scores_triviaqa.csv")

y_scores, y_true = kle_calc.get_y_scores()

eval_methods = EvalMethods(y_true=y_true, print_logs=True)

# Using contradiction score as confidence of hallucination
eval_methods.plot_roc(y_scores, 'kle', 'kernal_language_entropy_metric')