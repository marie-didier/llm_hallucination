from src.scores.semantic_clustering import SemanticClustering
from src.calculate_eval_metrics import EvalMethods

sc = SemanticClustering(
    nli_matrices_json="data/nli_matrices_scores.json",
    dataset_path="data/truthfulqa_with_hallucination_truth.json"
)

sc.compute_scores()

sc.save_scores()

y_scores, y_true = sc.get_y_scores()

eval_methods = EvalMethods(y_true=y_true, print_logs=True)

eval_methods.plot_roc(y_scores, 'semantic_clustering', 'semantic_entropy')
