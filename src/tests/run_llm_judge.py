from src.scores.llm_as_a_judge import LlmAsAJudge

judge = LlmAsAJudge("data/truthfulqa_with_hallucination_truth.json")
y_scores = judge.run_ollama_inference()

# TO DO : finish testing