import json
from tqdm import tqdm
import numpy as np

from lm_polygraph.estimators import KernelLanguageEntropy

from src.compute_nli import NLICalculator

class KernelLanguageEntropy:
    def __init__(self, dataset_path=None, nli_matrices_json=None):
        self.nli_matrices_json = nli_matrices_json
        if nli_matrices_json:
            # Load dataset with NLI matrices
            with open(self.nli_matrices_json, 'r', encoding='utf-8') as f:
                self.nli_scores = json.load(f)
        else:
            if dataset_path:
                # Call NLI Calculator
                nli_calc = NLICalculator(dataset_path)
                self.nli_scores = nli_calc.calculate_nli_matrices()
                nli_calc.save_nli_matrices_scores()
            else:
                raise AttributeError("No dataset given to generate NLI stats.")

    """
    Calculate KLE for each question
    """
    def compute_kle(self, nli_calculator=None):
        # Initialize KLE method
        kle_method = KernelLanguageEntropy(t=0.3, normalize=True, scale=True, jitter=1e-6)
        
        self.results = []
        
        for item in tqdm(self.nli_scores, desc="Processing questions..."):
            # Extract matrices
            matrix_entail = np.array(item['matrix_entail'])
            matrix_contra = np.array(item['matrix_contra'])
            # Get metadata
            question = item.get('question', 'Unknown')
            label = item.get('label', item.get('is_hallucination', None))
            n_responses = item.get('num_responses', matrix_entail.shape[0])
            ground_truth = item.get('ground_truth', '')

            assert matrix_entail.shape[0] == matrix_entail.shape[1]
            assert n_responses >= 2
            
            # Prepare stats with KLE
            stats = {
                'semantic_matrix_entail': np.array([matrix_entail]),
                'semantic_matrix_contra': np.array([matrix_contra])
            }
            
            # Calculate Kernel Language Entropy
            kle_scores = kle_method(stats)
            kle_score = kle_scores[0]
            
            self.results.append({
                'question': question,
                'ground_truth': ground_truth,
                'kle_score': kle_score,
                'label': label,
                'num_responses': n_responses,
                'matrix_shape': matrix_entail.shape
            })
        
    def save_kle_scores(self, file_name='outputs/kle/kle_scores.json'):
        # Save results
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)