import json
from tqdm import tqdm
import numpy as np

from lm_polygraph.estimators import KernelLanguageEntropy

class KernelLanguageEntropy:
    def __init__(self, nli_json=None, kle_json='kle_results.json'):
        if nli_json:
            # Load dataset with NLI
            with open(self.nli_json, 'r', encoding='utf-8') as f:
                self.nli_dataset = json.load(f)
        else:
            # Call NLI Calculator
            pass

        self.kle_json = kle_json

    """
    Calculate KLE for each question
    """
    def processar_dataset(self, nli_calculator=None):
        # Initialize KLE method
        kle_method = KernelLanguageEntropy(t=0.3, normalize=True, scale=True, jitter=1e-6)
        
        results = []
        
        for item in tqdm(self.nli_dataset, desc="Processing questions..."):
            question = item['question']
            answers = item['samples']
            ground_truth = item['ground_truth']
            is_hallucination = item['is_hallucination']
            
            # Calculate semantic matrixes
            matrix_entail, matrix_contra = nli_calculator.calcular_matrizes(answers)
            
            # Prepare stats with KLE
            stats = {
                'semantic_matrix_entail': np.array([matrix_entail]),
                'semantic_matrix_contra': np.array([matrix_contra])
            }
            
            # Calculate Kernel Language Entropy
            kle_scores = kle_method(stats)
            kle_score = kle_scores[0]
            
            # Comparar com outras métricas existentes (opcional)
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'is_hallucination': is_hallucination,
                'kle_score': float(kle_score),
                'semantic_entropy_original': item.get('semantic_entropy'),
                'num_semantic_clusters': item.get('num_semantic_clusters'),
                'coherence_score': item.get('coherence_score'),
                'num_samples': len(answers)
            })
        
        # Save results
        with open(self.kle_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
        self.results = results