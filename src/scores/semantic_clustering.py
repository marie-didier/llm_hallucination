import json
import math
import os
from collections import defaultdict
from tqdm import tqdm

from src.compute_nli import NLICalculator


class SemanticClustering:
    def __init__(self, nli_matrices_json=None, dataset_path=None, threshold=0.5):
        self.threshold = threshold
        try:
            with open(nli_matrices_json, 'r', encoding='utf-8') as f:
                self.nli_scores = json.load(f)
        except:
            if dataset_path:
                nli_calc = NLICalculator(dataset_path)
                self.nli_scores = nli_calc.calculate_nli_matrices()
                nli_calc.save_nli_matrices_scores()
            else:
                raise AttributeError("No dataset given to generate NLI stats.")

        self.output_path = 'outputs/semantic_clustering/'
        os.makedirs(self.output_path, exist_ok=True)

    def _build_clusters(self, entail_matrix):
        n = len(entail_matrix)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                if entail_matrix[i][j] > self.threshold and entail_matrix[j][i] > self.threshold:
                    union(i, j)

        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)
        return list(clusters.values())

    def _semantic_entropy(self, clusters, n):
        entropy = 0.0
        for cluster in clusters:
            p = len(cluster) / n
            entropy -= p * math.log(p)
        return entropy

    def compute_scores(self):
        self.results = []

        for item in tqdm(self.nli_scores, desc="Computing semantic clustering scores..."):
            entail_matrix = item['matrix_entail']
            question = item.get('question', 'Unknown')
            label = item.get('question_label', item.get('is_hallucination', None))
            n = item.get('num_responses', len(entail_matrix))

            clusters = self._build_clusters(entail_matrix)
            score = self._semantic_entropy(clusters, n)

            self.results.append({
                'question': question,
                'score': score,
                'label': label,
                'num_responses': n
            })

        return self.results

    def get_y_scores(self):
        y_scores = [r['score'] for r in self.results]
        y_true = [r['label'] for r in self.results]
        return y_scores, y_true

    def save_scores(self, file_name='outputs/semantic_clustering/semantic_clustering_scores.json'):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
