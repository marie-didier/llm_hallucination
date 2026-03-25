import json
import os
import re
import numpy as np
from tqdm import tqdm

from src.compute_nli import NLICalculator


class SnneScore:
    def __init__(
        self,
        dataset_path=None,
        nli_matrices_json=None,
        tau=1.0,
        similarity_mode='rougeL',
        alpha=0.5
    ):
        # facteur de temperature pour l'agregation SNNE
        self.tau = float(tau) if tau > 0 else 1.0

        # mode de similarite: entail, rougeL ou hybrid
        self.similarity_mode = similarity_mode

        # poids de rougeL dans le mode hybride
        self.alpha = float(alpha)

        self.dataset = None
        self.nli_scores = None

        # les modes entail et hybrid ont besoin des matrices NLI
        if self.similarity_mode in ['entail', 'hybrid']:
            try:
                with open(nli_matrices_json, 'r', encoding='utf-8') as f:
                    self.nli_scores = json.load(f)
            except FileNotFoundError:
                # si les matrices n'existent pas encore, on les calcule
                if dataset_path:
                    nli_calc = NLICalculator(dataset_path)
                    self.nli_scores = nli_calc.calculate_nli_matrices()

                    # on sauvegarde pour eviter de recalculer plus tard
                    if nli_matrices_json is not None:
                        nli_calc.save_nli_matrices_scores(nli_matrices_json)
                else:
                    raise AttributeError("No dataset given to generate NLI stats.")

        # les modes rougeL et hybrid utilisent aussi le dataset brut
        if self.similarity_mode in ['rougeL', 'hybrid']:
            if dataset_path is None:
                raise AttributeError("dataset_path is required for SNNE with ROUGE-L or hybrid mode.")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)

        # verification simple du mode choisi
        if self.similarity_mode not in ['entail', 'rougeL', 'hybrid']:
            raise ValueError(f"Unknown similarity_mode: {self.similarity_mode}")

        # dossier de sortie pour les scores SNNE
        self.output_path = 'outputs/snne/'
        os.makedirs(self.output_path, exist_ok=True)

        # expression reguliere pour une tokenisation simple
        self._tok_re = re.compile(r"[^a-z0-9]+", flags=re.IGNORECASE)

    def _logsumexp(self, x):
        # version stable numeriquement de log(sum(exp(x)))
        x = np.asarray(x, dtype=np.float64)
        m = np.max(x)
        return float(m + np.log(np.sum(np.exp(x - m))))

    def _compute_question_snne_from_matrix(self, sim_matrix):
        # convertit la matrice de similarite en score SNNE par question
        sim = np.array(sim_matrix, dtype=np.float64)
        n = sim.shape[0]

        # une seule reponse ne permet pas d'estimer une dispersion
        if n < 2:
            return 0.0

        row_terms = []
        for i in range(n):
            # on ignore la diagonale pour ne pas compter l'auto-similarite
            off_diag = np.delete(sim[i, :], i)

            # log-mean-exp sur les similarites de la reponse i
            lme = self._logsumexp(off_diag / self.tau) - np.log(n - 1)
            row_terms.append(lme)

        # plus le score est grand, plus la dispersion semantique est forte
        return -float(np.mean(row_terms))

    def _tokenize(self, text):
        # tokenisation minimale pour rougeL
        text = (text or "").lower()
        return [t for t in self._tok_re.split(text) if t]

    def _lcs_len(self, a, b):
        # calcule la longueur de la plus longue sous-sequence commune
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0

        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            ai = a[i - 1]
            for j in range(1, n + 1):
                if ai == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
            prev, curr = curr, prev

        return prev[n]

    def _rouge_l_f1(self, s1, s2):
        # calcule une similarite rougeL de type F1 entre deux textes
        t1 = self._tokenize(s1)
        t2 = self._tokenize(s2)

        if not t1 or not t2:
            return 0.0

        lcs = self._lcs_len(t1, t2)
        p = lcs / max(len(t1), 1)
        r = lcs / max(len(t2), 1)

        if p + r <= 0:
            return 0.0

        return float(2 * p * r / (p + r))

    def _build_rougeL_matrix(self, texts):
        # construit une matrice symetrique de similarites rougeL
        n = len(texts)
        sim = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            # la similarite d'un texte avec lui-meme vaut 1
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                s = self._rouge_l_f1(texts[i], texts[j])
                sim[i, j] = s
                sim[j, i] = s

        return sim

    def _extract_question_data(self, item):
        # extrait les textes et labels d'une question
        question = item.get('question', 'Unknown')

        if 'model_responses' in item:
            # format de type truthfulqa
            ground_truth = item.get('ground_truth_reference', '')
            responses = item.get('model_responses', [])

            texts = []
            labels = []

            for r in responses:
                text = r.get('text', '')
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                    labels.append(int(r.get('is_hallucination', 0)))

        elif 'generations' in item:
            # format de type triviaqa annote
            ground_truth = item.get('gold_answer', '')
            raw_texts = item.get('generations', [])
            raw_labels = item.get('hallucination_labels', [])

            texts = []
            labels = []

            for text, label in zip(raw_texts, raw_labels):
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                    labels.append(int(label))

        else:
            return None

        if len(texts) == 0:
            return None

        # proportion de generations hallucinées pour la question
        hallucination_fraction = float(np.mean(labels)) if len(labels) > 0 else 0.0

        # etiquette binaire de question utilisee pour l'evaluation
        question_label = 1 if hallucination_fraction > 0.5 else 0

        return {
            'question': question,
            'ground_truth': ground_truth,
            'texts': texts,
            'labels': labels,
            'question_label': question_label,
            'hallucination_fraction': hallucination_fraction,
            'num_responses': len(texts)
        }

    def _build_dataset_lookup(self):
        # cree une table pour retrouver rapidement les textes d'une question
        lookup = {}
        for item in self.dataset:
            parsed = self._extract_question_data(item)
            if parsed is None:
                continue

            # on utilise question + nombre de reponses comme cle simple
            key = (parsed['question'], parsed['num_responses'])
            lookup[key] = parsed
        return lookup

    def compute_scores(self):
        # calcule un score SNNE par question selon le mode choisi
        self.results = []

        if self.similarity_mode == 'entail':
            for item in tqdm(self.nli_scores, desc="Computing SNNE scores (entail)..."):
                matrix_entail = item['matrix_entail']
                question = item.get('question', 'Unknown')
                label = item.get('question_label', item.get('is_hallucination', None))
                n_responses = item.get('num_responses', len(matrix_entail))
                ground_truth = item.get('ground_truth', '')

                # symetrise la matrice d'entailment
                entail = np.array(matrix_entail, dtype=np.float64)
                sim = 0.5 * (entail + entail.T)

                score = self._compute_question_snne_from_matrix(sim)

                self.results.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'snne_score': score,
                    'label': label,
                    'num_responses': n_responses,
                    'tau': self.tau,
                    'similarity_mode': self.similarity_mode
                })

        elif self.similarity_mode == 'rougeL':
            for item in tqdm(self.dataset, desc="Computing SNNE scores (ROUGE-L)..."):
                parsed = self._extract_question_data(item)
                if parsed is None:
                    continue

                # calcule directement la matrice rougeL a partir des textes
                sim = self._build_rougeL_matrix(parsed['texts'])
                score = self._compute_question_snne_from_matrix(sim)

                self.results.append({
                    'question': parsed['question'],
                    'ground_truth': parsed['ground_truth'],
                    'snne_score': score,
                    'label': parsed['question_label'],
                    'hallucination_fraction': parsed['hallucination_fraction'],
                    'num_responses': parsed['num_responses'],
                    'tau': self.tau,
                    'similarity_mode': self.similarity_mode
                })

        elif self.similarity_mode == 'hybrid':
            # associe les questions du dataset brut aux matrices NLI
            dataset_lookup = self._build_dataset_lookup()

            for item in tqdm(self.nli_scores, desc="Computing SNNE scores (hybrid)..."):
                question = item.get('question', 'Unknown')
                matrix_entail = item['matrix_entail']
                n_responses = item.get('num_responses', len(matrix_entail))
                key = (question, n_responses)

                parsed = dataset_lookup.get(key, None)
                if parsed is None:
                    continue

                # matrice semantique issue du NLI
                entail = np.array(matrix_entail, dtype=np.float64)
                sim_entail = 0.5 * (entail + entail.T)

                # matrice lexicale issue de rougeL
                sim_rouge = self._build_rougeL_matrix(parsed['texts'])

                # securite si les dimensions ne correspondent pas
                if sim_entail.shape != sim_rouge.shape:
                    continue

                # combine les deux sources de similarite
                sim = self.alpha * sim_rouge + (1.0 - self.alpha) * sim_entail
                score = self._compute_question_snne_from_matrix(sim)

                self.results.append({
                    'question': parsed['question'],
                    'ground_truth': parsed['ground_truth'],
                    'snne_score': score,
                    'label': parsed['question_label'],
                    'hallucination_fraction': parsed['hallucination_fraction'],
                    'num_responses': parsed['num_responses'],
                    'tau': self.tau,
                    'similarity_mode': self.similarity_mode,
                    'alpha': self.alpha
                })

        return self.results

    def get_y_scores(self):
        # retourne les scores et labels utilisables pour l'evaluation
        valid_results = [r for r in self.results if r['label'] is not None]
        y_scores = [r['snne_score'] for r in valid_results]
        y_true = [r['label'] for r in valid_results]
        return y_scores, y_true

    def save_scores(self, file_name='outputs/snne/snne_scores.json'):
        # sauvegarde les resultats pour une analyse ulterieure
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)