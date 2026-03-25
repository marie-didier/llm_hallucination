from src.scores.snne import SnneScore
from src.calculate_eval_metrics import EvalMethods


def sweep_taus(dataset_path, nli_json, base_name, similarity_mode, taus, alphas=None):
    print(f"\n===== sweep for {base_name} | mode={similarity_mode} =====")

    best_config = None
    best_auroc = -1.0

    if similarity_mode == 'hybrid':
        if alphas is None:
            alphas = [0.25, 0.5, 0.75]
    else:
        alphas = [None]

    for alpha in alphas:
        for tau in taus:
            if similarity_mode == 'hybrid':
                print(f"\nRunning SNNE-{similarity_mode} with tau={tau}, alpha={alpha}")
            else:
                print(f"\nRunning SNNE-{similarity_mode} with tau={tau}")

            # calcule les scores SNNE pour une configuration donnee
            snne_calc = SnneScore(
                dataset_path=dataset_path,
                nli_matrices_json=nli_json,
                tau=tau,
                similarity_mode=similarity_mode,
                alpha=alpha if alpha is not None else 0.5
            )

            snne_calc.compute_scores()
            y_scores, y_true = snne_calc.get_y_scores()

            eval_methods = EvalMethods(y_true=y_true, print_logs=True)

            name = f"{base_name}_{similarity_mode}_tau_{tau}"
            if alpha is not None:
                name += f"_alpha_{alpha}"

            # evalue la configuration courante
            auroc, aurc, ece = eval_methods.compute_metrics(y_scores, name)

            # garde la meilleure configuration selon l'auroc
            if auroc > best_auroc:
                best_auroc = auroc
                best_config = {
                    'tau': tau,
                    'alpha': alpha,
                    'similarity_mode': similarity_mode
                }

    print(f"\nBest config for {base_name}: {best_config} with AUROC={best_auroc:.4f}")
    return best_config


def run_one(dataset_path, nli_json, score_json, score_name, tau, similarity_mode, alpha=None):
    print(f"\n===== final run for {score_name} | mode={similarity_mode} | tau={tau} | alpha={alpha} =====")

    # relance le meilleur mode et sauvegarde les scores
    snne_calc = SnneScore(
        dataset_path=dataset_path,
        nli_matrices_json=nli_json,
        tau=tau,
        similarity_mode=similarity_mode,
        alpha=alpha if alpha is not None else 0.5
    )

    snne_calc.compute_scores()
    snne_calc.save_scores(file_name=score_json)

    y_scores, y_true = snne_calc.get_y_scores()

    # genere la courbe roc finale
    eval_methods = EvalMethods(y_true=y_true, print_logs=True)
    eval_methods.plot_roc(y_scores, 'snne', score_name)


def run_all_for_dataset(dataset_path, nli_json, dataset_name):
    taus = [0.1, 0.3, 0.5, 1.0, 2.0]
    alphas = [0.1, 0.2, 0.25, 0.3, 0.4]

    # cherche le meilleur tau pour le mode entail
    best_entail = sweep_taus(
        dataset_path=dataset_path,
        nli_json=nli_json,
        base_name=f"snne_{dataset_name}",
        similarity_mode='entail',
        taus=taus
    )

    run_one(
        dataset_path=dataset_path,
        nli_json=nli_json,
        score_json=f"outputs/snne/snne_scores_{dataset_name}_entail.json",
        score_name=f"snne_{dataset_name}_entail",
        tau=best_entail['tau'],
        similarity_mode='entail'
    )

    # cherche le meilleur tau pour le mode rougeL
    best_rouge = sweep_taus(
        dataset_path=dataset_path,
        nli_json=nli_json,
        base_name=f"snne_{dataset_name}",
        similarity_mode='rougeL',
        taus=taus
    )

    run_one(
        dataset_path=dataset_path,
        nli_json=nli_json,
        score_json=f"outputs/snne/snne_scores_{dataset_name}_rougeL.json",
        score_name=f"snne_{dataset_name}_rougeL",
        tau=best_rouge['tau'],
        similarity_mode='rougeL'
    )

    # explore les couples tau/alpha pour le mode hybride
    best_hybrid = sweep_taus(
        dataset_path=dataset_path,
        nli_json=nli_json,
        base_name=f"snne_{dataset_name}",
        similarity_mode='hybrid',
        taus=taus,
        alphas=alphas
    )

    run_one(
        dataset_path=dataset_path,
        nli_json=nli_json,
        score_json=f"outputs/snne/snne_scores_{dataset_name}_hybrid.json",
        score_name=f"snne_{dataset_name}_hybrid",
        tau=best_hybrid['tau'],
        similarity_mode='hybrid',
        alpha=best_hybrid['alpha']
    )


run_all_for_dataset(
    dataset_path="data/truthfulqa/truthfulqa_with_hallucination_truth.json",
    nli_json="outputs/stats/nli_matrices_scores_truthfulqa.json",
    dataset_name="truthfulqa"
)

run_all_for_dataset(
    dataset_path="data/triviaqa/triviaqa_100_10_annotated.json",
    nli_json="outputs/stats/nli_matrices_scores_triviaqa.json",
    dataset_name="triviaqa"
)