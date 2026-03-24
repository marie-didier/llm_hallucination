import os
import re
import json
import random
import unicodedata
import numpy as np
import torch

from collections import Counter
from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT = "Answer with one short sentence or a short phrase. Do not give lists. Do not explain."

INPUT_JSON = "qa_generations_820_12_annotated.json"

OUTPUT_SUMMARY = "outputs/stability_k20_summary_all.json"
OUTPUT_INDICES = "outputs/stability_k20_indices_all.json"

N_SAMPLES = "all"
SEED_DATASET = 42
GENERATION_SEEDS = [41, 42, 43]

K = 20
TEMPERATURE = 0.5
TOP_P = 0.99
TOP_K = 5
MAX_NEW_TOKENS = 32
ALPHA = 0.001


def strip_accents(text):
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def normalize_text(text):
    text = text.lower().strip()
    text = strip_accents(text)
    text = text.replace("metres", "meters")
    text = text.replace("milk wood", "milkwood")
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^\w\s°-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rouge_l_f1(pred, ref):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ref, pred)["rougeL"].fmeasure


def pick_majority_answer(generations):
    normalized = [normalize_text(g) for g in generations]
    counts = Counter(normalized)
    majority_norm, _ = counts.most_common(1)[0]

    for g in generations:
        if normalize_text(g) == majority_norm:
            return g
    return generations[0]


def correctness_score_vs_gold(answer, gold_answer):
    answer_norm = normalize_text(answer)
    gold_norm = normalize_text(gold_answer)

    if answer_norm == gold_norm:
        return 1.0

    if gold_norm in answer_norm:
        return 1.0

    if answer_norm in gold_norm:
        return 1.0

    answer_tokens = set(answer_norm.split())
    gold_tokens = set(gold_norm.split())
    if gold_tokens and gold_tokens.issubset(answer_tokens):
        return 1.0

    return rouge_l_f1(answer_norm, gold_norm)


def compute_aurc(majority_is_correct_list, eigenscores):
    correct = np.array(majority_is_correct_list, dtype=np.int32)
    scores = np.array(eigenscores, dtype=np.float64)

    if len(correct) == 0:
        return None

    order = np.argsort(scores)
    correct_sorted = correct[order]

    risks = []
    cum_correct = 0
    n = len(correct_sorted)

    for i in range(1, n + 1):
        cum_correct += correct_sorted[i - 1]
        accuracy = cum_correct / i
        risk = 1.0 - accuracy
        risks.append(risk)

    aurc = float(np.mean(risks))
    return aurc


def compute_pcc(correctness_scores, eigenscores):
    x = np.array(correctness_scores, dtype=np.float64)
    y = np.array(eigenscores, dtype=np.float64)

    if len(x) < 2:
        return None

    x_std = x.std()
    y_std = y.std()

    if x_std == 0.0 or y_std == 0.0:
        return None

    pcc = float(np.corrcoef(x, y)[0, 1])
    return pcc


def safe_mean(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def safe_std(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    mean_value = sum(valid) / len(valid)
    variance = sum((v - mean_value) ** 2 for v in valid) / len(valid)
    return float(variance ** 0.5)


def compute_eigenscore(question, model, tokenizer, generation_seed):
    num_layers = model.config.num_hidden_layers
    layer_index = num_layers // 2
    hidden_states_index = layer_index + 1

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    prompt_texts = [prompt_text] * K
    prompt_inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    prompt_len = prompt_inputs["input_ids"].shape[1]

    with torch.inference_mode():
        torch.manual_seed(generation_seed)
        torch.cuda.manual_seed_all(generation_seed)

        output = model.generate(
            **prompt_inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    gen_ids = output[:, prompt_len:]
    generations = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    generations = [g.strip() for g in generations]

    full_texts = []
    for answer in generations:
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_texts.append(full_text)

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    full_inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    tokenizer.padding_side = old_padding_side

    with torch.inference_mode():
        outputs = model(
            **full_inputs,
            output_hidden_states=True,
            use_cache=False
        )

    hs = outputs.hidden_states[hidden_states_index]
    last_idx = full_inputs["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(hs.shape[0], device=hs.device)
    sentence_embeddings = hs[batch_idx, last_idx, :].float().cpu()

    Z = sentence_embeddings.T.to(torch.float64)
    Z_centered = Z - Z.mean(dim=0, keepdim=True)
    Sigma = Z_centered.T @ Z_centered

    Sigma_reg = Sigma + ALPHA * torch.eye(Sigma.shape[0], dtype=Sigma.dtype)
    eigvals = torch.linalg.eigvalsh(Sigma_reg)
    eigenscore = torch.log(eigvals).mean().item()

    return {
        "generations": generations,
        "unique_generations": len(set(generations)),
        "eigenscore": eigenscore,
        "eigvals_min": float(eigvals.min().item()),
        "eigvals_max": float(eigvals.max().item())
    }


def run_for_seed(data, selected_indices, model, tokenizer, run_seed):
    print("\n====================")
    print(f"running generation_seed = {run_seed}")
    print("====================")

    results = []

    for j, idx in enumerate(selected_indices):
        item = data[idx]

        qid = item["id"]
        question = item["question"]
        gold_answer = item["gold_answer"]

        print(f"\n[seed={run_seed}] item {j+1}/{len(selected_indices)} | id={qid}")

        item_seed = run_seed * 100000 + j

        out = compute_eigenscore(
            question=question,
            model=model,
            tokenizer=tokenizer,
            generation_seed=item_seed
        )

        majority_answer = pick_majority_answer(out["generations"])
        corr_score = correctness_score_vs_gold(majority_answer, gold_answer)
        majority_is_correct = corr_score >= 0.5

        row = {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "majority_answer": majority_answer,
            "correctness_score_vs_gold": float(corr_score),
            "majority_is_correct": bool(majority_is_correct),
            "unique_generations": out["unique_generations"],
            "eigenscore": float(out["eigenscore"])
        }
        results.append(row)

        print("majority_answer:", majority_answer)
        print("majority_is_correct:", majority_is_correct)
        print("correctness_score_vs_gold:", corr_score)
        print("unique_generations:", out["unique_generations"])
        print("eigenscore:", out["eigenscore"])

    y_true_incorrect = [0 if r["majority_is_correct"] else 1 for r in results]
    y_score = [r["eigenscore"] for r in results]
    correct_binary = [1 if r["majority_is_correct"] else 0 for r in results]
    correctness_scores = [r["correctness_score_vs_gold"] for r in results]

    num_correct = sum(r["majority_is_correct"] for r in results)
    num_incorrect = len(results) - num_correct

    auroc = None
    if num_correct > 0 and num_incorrect > 0:
        auroc = float(roc_auc_score(y_true_incorrect, y_score))

    aurc = compute_aurc(correct_binary, y_score)
    pcc = compute_pcc(correctness_scores, y_score)

    summary = {
        "model_name": MODEL_NAME,
        "seed_dataset": SEED_DATASET,
        "generation_seed": run_seed,
        "n_samples": len(results),
        "k": K,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "alpha": ALPHA,
        "num_correct": int(num_correct),
        "num_incorrect": int(num_incorrect),
        "mean_eigenscore_correct": None,
        "mean_eigenscore_incorrect": None,
        "auroc_incorrect_vs_eigenscore": auroc,
        "aurc_incorrect_vs_eigenscore": aurc,
        "pcc_correctness_vs_eigenscore": pcc
    }

    if num_correct > 0:
        summary["mean_eigenscore_correct"] = float(
            sum(r["eigenscore"] for r in results if r["majority_is_correct"]) / num_correct
        )

    if num_incorrect > 0:
        summary["mean_eigenscore_incorrect"] = float(
            sum(r["eigenscore"] for r in results if not r["majority_is_correct"]) / num_incorrect
        )

    result_path = f"outputs/stability_k20_results_seed{run_seed}.json"
    summary_path = f"outputs/stability_k20_summary_seed{run_seed}.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nsummary:")
    print(summary)
    print("saved:", result_path)
    print("saved:", summary_path)

    return summary


def main():
    random.seed(SEED_DATASET)

    hf_token = os.environ.get("HF_TOKEN", None)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_items = len(data)

    if N_SAMPLES == "all" or N_SAMPLES is None:
        selected_indices = list(range(total_items))
        effective_n_samples = total_items
    else:
        indices = list(range(total_items))
        random.shuffle(indices)
        selected_indices = indices[:N_SAMPLES]
        effective_n_samples = len(selected_indices)

    os.makedirs("outputs", exist_ok=True)

    with open(OUTPUT_INDICES, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed_dataset": SEED_DATASET,
                "n_samples_requested": N_SAMPLES,
                "n_samples_effective": effective_n_samples,
                "selected_indices": selected_indices
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to("cuda")

    model.eval()

    all_summaries = []

    for run_seed in GENERATION_SEEDS:
        summary = run_for_seed(data, selected_indices, model, tokenizer, run_seed)
        all_summaries.append(summary)

    final_summary = {
        "model_name": MODEL_NAME,
        "seed_dataset": SEED_DATASET,
        "generation_seeds": GENERATION_SEEDS,
        "k": K,
        "n_samples": effective_n_samples,
        "mean_auroc_incorrect_vs_eigenscore": safe_mean(
            [s["auroc_incorrect_vs_eigenscore"] for s in all_summaries]
        ),
        "std_auroc_incorrect_vs_eigenscore": safe_std(
            [s["auroc_incorrect_vs_eigenscore"] for s in all_summaries]
        ),
        "mean_aurc_incorrect_vs_eigenscore": safe_mean(
            [s["aurc_incorrect_vs_eigenscore"] for s in all_summaries]
        ),
        "std_aurc_incorrect_vs_eigenscore": safe_std(
            [s["aurc_incorrect_vs_eigenscore"] for s in all_summaries]
        ),
        "mean_pcc_correctness_vs_eigenscore": safe_mean(
            [s["pcc_correctness_vs_eigenscore"] for s in all_summaries]
        ),
        "std_pcc_correctness_vs_eigenscore": safe_std(
            [s["pcc_correctness_vs_eigenscore"] for s in all_summaries]
        ),
        "runs": all_summaries
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print("\n====================")
    print("final stability summary")
    print("====================")
    print(final_summary)

    print("\nsaved:")
    print("-", OUTPUT_INDICES)
    print("-", OUTPUT_SUMMARY)
    for run_seed in GENERATION_SEEDS:
        print(f"- outputs/stability_k20_results_seed{run_seed}.json")
        print(f"- outputs/stability_k20_summary_seed{run_seed}.json")


if __name__ == "__main__":
    main()
