import os
import re
import json
import random
import torch
from collections import Counter
from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT = "Answer with one short sentence or a short phrase. Do not give lists. Do not explain."

INPUT_JSON = "qa_generations_820_12_annotated.json"

OUTPUT_RESULTS = "outputs/annotated100_inside_results.json"
OUTPUT_SUMMARY = "outputs/annotated100_inside_summary.json"
OUTPUT_REPRESENTATIVE = "outputs/annotated100_representative_cases.json"
OUTPUT_BRIEF = "outputs/annotated100_brief_table.json"

N_SAMPLES = 100
SEED = 42

K = 10
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

def compute_eigenscore(question, model, tokenizer):
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

    prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    generations = []

    with torch.no_grad():
        for _ in range(K):
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

            gen_ids = output[0][prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            generations.append(gen_text)

    sentence_embeddings = []

    with torch.no_grad():
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

            full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

            outputs = model(
                **full_inputs,
                output_hidden_states=True,
                use_cache=False
            )

            hs = outputs.hidden_states[hidden_states_index]
            last_token_vec = hs[0, -1, :].float().cpu()
            sentence_embeddings.append(last_token_vec)

    sentence_embeddings = torch.stack(sentence_embeddings, dim=0)   # (K, d)

    Z = sentence_embeddings.T.to(torch.float64)                     # (d, K)
    Z_centered = Z - Z.mean(dim=0, keepdim=True)                   # center each column over features
    Sigma = Z_centered.T @ Z_centered                              # (K, K)

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


def select_representative_cases(results, top_n=5):
    incorrect = [r for r in results if not r["majority_is_correct"]]
    correct = [r for r in results if r["majority_is_correct"]]

    incorrect_high = sorted(incorrect, key=lambda x: x["eigenscore"], reverse=True)[:top_n]
    incorrect_low = sorted(incorrect, key=lambda x: x["eigenscore"])[:top_n]
    correct_high = sorted(correct, key=lambda x: x["eigenscore"], reverse=True)[:top_n]

    return {
        "incorrect_high_score": incorrect_high,
        "incorrect_low_score": incorrect_low,
        "correct_high_score": correct_high
    }


def make_brief_table(results):
    table = []
    for r in results:
        table.append({
            "id": r["id"],
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "majority_answer": r["majority_answer"],
            "majority_is_correct": r["majority_is_correct"],
            "correctness_score_vs_gold": r["correctness_score_vs_gold"],
            "source_hallucination_rate": r["source_hallucination_rate"],
            "unique_generations": r["unique_generations"],
            "eigenscore": r["eigenscore"]
        })
    return table


def main():
    random.seed(SEED)

    hf_token = os.environ.get("HF_TOKEN", None)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    indices = list(range(len(data)))
    random.shuffle(indices)
    indices = indices[:N_SAMPLES]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    results = []

    for j, idx in enumerate(indices):
        item = data[idx]

        qid = item["id"]
        question = item["question"]
        gold_answer = item["gold_answer"]
        source_labels = item["hallucination_labels"]
        source_hallucination_rate = sum(source_labels) / len(source_labels)

        print(f"\n===== item {j+1}/{N_SAMPLES} | id={qid} =====")
        print("question:", question)

        out = compute_eigenscore(question, model, tokenizer)

        majority_answer = pick_majority_answer(out["generations"])
        corr_score = correctness_score_vs_gold(majority_answer, gold_answer)
        majority_is_correct = corr_score >= 0.5

        row = {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "source_num_generations_real": len(item["generations"]),
            "source_hallucination_rate": source_hallucination_rate,
            "source_majority_hallucinated": int(sum(source_labels) > len(source_labels) / 2),
            "our_generations": out["generations"],
            "majority_answer": majority_answer,
            "correctness_score_vs_gold": corr_score,
            "majority_is_correct": majority_is_correct,
            "unique_generations": out["unique_generations"],
            "eigenscore": out["eigenscore"],
            "eigvals_min": out["eigvals_min"],
            "eigvals_max": out["eigvals_max"]
        }
        results.append(row)

        print("gold_answer:", gold_answer)
        print("majority_answer:", majority_answer)
        print("correctness_score_vs_gold:", corr_score)
        print("majority_is_correct:", majority_is_correct)
        print("unique_generations:", out["unique_generations"])
        print("eigenscore:", out["eigenscore"])

    y_true_incorrect = [0 if r["majority_is_correct"] else 1 for r in results]
    y_score = [r["eigenscore"] for r in results]

    num_correct = sum(r["majority_is_correct"] for r in results)
    num_incorrect = len(results) - num_correct

    summary = {
        "model_name": MODEL_NAME,
        "n_samples": len(results),
        "seed": SEED,
        "k": K,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "alpha": ALPHA,
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "mean_eigenscore_correct": None,
        "mean_eigenscore_incorrect": None,
        "auroc_incorrect_vs_eigenscore": None
    }

    if num_correct > 0:
        summary["mean_eigenscore_correct"] = float(
            sum(r["eigenscore"] for r in results if r["majority_is_correct"]) / num_correct
        )

    if num_incorrect > 0:
        summary["mean_eigenscore_incorrect"] = float(
            sum(r["eigenscore"] for r in results if not r["majority_is_correct"]) / num_incorrect
        )

    if num_correct > 0 and num_incorrect > 0:
        summary["auroc_incorrect_vs_eigenscore"] = float(
            roc_auc_score(y_true_incorrect, y_score)
        )

    representative_cases = select_representative_cases(results, top_n=5)
    brief_table = make_brief_table(results)

    os.makedirs("outputs", exist_ok=True)

    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_REPRESENTATIVE, "w", encoding="utf-8") as f:
        json.dump(representative_cases, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_BRIEF, "w", encoding="utf-8") as f:
        json.dump(brief_table, f, ensure_ascii=False, indent=2)

    print("\n===== summary =====")
    print(summary)
    print("\nsaved:")
    print("-", OUTPUT_RESULTS)
    print("-", OUTPUT_SUMMARY)
    print("-", OUTPUT_REPRESENTATIVE)
    print("-", OUTPUT_BRIEF)


if __name__ == "__main__":
    main()
