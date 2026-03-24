import os
import re
import json
import random
import torch
from collections import Counter
from datasets import load_dataset
from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT = "Answer with one short sentence or a short phrase. Do not give lists. Do not explain."

K = 10
TEMPERATURE = 0.5
TOP_P = 0.99
TOP_K = 5
MAX_NEW_TOKENS = 32
ALPHA = 0.001

N_SAMPLES = 30
SEED = 42

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s°]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def pick_majority_answer(generations):
    normalized = [normalize_text(g) for g in generations]
    counts = Counter(normalized)
    majority_norm, _ = counts.most_common(1)[0]
    for g in generations:
        if normalize_text(g) == majority_norm:
            return g
    return generations[0]

def rouge_l_f1(pred, ref):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ref, pred)["rougeL"].fmeasure

def max_correctness_score(answer, aliases):
    answer_norm = normalize_text(answer)
    best = 0.0

    for alias in aliases:
        alias_norm = normalize_text(alias)

        if answer_norm == alias_norm:
            return 1.0

        score = rouge_l_f1(answer_norm, alias_norm)
        if score > best:
            best = score

    return best

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

def main():
    random.seed(SEED)

    hf_token = os.environ.get("HF_TOKEN", None)

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

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:N_SAMPLES]

    results = []

    for i, idx in enumerate(indices):
        item = ds[idx]
        question = item["question"]
        aliases = item["answer"]["aliases"]

        print(f"\n===== item {i} =====")
        print("question:", question)

        out = compute_eigenscore(question, model, tokenizer)

        majority_answer = pick_majority_answer(out["generations"])
        correctness_score = max_correctness_score(majority_answer, aliases)
        majority_is_correct = correctness_score >= 0.5

        row = {
            "question": question,
            "aliases": aliases,
            "majority_answer": majority_answer,
            "correctness_score": correctness_score,
            "majority_is_correct": majority_is_correct,
            "unique_generations": out["unique_generations"],
            "eigenscore": out["eigenscore"],
            "eigvals_min": out["eigvals_min"],
            "eigvals_max": out["eigvals_max"],
            "generations": out["generations"]
        }
        results.append(row)

        print("majority_answer:", majority_answer)
        print("correctness_score:", correctness_score)
        print("majority_is_correct:", majority_is_correct)
        print("unique_generations:", out["unique_generations"])
        print("eigenscore:", out["eigenscore"])

    y_true_incorrect = [0 if r["majority_is_correct"] else 1 for r in results]
    y_score = [r["eigenscore"] for r in results]

    num_correct = sum(r["majority_is_correct"] for r in results)
    num_incorrect = len(results) - num_correct

    summary = {
        "n_samples": len(results),
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "auroc_incorrect_vs_eigenscore": None
    }

    if num_correct > 0 and num_incorrect > 0:
        summary["auroc_incorrect_vs_eigenscore"] = float(roc_auc_score(y_true_incorrect, y_score))

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/triviaqa_30_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open("outputs/triviaqa_30_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== summary =====")
    print(summary)
    print("\nsaved:")
    print("- outputs/triviaqa_30_results.json")
    print("- outputs/triviaqa_30_summary.json")

if __name__ == "__main__":
    main()
