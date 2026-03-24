import os
import re
import json
import random
import unicodedata
import torch
from collections import Counter
from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT = "Answer with one short sentence or a short phrase. Do not give lists. Do not explain."

INPUT_JSON = "qa_generations_820_12_annotated.json"

OUTPUT_SUMMARY = "outputs/ablation_k_summary.json"
OUTPUT_INDICES = "outputs/ablation_k_indices.json"

N_SAMPLES = 100
SEED = 42

K_VALUES = [5, 10, 20]

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

def compute_eigenscore(question, model, tokenizer, k):
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

    # batched generation: same prompt repeated k times
    prompt_texts = [prompt_text] * k
    prompt_inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    prompt_len = prompt_inputs["input_ids"].shape[1]

    with torch.inference_mode():
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

    # build full sequences for hidden-state extraction
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

    # use right padding here to recover the last valid token easily
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

    hs = outputs.hidden_states[hidden_states_index]  # (k, T, d)

    last_idx = full_inputs["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(hs.shape[0], device=hs.device)
    sentence_embeddings = hs[batch_idx, last_idx, :].float().cpu()  # (k, d)

    Z = sentence_embeddings.T.to(torch.float64)   # (d, k)
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

def run_for_k(data, selected_indices, model, tokenizer, k):
    print(f"\n====================")
    print(f"running k = {k}")
    print(f"====================")

    results = []

    for j, idx in enumerate(selected_indices):
        item = data[idx]

        qid = item["id"]
        question = item["question"]
        gold_answer = item["gold_answer"]
        source_labels = item["hallucination_labels"]
        source_hallucination_rate = sum(source_labels) / len(source_labels)

        print(f"\n[k={k}] item {j+1}/{len(selected_indices)} | id={qid}")
        print("question:", question)

        out = compute_eigenscore(question, model, tokenizer, k)

        majority_answer = pick_majority_answer(out["generations"])
        corr_score = correctness_score_vs_gold(majority_answer, gold_answer)
        majority_is_correct = corr_score >= 0.5

        row = {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "source_hallucination_rate": source_hallucination_rate,
            "majority_answer": majority_answer,
            "correctness_score_vs_gold": corr_score,
            "majority_is_correct": majority_is_correct,
            "unique_generations": out["unique_generations"],
            "eigenscore": out["eigenscore"],
            "eigvals_min": out["eigvals_min"],
            "eigvals_max": out["eigvals_max"]
        }
        results.append(row)

        print("majority_answer:", majority_answer)
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
        "k": k,
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

    result_path = f"outputs/ablation_k_results_k{k}.json"
    summary_path = f"outputs/ablation_k_summary_k{k}.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nsummary for k={k}:")
    print(summary)
    print("saved:", result_path)
    print("saved:", summary_path)

    return summary


def main():
    random.seed(SEED)

    hf_token = os.environ.get("HF_TOKEN", None)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    indices = list(range(len(data)))
    random.shuffle(indices)
    selected_indices = indices[:N_SAMPLES]

    os.makedirs("outputs", exist_ok=True)

    with open(OUTPUT_INDICES, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "n_samples": N_SAMPLES,
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

    for k in K_VALUES:
        summary = run_for_k(data, selected_indices, model, tokenizer, k)
        all_summaries.append(summary)

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\n====================")
    print("final ablation summary")
    print("====================")
    for s in all_summaries:
        print(s)

    print("\nsaved:")
    print("-", OUTPUT_INDICES)
    print("-", OUTPUT_SUMMARY)
    for k in K_VALUES:
        print(f"- outputs/ablation_k_results_k{k}.json")
        print(f"- outputs/ablation_k_summary_k{k}.json")


if __name__ == "__main__":
    main()
