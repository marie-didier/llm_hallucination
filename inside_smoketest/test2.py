import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import Counter

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

def is_correct_answer(answer, gold_aliases):
    answer_norm = normalize_text(answer)
    for alias in gold_aliases:
        alias_norm = normalize_text(alias)
        if answer_norm == alias_norm:
            return True
        if alias_norm in answer_norm:
            return True
    return False

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
K = 10
TEMPERATURE = 0.5
TOP_P = 0.99
TOP_K = 5
MAX_NEW_TOKENS = 96
ALPHA = 0.001

SYSTEM_PROMPT = "Answer with one short sentence or a short phrase. Do not give lists. Do not explain."
MAX_NEW_TOKENS = 32

ITEMS = [
    {
        "question": "What is the capital of Japan?",
        "gold_aliases": ["tokyo"]
    },
    {
        "question": "Who wrote Pride and Prejudice?",
        "gold_aliases": ["jane austen", "austen"]
    },
    {
        "question": "What planet is known as the Red Planet?",
        "gold_aliases": ["mars"]
    },
    {
        "question": "What is the chemical symbol for gold?",
        "gold_aliases": ["au"]
    },
    {
        "question": "Who painted the Mona Lisa?",
        "gold_aliases": ["leonardo da vinci", "da vinci", "leonardo"]
    },
    {
        "question": "What is the largest ocean on Earth?",
        "gold_aliases": ["pacific ocean", "the pacific ocean", "pacific"]
    },
    {
        "question": "Who developed the theory of relativity?",
        "gold_aliases": ["albert einstein", "einstein"]
    },
    {
        "question": "What gas do plants absorb from the atmosphere?",
        "gold_aliases": ["carbon dioxide", "co2"]
    },
    {
        "question": "What is the capital of Australia?",
        "gold_aliases": ["canberra"]
    },
    {
        "question": "Who discovered penicillin?",
        "gold_aliases": ["alexander fleming", "fleming"]
    },
    {
        "question": "What is the square root of 144?",
        "gold_aliases": ["12", "twelve"]
    },
    {
        "question": "What is the boiling point of water at sea level in degrees Celsius?",
        "gold_aliases": ["100", "100 degrees celsius", "100 c", "100°c"]
    }
]

def compute_eigenscore(question, model, tokenizer):
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

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

    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    ).to(model.device)

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

            full_inputs = tokenizer(
                full_text,
                return_tensors="pt"
            ).to(model.device)

            outputs = model(
                **full_inputs,
                output_hidden_states=True,
                use_cache=False
            )

            hs = outputs.hidden_states[hidden_states_index]
            last_token_vec = hs[0, -1, :].float().cpu()
            sentence_embeddings.append(last_token_vec)

    sentence_embeddings = torch.stack(sentence_embeddings, dim=0)   # (K, d)

    # paper-faithful layout: Z in R^(d x K)
    Z = sentence_embeddings.T.to(torch.float64)                     # (d, K)

    # center across embedding dimensions, as in Z^T J_d Z
    Z_centered = Z - Z.mean(dim=0, keepdim=True)                   # subtract per-column mean
    Sigma = Z_centered.T @ Z_centered                              # (K, K)

    Sigma_reg = Sigma + ALPHA * torch.eye(Sigma.shape[0], dtype=Sigma.dtype)
    eigvals = torch.linalg.eigvalsh(Sigma_reg)
    eigenscore = torch.log(eigvals).mean().item()

    return {
        "question": question,
        "k": K,
        "alpha": ALPHA,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "hidden_size": hidden_size,
        "middle_layer_index": layer_index,
        "unique_generations": len(set(generations)),
        "eigenscore": eigenscore,
        "eigvals_min": float(eigvals.min().item()),
        "eigvals_max": float(eigvals.max().item()),
        "generations": generations
    }

def main():
    hf_token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token
    )

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

    for i, item in enumerate(ITEMS):
        question = item["question"]
        gold_aliases = item["gold_aliases"]

        print(f"\n===== question {i} =====")
        print("gold_aliases:", gold_aliases)
        print(question)

        result = compute_eigenscore(question, model, tokenizer)
        majority_answer = pick_majority_answer(result["generations"])
        majority_is_correct = is_correct_answer(majority_answer, gold_aliases)

        result["gold_aliases"] = gold_aliases
        result["majority_answer"] = majority_answer
        result["majority_is_correct"] = majority_is_correct

        print("majority_answer:", majority_answer)
        print("majority_is_correct:", majority_is_correct)
        print("unique_generations:", result["unique_generations"])
        print("eigvals_min:", result["eigvals_min"])
        print("eigvals_max:", result["eigvals_max"])
        print("eigenscore:", result["eigenscore"])

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/manual_8q_eigenscore.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = []
    for r in results:
        summary.append({
            "question": r["question"],
            "majority_answer": r["majority_answer"],
            "majority_is_correct": r["majority_is_correct"],
            "unique_generations": r["unique_generations"],
            "eigvals_min": r["eigvals_min"],
            "eigvals_max": r["eigvals_max"],
            "eigenscore": r["eigenscore"]
        })

    with open("outputs/manual_8q_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nsaved:")
    print("- outputs/manual_8q_eigenscore.json")
    print("- outputs/manual_8q_summary.json")

if __name__ == "__main__":
    main()
