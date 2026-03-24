import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
K = 10
TEMPERATURE = 0.5
TOP_P = 0.99
TOP_K = 5
MAX_NEW_TOKENS = 64

QUESTION = "What is the main cause of the French Revolution?"

def build_messages(question: str):
    return [
        {
            "role": "user",
            "content": question
        }
    ]

def main():
    print("loading model...")

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

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    layer_index = num_layers // 2
    hidden_states_index = layer_index + 1

    print("model_loaded:", MODEL_NAME)
    print("num_layers:", num_layers)
    print("hidden_size:", hidden_size)
    print("middle_layer_index:", layer_index)
    print("hidden_states_index_used:", hidden_states_index)

    messages = build_messages(QUESTION)

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

    print("\ngenerating samples...\n")

    with torch.no_grad():
        for i in range(K):
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

            print(f"[{i}] {gen_text}")

    print("\nextracting hidden states...\n")

    sentence_embeddings = []

    with torch.no_grad():
        for i, answer in enumerate(generations):
            full_messages = [
                {"role": "user", "content": QUESTION},
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

            print(f"embedding[{i}] shape: {tuple(last_token_vec.shape)}")

    sentence_embeddings = torch.stack(sentence_embeddings, dim=0)

    ALPHA = 0.001

    # sentence_embeddings has shape (K, hidden_size)
    # we center across the K generations
    X = sentence_embeddings.to(torch.float64)
    X_centered = X - X.mean(dim=0, keepdim=True)

    # Gram/covariance-like matrix across generations: shape (K, K)
    Sigma = X_centered @ X_centered.T

    # regularization
    K_eff = Sigma.shape[0]
    Sigma_reg = Sigma + ALPHA * torch.eye(K_eff, dtype=Sigma.dtype)

    # eigenvalues of the regularized matrix
    eigvals = torch.linalg.eigvalsh(Sigma_reg)

    # EigenScore = average log eigenvalue
    eigenscore = torch.log(eigvals).mean().item()

    print("\neigenscore diagnostics:")
    print("Sigma_shape:", tuple(Sigma.shape))
    print("Sigma_reg_shape:", tuple(Sigma_reg.shape))
    print("eigvals_shape:", tuple(eigvals.shape))
    print("eigvals_min:", eigvals.min().item())
    print("eigvals_max:", eigvals.max().item())
    print("eigenscore:", eigenscore)

    # optional: useful for understanding whether generations collapsed
    unique_generations = len(set(generations))
    print("unique_generations:", unique_generations)

    # save score and eigenvalues
    with open("outputs/eigenscore.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "question": QUESTION,
                "k": K,
                "alpha": ALPHA,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "unique_generations": unique_generations,
                "eigenscore": eigenscore,
                "eigvals": [float(x) for x in eigvals.tolist()]
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nfinal checks:")
    print("num_generations:", len(generations))
    print("sentence_embeddings_shape:", tuple(sentence_embeddings.shape))

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/generations.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "question": QUESTION,
                "generations": generations
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    torch.save(sentence_embeddings, "outputs/sentence_embeddings.pt")

    print("\nsaved:")
    print("- outputs/generations.json")
    print("- outputs/sentence_embeddings.pt")
    print("- outputs/eigenscore.json")

if __name__ == "__main__":
    main()
