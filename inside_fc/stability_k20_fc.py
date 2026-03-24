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


# ============================================================
# global config
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT = "Answer with one short sentence or a short phrase. Do not give lists. Do not explain."

INPUT_JSON = "qa_generations_820_12_annotated.json"

SEED_DATASET = 42
GENERATION_SEEDS = [41,42,43]

N_SAMPLES = 300

K = 20
TEMPERATURE = 0.5
TOP_P = 0.99
TOP_K = 5
MAX_NEW_TOKENS = 32
ALPHA = 0.001

# keep legacy by default to preserve comparability with your previous runs
EIGENSCORE_MODE = "legacy"   # "legacy" or "centered"

# ------------------------------------------------------------
# experiment mode
# ------------------------------------------------------------

EXPERIMENT_NAME = "baseline03_n300_s41_42_43"   # "baseline" or "fc_precomputed"
USE_FEATURE_CLIPPING = False

# ------------------------------------------------------------
# feature clipping config
# ------------------------------------------------------------

FC_HOOK_LOCATION = "model.model.norm"
FC_THRESHOLD_SOURCE = "precomputed"

# percentiles in [0, 100]
FC_LOWER_PERCENTILE = 0.5
FC_UPPER_PERCENTILE = 99.5

FC_CALIBRATION_JSON = "qa_generations_820_12_annotated.json"
FC_CALIBRATION_N = 100
FC_CALIBRATION_SEED = 123
FC_CALIBRATION_K = 5

# if calibration json is the same as evaluation json, try to exclude eval indices
FC_EXCLUDE_EVAL_FROM_CALIBRATION = True

FC_SAVE_THRESHOLDS = "inside_fc/outputs/fc_precomputed_thresholds_p03.pt"
FC_LOAD_THRESHOLDS = None

# ------------------------------------------------------------
# outputs
# ------------------------------------------------------------

OUTPUT_DIR = "inside_fc/outputs"
OUTPUT_SUMMARY = f"{OUTPUT_DIR}/stability_k20_{EXPERIMENT_NAME}_summary_all.json"
OUTPUT_INDICES = f"{OUTPUT_DIR}/stability_k20_{EXPERIMENT_NAME}_indices_all.json"


# ============================================================
# text utils
# ============================================================

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


# ============================================================
# metrics
# ============================================================

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


# ============================================================
# feature clipping helpers
# ============================================================

def get_fc_target_module(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    raise RuntimeError(
        f"could not find target module for FC hook using location={FC_HOOK_LOCATION}. "
        "expected model.model.norm for llama-like architectures."
    )


class PrecomputedFeatureClipper:
    def __init__(self, lower, upper):
        self.lower = lower.detach().cpu().to(torch.float32)
        self.upper = upper.detach().cpu().to(torch.float32)
        self.handle = None
        self.total_changed = 0
        self.total_elements = 0
        self.last_clip_fraction = None

    def reset_stats(self):
        self.total_changed = 0
        self.total_elements = 0
        self.last_clip_fraction = None

    def _hook_fn(self, module, inputs, output):
        x = output

        lower = self.lower.to(device=x.device, dtype=x.dtype)
        upper = self.upper.to(device=x.device, dtype=x.dtype)

        x_clipped = torch.maximum(torch.minimum(x, upper), lower)

        num_changed = (x_clipped != x).sum().item()
        total = x.numel()

        self.total_changed += int(num_changed)
        self.total_elements += int(total)

        if self.total_elements > 0:
            self.last_clip_fraction = float(self.total_changed / self.total_elements)
        else:
            self.last_clip_fraction = 0.0

        return x_clipped

    def attach(self, model):
        self.reset_stats()
        module = get_fc_target_module(model)
        self.handle = module.register_forward_hook(self._hook_fn)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def collect_generation_activations_for_question(
    model,
    tokenizer,
    question,
    generation_seed,
    k
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    prompt_texts = [prompt_text] * k
    prompt_inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    captured = []

    def capture_hook(module, inputs, output):
        x = output.detach()
        x = x.reshape(-1, x.shape[-1]).float().cpu()
        captured.append(x)
        return output

    module = get_fc_target_module(model)
    handle = module.register_forward_hook(capture_hook)

    try:
        with torch.inference_mode():
            torch.manual_seed(generation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(generation_seed)

            model.generate(
                **prompt_inputs,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    finally:
        handle.remove()

    if not captured:
        return None

    return torch.cat(captured, dim=0)


def build_fc_thresholds_precomputed(model, tokenizer, exclude_eval_indices=None):
    with open(FC_CALIBRATION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    same_dataset = os.path.abspath(FC_CALIBRATION_JSON) == os.path.abspath(INPUT_JSON)

    candidate_indices = list(range(len(data)))

    if (
        same_dataset
        and FC_EXCLUDE_EVAL_FROM_CALIBRATION
        and exclude_eval_indices is not None
    ):
        exclude_set = set(exclude_eval_indices)
        filtered = [idx for idx in candidate_indices if idx not in exclude_set]

        if len(filtered) >= FC_CALIBRATION_N:
            candidate_indices = filtered
            print(
                f"[fc calibration] using {len(candidate_indices)} candidates after excluding eval indices"
            )
        else:
            print(
                "[fc calibration] warning: not enough remaining items after excluding eval indices. "
                "falling back to full calibration pool."
            )

    rng = random.Random(FC_CALIBRATION_SEED)
    rng.shuffle(candidate_indices)
    selected = candidate_indices[:FC_CALIBRATION_N]

    all_activations = []

    print("\n====================")
    print("building precomputed FC thresholds")
    print("====================")
    print(f"calibration_json: {FC_CALIBRATION_JSON}")
    print(f"calibration_n: {len(selected)}")
    print(f"calibration_k: {FC_CALIBRATION_K}")
    print(f"lower_percentile: {FC_LOWER_PERCENTILE}")
    print(f"upper_percentile: {FC_UPPER_PERCENTILE}")

    for i, idx in enumerate(selected):
        question = data[idx]["question"]
        item_seed = FC_CALIBRATION_SEED * 100000 + i

        print(f"[fc calibration] item {i+1}/{len(selected)} | idx={idx}")

        acts = collect_generation_activations_for_question(
            model=model,
            tokenizer=tokenizer,
            question=question,
            generation_seed=item_seed,
            k=FC_CALIBRATION_K
        )

        if acts is not None and acts.numel() > 0:
            all_activations.append(acts)
            print(f"  collected tokens: {acts.shape[0]}")
        else:
            print("  warning: no activations collected")

    if not all_activations:
        raise RuntimeError("no activations collected for feature clipping thresholds")

    X = torch.cat(all_activations, dim=0).to(torch.float32)

    lower = torch.quantile(X, FC_LOWER_PERCENTILE / 100.0, dim=0)
    upper = torch.quantile(X, FC_UPPER_PERCENTILE / 100.0, dim=0)

    thresholds = {
        "lower": lower.cpu(),
        "upper": upper.cpu(),
        "hidden_dim": int(X.shape[1]),
        "n_tokens_total": int(X.shape[0]),
        "lower_percentile": float(FC_LOWER_PERCENTILE),
        "upper_percentile": float(FC_UPPER_PERCENTILE),
        "calibration_json": FC_CALIBRATION_JSON,
        "calibration_n": int(len(selected)),
        "calibration_k": int(FC_CALIBRATION_K),
        "calibration_seed": int(FC_CALIBRATION_SEED),
        "hook_location": FC_HOOK_LOCATION
    }

    return thresholds


def load_or_build_fc_thresholds(model, tokenizer, exclude_eval_indices=None):
    if not USE_FEATURE_CLIPPING:
        return None

    if FC_LOAD_THRESHOLDS is not None and os.path.exists(FC_LOAD_THRESHOLDS):
        print(f"loading precomputed thresholds from: {FC_LOAD_THRESHOLDS}")
        data = torch.load(FC_LOAD_THRESHOLDS, map_location="cpu")
        return data

    thresholds = build_fc_thresholds_precomputed(
        model=model,
        tokenizer=tokenizer,
        exclude_eval_indices=exclude_eval_indices
    )

    save_path = FC_SAVE_THRESHOLDS
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(thresholds, save_path)
    print(f"saved thresholds to: {save_path}")

    return thresholds


# ============================================================
# eigenscore helpers
# ============================================================

def compute_eigenscore_from_embeddings(sentence_embeddings):
    if EIGENSCORE_MODE == "legacy":
        Z = sentence_embeddings.T.to(torch.float64)
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        Sigma = Z_centered.T @ Z_centered

    elif EIGENSCORE_MODE == "centered":
        X = sentence_embeddings.to(torch.float64)
        X_centered = X - X.mean(dim=0, keepdim=True)
        Sigma = X_centered @ X_centered.T

    else:
        raise ValueError(f"unknown EIGENSCORE_MODE: {EIGENSCORE_MODE}")

    Sigma_reg = Sigma + ALPHA * torch.eye(Sigma.shape[0], dtype=Sigma.dtype)
    eigvals = torch.linalg.eigvalsh(Sigma_reg)
    eigenscore = torch.log(eigvals).mean().item()

    return eigenscore, eigvals


def compute_eigenscore(question, model, tokenizer, generation_seed, feature_clipper=None):
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

    clip_fraction = None

    if feature_clipper is not None:
        feature_clipper.attach(model)

    try:
        with torch.inference_mode():
            torch.manual_seed(generation_seed)
            if torch.cuda.is_available():
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

        if feature_clipper is not None:
            clip_fraction = feature_clipper.last_clip_fraction

    finally:
        if feature_clipper is not None:
            feature_clipper.detach()

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

    eigenscore, eigvals = compute_eigenscore_from_embeddings(sentence_embeddings)

    return {
        "generations": generations,
        "unique_generations": len(set(generations)),
        "eigenscore": eigenscore,
        "eigvals_min": float(eigvals.min().item()),
        "eigvals_max": float(eigvals.max().item()),
        "clip_fraction": clip_fraction
    }


# ============================================================
# experiment loop
# ============================================================

def run_for_seed(data, selected_indices, model, tokenizer, run_seed, feature_clipper=None):
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
            generation_seed=item_seed,
            feature_clipper=feature_clipper
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
            "eigenscore": float(out["eigenscore"]),
            "clip_fraction": None if out["clip_fraction"] is None else float(out["clip_fraction"])
        }
        results.append(row)

        print("majority_answer:", majority_answer)
        print("majority_is_correct:", majority_is_correct)
        print("correctness_score_vs_gold:", corr_score)
        print("unique_generations:", out["unique_generations"])
        print("eigenscore:", out["eigenscore"])
        if out["clip_fraction"] is not None:
            print("clip_fraction:", out["clip_fraction"])

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

    clip_values = [r["clip_fraction"] for r in results if r["clip_fraction"] is not None]

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "use_feature_clipping": bool(USE_FEATURE_CLIPPING),
        "fc_threshold_source": FC_THRESHOLD_SOURCE if USE_FEATURE_CLIPPING else None,
        "fc_hook_location": FC_HOOK_LOCATION if USE_FEATURE_CLIPPING else None,
        "fc_lower_percentile": FC_LOWER_PERCENTILE if USE_FEATURE_CLIPPING else None,
        "fc_upper_percentile": FC_UPPER_PERCENTILE if USE_FEATURE_CLIPPING else None,
        "eigenscore_mode": EIGENSCORE_MODE,
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
        "mean_clip_fraction": safe_mean(clip_values),
        "std_clip_fraction": safe_std(clip_values),
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

    result_path = os.path.join(OUTPUT_DIR, f"stability_k20_{EXPERIMENT_NAME}_results_seed{run_seed}.json")
    summary_path = os.path.join(OUTPUT_DIR, f"stability_k20_{EXPERIMENT_NAME}_summary_seed{run_seed}.json")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nsummary:")
    print(summary)
    print("saved:", result_path)
    print("saved:", summary_path)

    return summary


# ============================================================
# main
# ============================================================

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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_INDICES, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": EXPERIMENT_NAME,
                "use_feature_clipping": bool(USE_FEATURE_CLIPPING),
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
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to("cuda")

    model.eval()

    fc_thresholds = load_or_build_fc_thresholds(
        model=model,
        tokenizer=tokenizer,
        exclude_eval_indices=selected_indices
    )

    feature_clipper = None
    if USE_FEATURE_CLIPPING:
        feature_clipper = PrecomputedFeatureClipper(
            lower=fc_thresholds["lower"],
            upper=fc_thresholds["upper"]
        )

        print("\n====================")
        print("loaded FC thresholds")
        print("====================")
        print(f"hidden_dim: {fc_thresholds['hidden_dim']}")
        print(f"n_tokens_total: {fc_thresholds['n_tokens_total']}")
        print(f"lower_percentile: {fc_thresholds['lower_percentile']}")
        print(f"upper_percentile: {fc_thresholds['upper_percentile']}")
        print(f"hook_location: {fc_thresholds['hook_location']}")

    all_summaries = []

    for run_seed in GENERATION_SEEDS:
        summary = run_for_seed(
            data=data,
            selected_indices=selected_indices,
            model=model,
            tokenizer=tokenizer,
            run_seed=run_seed,
            feature_clipper=feature_clipper
        )
        all_summaries.append(summary)

    final_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "use_feature_clipping": bool(USE_FEATURE_CLIPPING),
        "fc_threshold_source": FC_THRESHOLD_SOURCE if USE_FEATURE_CLIPPING else None,
        "fc_hook_location": FC_HOOK_LOCATION if USE_FEATURE_CLIPPING else None,
        "fc_lower_percentile": FC_LOWER_PERCENTILE if USE_FEATURE_CLIPPING else None,
        "fc_upper_percentile": FC_UPPER_PERCENTILE if USE_FEATURE_CLIPPING else None,
        "fc_calibration_json": FC_CALIBRATION_JSON if USE_FEATURE_CLIPPING else None,
        "fc_calibration_n": FC_CALIBRATION_N if USE_FEATURE_CLIPPING else None,
        "fc_calibration_seed": FC_CALIBRATION_SEED if USE_FEATURE_CLIPPING else None,
        "fc_calibration_k": FC_CALIBRATION_K if USE_FEATURE_CLIPPING else None,
        "eigenscore_mode": EIGENSCORE_MODE,
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
        "mean_clip_fraction": safe_mean(
            [s["mean_clip_fraction"] for s in all_summaries]
        ),
        "std_clip_fraction": safe_std(
            [s["mean_clip_fraction"] for s in all_summaries]
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
        print(f"- {OUTPUT_DIR}/stability_k20_{EXPERIMENT_NAME}_results_seed{run_seed}.json")
        print(f"- {OUTPUT_DIR}/stability_k20_{EXPERIMENT_NAME}_summary_seed{run_seed}.json")


if __name__ == "__main__":
    main()
