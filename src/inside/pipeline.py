from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .feature_clipping import (
    ActivationCollector,
    TemporaryFeatureClipping,
    compute_thresholds_from_activations,
    load_thresholds,
    resolve_module_by_name,
    save_thresholds,
)
from .metrics import aggregate_seed_summaries, summarize_example_level_results
from .utils import (
    build_experiment_name,
    ensure_dir,
    find_question,
    get_repo_root,
    load_json,
    normalize_hallucination_labels,
    resolve_output_paths,
    save_json,
    select_subset_indices,
    set_global_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


DEFAULT_PROMPT_INSTRUCTION = (
    "Answer with one short sentence or a short phrase. "
    "Do not give lists. Do not explain."
)


def apply_default_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(config)

    cfg.setdefault("method", "baseline")
    cfg.setdefault("dataset_name", "dataset")
    cfg.setdefault("seed_dataset", 42)
    cfg.setdefault("k", 20)
    cfg.setdefault("temperature", 0.5)
    cfg.setdefault("top_p", 0.99)
    cfg.setdefault("top_k", 5)
    cfg.setdefault("max_new_tokens", 32)
    cfg.setdefault("alpha", 0.001)
    cfg.setdefault("embedding_layer", "middle")
    cfg.setdefault("embedding_token", "last")
    cfg.setdefault("eigenscore_mode", "legacy")
    cfg.setdefault("prompt_instruction", DEFAULT_PROMPT_INSTRUCTION)
    cfg.setdefault("use_feature_clipping", False)
    cfg.setdefault("device", "auto")
    cfg.setdefault("dtype", "auto")

    cfg.setdefault("correctness_threshold", 0.5)

    cfg.setdefault("fc_hook_location", "model.norm")
    cfg.setdefault("fc_threshold_source", "precomputed")
    cfg.setdefault("fc_calibration_seed", cfg["seed_dataset"])
    cfg.setdefault("calibration_n", 100)
    cfg.setdefault("calibration_k", 5)

    if "experiment_name" not in cfg or not cfg["experiment_name"]:
        cfg["experiment_name"] = build_experiment_name(cfg)

    return cfg


def load_examples_from_dataset(dataset_path: str | Path) -> List[Dict[str, Any]]:
    data = load_json(dataset_path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "examples", "items", "rows"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"unsupported dataset format in {dataset_path}")


def get_gold_candidates(example: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []

    if "gold_answer" in example and example["gold_answer"] is not None:
        if isinstance(example["gold_answer"], list):
            candidates.extend(str(x) for x in example["gold_answer"] if x is not None)
        else:
            candidates.append(str(example["gold_answer"]))

    if "answer" in example and example["answer"] is not None:
        if isinstance(example["answer"], list):
            candidates.extend(str(x) for x in example["answer"] if x is not None)
        else:
            candidates.append(str(example["answer"]))

    if "answers" in example and example["answers"]:
        if isinstance(example["answers"], list):
            candidates.extend(str(x) for x in example["answers"] if x is not None)

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            unique_candidates.append(cleaned)
            seen.add(cleaned)

    return unique_candidates


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = strip_accents(text)
    text = text.replace("metres", "meters")
    text = text.replace("milk wood", "milkwood")
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^\w\s°-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rouge_l_f1(pred: str, ref: str, scorer: rouge_scorer.RougeScorer) -> float:
    return float(scorer.score(ref, pred)["rougeL"].fmeasure)


def pick_majority_answer(generations: List[str]) -> str:
    normalized = [normalize_text(g) for g in generations]
    counts = Counter(normalized)
    majority_norm, _ = counts.most_common(1)[0]

    for g in generations:
        if normalize_text(g) == majority_norm:
            return g

    return generations[0]


def correctness_score_vs_gold_single(
    answer: str,
    gold_answer: str,
    scorer: rouge_scorer.RougeScorer,
) -> float:
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

    return rouge_l_f1(answer_norm, gold_norm, scorer)


def correctness_score_vs_gold(
    answer: str,
    gold_candidates: List[str],
    scorer: rouge_scorer.RougeScorer,
) -> Tuple[float, str]:
    if not gold_candidates:
        return 0.0, ""

    best_score = -1.0
    best_gold = gold_candidates[0]

    for gold in gold_candidates:
        score = correctness_score_vs_gold_single(answer, gold, scorer)
        if score > best_score:
            best_score = score
            best_gold = gold

    return float(best_score), best_gold


def choose_hidden_state_layer_index(hidden_states: Tuple[torch.Tensor, ...], embedding_layer: Any) -> int:
    if embedding_layer == "middle":
        n_transformer_layers = len(hidden_states) - 1
        layer_index = n_transformer_layers // 2
        hidden_states_index = layer_index + 1
        return hidden_states_index

    if isinstance(embedding_layer, int):
        if embedding_layer < 0:
            return len(hidden_states) + embedding_layer
        return embedding_layer

    raise ValueError(f"unsupported embedding_layer value: {embedding_layer}")


def get_last_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


def build_chat_texts(
    tokenizer: AutoTokenizer,
    batch_messages: List[List[Dict[str, str]]],
    add_generation_prompt: bool,
) -> List[str]:
    texts = []
    for messages in batch_messages:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        texts.append(text)
    return texts


def tokenize_chat_batch(
    tokenizer: AutoTokenizer,
    batch_messages: List[List[Dict[str, str]]],
    device: torch.device,
    add_generation_prompt: bool,
    padding_side: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    texts = build_chat_texts(
        tokenizer=tokenizer,
        batch_messages=batch_messages,
        add_generation_prompt=add_generation_prompt,
    )

    old_padding_side = tokenizer.padding_side
    if padding_side is not None:
        tokenizer.padding_side = padding_side

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    tokenizer.padding_side = old_padding_side

    return {k: v.to(device) for k, v in encoded.items()}


@torch.inference_mode()
def generate_k_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_messages: List[Dict[str, str]],
    k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    device: torch.device,
    generation_seed: Optional[int] = None,
) -> List[str]:
    batch_messages = [prompt_messages for _ in range(k)]
    batch = tokenize_chat_batch(
        tokenizer=tokenizer,
        batch_messages=batch_messages,
        device=device,
        add_generation_prompt=True,
        padding_side="left",
    )

    prompt_len = batch["input_ids"].shape[1]

    if generation_seed is not None:
        torch.manual_seed(int(generation_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(generation_seed))

    generated = model.generate(
        **batch,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_only = generated[:, prompt_len:]
    texts = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

    return [text.strip() for text in texts]


@torch.inference_mode()
def extract_sentence_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_conversations: List[List[Dict[str, str]]],
    device: torch.device,
    embedding_layer: Any,
    embedding_token: str,
) -> torch.Tensor:
    if embedding_token != "last":
        raise ValueError(f"unsupported embedding_token value: {embedding_token}")

    batch = tokenize_chat_batch(
        tokenizer=tokenizer,
        batch_messages=full_conversations,
        device=device,
        add_generation_prompt=False,
        padding_side="right",
    )

    outputs = model(
        **batch,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states
    layer_idx = choose_hidden_state_layer_index(hidden_states, embedding_layer)
    hs = hidden_states[layer_idx]

    last_positions = get_last_token_positions(batch["attention_mask"])
    batch_indices = torch.arange(hs.shape[0], device=hs.device)
    sentence_embeddings = hs[batch_indices, last_positions, :].float().cpu()

    return sentence_embeddings


def compute_eigenscore_from_embeddings(
    sentence_embeddings: torch.Tensor,
    alpha: float,
    mode: str,
) -> Tuple[float, torch.Tensor]:
    if mode == "legacy":
        Z = sentence_embeddings.T.to(torch.float64)
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        Sigma = Z_centered.T @ Z_centered

    elif mode == "centered":
        X = sentence_embeddings.to(torch.float64)
        X_centered = X - X.mean(dim=0, keepdim=True)
        Sigma = X_centered @ X_centered.T

    else:
        raise ValueError(f"unknown eigenscore_mode: {mode}")

    Sigma_reg = Sigma + float(alpha) * torch.eye(Sigma.shape[0], dtype=Sigma.dtype)
    eigvals = torch.linalg.eigvalsh(Sigma_reg)
    eigenscore = torch.log(eigvals).mean().item()

    return float(eigenscore), eigvals


def prepare_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    requested_device = str(config.get("device", "auto")).lower()
    requested_dtype = str(config.get("dtype", "auto")).lower()

    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requested device='cuda' but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if requested_dtype == "float32":
        model_dtype = torch.float32
    elif requested_dtype == "float16":
        model_dtype = torch.float16
    elif requested_dtype == "bfloat16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_kwargs: Dict[str, Any] = {
        "dtype": model_dtype,
    }

    if device.type == "cuda":
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        **model_kwargs,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def select_calibration_indices(
    n_total: int,
    evaluation_indices: List[int],
    calibration_n: int,
    seed: int,
) -> List[int]:
    evaluation_set = set(evaluation_indices)
    available = [idx for idx in range(n_total) if idx not in evaluation_set]

    if len(available) >= calibration_n:
        pool = available
    else:
        pool = list(range(n_total))

    relative_indices = select_subset_indices(
        n_total=len(pool),
        n_samples=calibration_n,
        seed=seed,
    )
    return [pool[i] for i in relative_indices]


def default_fc_thresholds_path(config: Dict[str, Any]) -> Path:
    repo_root = get_repo_root()
    base_dir = repo_root / "results" / "local_runs" / "_shared_fc_thresholds"
    ensure_dir(base_dir)

    dataset_name = config["dataset_name"]
    module_name = str(config["fc_hook_location"]).replace(".", "_")
    lower_tag = str(config["fc_lower_percentile"]).replace(".", "")
    upper_tag = str(config["fc_upper_percentile"]).replace(".", "")
    calibration_n = config["calibration_n"]
    calibration_k = config["calibration_k"]

    filename = (
        f"{dataset_name}_{module_name}"
        f"_calib{calibration_n}x{calibration_k}"
        f"_p{lower_tag}_{upper_tag}.pt"
    )

    return base_dir / filename


@torch.inference_mode()
def build_fc_thresholds_from_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Dict[str, Any]],
    indices: List[int],
    config: Dict[str, Any],
    device: torch.device,
) -> Any:
    target_module = resolve_module_by_name(model, config["fc_hook_location"])
    collector = ActivationCollector(store_dtype=torch.float32)
    collector.attach(target_module)

    try:
        iterator = tqdm(indices, desc="calibrating feature clipping", leave=False)

        for local_i, idx in enumerate(iterator):
            example = examples[idx]
            question = find_question(example)
            prompt_messages = [
                {"role": "system", "content": config["prompt_instruction"]},
                {"role": "user", "content": question},
            ]

            batch_messages = [prompt_messages for _ in range(int(config["calibration_k"]))]
            batch = tokenize_chat_batch(
                tokenizer=tokenizer,
                batch_messages=batch_messages,
                device=device,
                add_generation_prompt=True,
                padding_side="left",
            )

            item_seed = int(config["fc_calibration_seed"]) * 100000 + local_i
            torch.manual_seed(item_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(item_seed)

            model.generate(
                **batch,
                do_sample=True,
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        collector.detach()

    activations = collector.get_activations()

    thresholds = compute_thresholds_from_activations(
        activations=activations,
        module_name=config["fc_hook_location"],
        lower_percentile=config["fc_lower_percentile"],
        upper_percentile=config["fc_upper_percentile"],
    )

    return thresholds


def prepare_feature_clipping_thresholds(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Dict[str, Any]],
    evaluation_indices: List[int],
    config: Dict[str, Any],
    device: torch.device,
) -> Any:
    if not config.get("use_feature_clipping", False):
        return None

    if config.get("fc_threshold_source") != "precomputed":
        raise ValueError("this repo currently supports only fc_threshold_source='precomputed'")

    thresholds_path = config.get("fc_thresholds_path")
    if thresholds_path:
        thresholds_path = Path(thresholds_path)
    else:
        thresholds_path = default_fc_thresholds_path(config)

    if thresholds_path.exists():
        return load_thresholds(thresholds_path)

    calibration_seed = int(config["fc_calibration_seed"])
    calibration_indices = select_calibration_indices(
        n_total=len(examples),
        evaluation_indices=evaluation_indices,
        calibration_n=int(config["calibration_n"]),
        seed=calibration_seed,
    )

    set_global_seed(calibration_seed)

    thresholds = build_fc_thresholds_from_generation(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        indices=calibration_indices,
        config=config,
        device=device,
    )

    save_thresholds(thresholds, thresholds_path)
    return thresholds


def build_full_conversations(
    question: str,
    generations: List[str],
    instruction: str,
) -> List[List[Dict[str, str]]]:
    conversations = []
    for answer in generations:
        conversations.append(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
    return conversations


def process_single_example(
    example: Dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    config: Dict[str, Any],
    scorer: rouge_scorer.RougeScorer,
    item_generation_seed: int,
    fc_thresholds: Any = None,
) -> Dict[str, Any]:
    question = find_question(example)
    gold_candidates = get_gold_candidates(example)
    question_id = example.get("id")
    hallucination_labels = normalize_hallucination_labels(example)

    prompt_messages = [
        {"role": "system", "content": config["prompt_instruction"]},
        {"role": "user", "content": question},
    ]

    clip_fraction = None

    if fc_thresholds is None:
        generations = generate_k_responses(
            model=model,
            tokenizer=tokenizer,
            prompt_messages=prompt_messages,
            k=int(config["k"]),
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            top_k=int(config["top_k"]),
            max_new_tokens=int(config["max_new_tokens"]),
            device=device,
            generation_seed=item_generation_seed,
        )
    else:
        with TemporaryFeatureClipping(model, fc_thresholds) as fc_ctx:
            generations = generate_k_responses(
                model=model,
                tokenizer=tokenizer,
                prompt_messages=prompt_messages,
                k=int(config["k"]),
                temperature=float(config["temperature"]),
                top_p=float(config["top_p"]),
                top_k=int(config["top_k"]),
                max_new_tokens=int(config["max_new_tokens"]),
                device=device,
                generation_seed=item_generation_seed,
            )
            clip_fraction = fc_ctx.clip_fraction

    full_conversations = build_full_conversations(
        question=question,
        generations=generations,
        instruction=config["prompt_instruction"],
    )

    sentence_embeddings = extract_sentence_embeddings(
        model=model,
        tokenizer=tokenizer,
        full_conversations=full_conversations,
        device=device,
        embedding_layer=config["embedding_layer"],
        embedding_token=config["embedding_token"],
    )

    eigenscore, eigvals = compute_eigenscore_from_embeddings(
        sentence_embeddings=sentence_embeddings,
        alpha=float(config["alpha"]),
        mode=config["eigenscore_mode"],
    )

    majority_answer = pick_majority_answer(generations)
    corr_score, matched_gold = correctness_score_vs_gold(
        answer=majority_answer,
        gold_candidates=gold_candidates,
        scorer=scorer,
    )
    majority_is_correct = bool(corr_score >= float(config["correctness_threshold"]))

    row = {
        "id": question_id,
        "question": question,
        "gold_candidates": gold_candidates,
        "hallucination_labels": hallucination_labels,
        "generations": generations,
        "unique_generations": int(len(set(generations))),
        "majority_answer": majority_answer,
        "correctness_score_vs_gold": float(corr_score),
        "matched_gold": matched_gold,
        "majority_is_correct": majority_is_correct,
        "is_correct": majority_is_correct,
        "eigenscore": float(eigenscore),
        "eigvals_min": float(eigvals.min().item()),
        "eigvals_max": float(eigvals.max().item()),
        "clip_fraction": None if clip_fraction is None else float(clip_fraction),
    }

    return row


def run_single_seed_experiment(
    base_config: Dict[str, Any],
    examples: List[Dict[str, Any]],
    evaluation_indices: List[int],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    fc_thresholds: Any = None,
) -> Dict[str, Any]:
    if "generation_seed" not in base_config:
        raise ValueError("run_single_seed_experiment expects a config with generation_seed")

    config = apply_default_config(base_config)
    paths = resolve_output_paths(config)

    save_json(config, paths["config_snapshot_path"])
    save_json({"evaluation_indices": evaluation_indices}, paths["indices_path"])

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rows: List[Dict[str, Any]] = []
    iterator = tqdm(evaluation_indices, desc=config["experiment_name"])

    for j, idx in enumerate(iterator):
        item_seed = int(config["generation_seed"]) * 100000 + j
        example = examples[idx]

        row = process_single_example(
            example=example,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            scorer=scorer,
            item_generation_seed=item_seed,
            fc_thresholds=fc_thresholds,
        )
        rows.append(row)

    summary = summarize_example_level_results(rows, config)

    save_json(rows, paths["results_path"])
    save_json(summary, paths["summary_path"])

    return summary


def prepare_suite_config_for_seed(config: Dict[str, Any], generation_seed: int) -> Dict[str, Any]:
    seed_config = deepcopy(config)
    seed_config.pop("generation_seeds", None)
    seed_config["generation_seed"] = int(generation_seed)
    seed_config["experiment_name"] = build_experiment_name(seed_config)
    return seed_config


def run_experiment_suite(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = apply_default_config(config)

    if "dataset_path" not in cfg:
        raise ValueError("dataset_path is required in config")

    examples = load_examples_from_dataset(cfg["dataset_path"])
    evaluation_indices = select_subset_indices(
        n_total=len(examples),
        n_samples=cfg.get("n_samples"),
        seed=int(cfg["seed_dataset"]),
    )

    model, tokenizer, device = prepare_model_and_tokenizer(cfg)

    fc_thresholds = prepare_feature_clipping_thresholds(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        evaluation_indices=evaluation_indices,
        config=cfg,
        device=device,
    )

    if "generation_seeds" in cfg and cfg["generation_seeds"] is not None:
        seed_summaries = []
        for generation_seed in cfg["generation_seeds"]:
            seed_config = prepare_suite_config_for_seed(cfg, int(generation_seed))
            summary = run_single_seed_experiment(
                base_config=seed_config,
                examples=examples,
                evaluation_indices=evaluation_indices,
                model=model,
                tokenizer=tokenizer,
                device=device,
                fc_thresholds=fc_thresholds,
            )
            seed_summaries.append(summary)

        aggregate_config = deepcopy(cfg)
        aggregate_config["experiment_name"] = build_experiment_name(aggregate_config)
        aggregate_paths = resolve_output_paths(aggregate_config)

        save_json(aggregate_config, aggregate_paths["config_snapshot_path"])
        save_json({"evaluation_indices": evaluation_indices}, aggregate_paths["indices_path"])

        aggregated_summary = aggregate_seed_summaries(seed_summaries)
        aggregated_summary["experiment_name"] = aggregate_config["experiment_name"]
        save_json(aggregated_summary, aggregate_paths["summary_path"])

        return aggregated_summary

    if "generation_seed" not in cfg:
        raise ValueError("config must contain either generation_seed or generation_seeds")

    cfg["experiment_name"] = build_experiment_name(cfg)

    return run_single_seed_experiment(
        base_config=cfg,
        examples=examples,
        evaluation_indices=evaluation_indices,
        model=model,
        tokenizer=tokenizer,
        device=device,
        fc_thresholds=fc_thresholds,
    )
