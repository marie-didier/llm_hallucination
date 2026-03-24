import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def save_jsonl(rows: List[Dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def normalize_basic_text(text: str) -> str:
    text = text.lower()
    text = strip_accents(text)
    text = text.replace("metres", "meters")
    text = text.replace("metre", "meter")
    text = text.replace("kilometres", "kilometers")
    text = text.replace("kilometre", "kilometer")
    text = re.sub(r"[\"'`´]", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = remove_articles(text)
    text = normalize_whitespace(text)
    return text


def token_set(text: str) -> set[str]:
    normalized = normalize_basic_text(text)
    if not normalized:
        return set()
    return set(normalized.split())


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def safe_std(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.std(values))


def select_subset_indices(
    n_total: int,
    n_samples: Optional[int],
    seed: int,
) -> List[int]:
    if n_samples is None or n_samples >= n_total:
        return list(range(n_total))
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    return sorted(indices[:n_samples])


def build_experiment_name(config: Dict[str, Any]) -> str:
    method = config.get("method", "baseline")
    dataset_name = config.get("dataset_name", "dataset")
    n_samples = config.get("n_samples", "all")
    k = config.get("k", "na")
    seed_dataset = config.get("seed_dataset", "na")

    parts = [
        method,
        dataset_name,
        f"n{n_samples}",
        f"k{k}",
        f"ds{seed_dataset}",
    ]

    generation_seeds = config.get("generation_seeds")
    generation_seed = config.get("generation_seed")

    if generation_seeds is not None:
        seed_part = "_".join(str(s) for s in generation_seeds)
        parts.append(f"gs{seed_part}")
    elif generation_seed is not None:
        parts.append(f"gs{generation_seed}")

    if config.get("use_feature_clipping", False):
        lower = config.get("fc_lower_percentile")
        upper = config.get("fc_upper_percentile")
        if lower is not None and upper is not None:
            lower_tag = str(lower).replace(".", "")
            upper_tag = str(upper).replace(".", "")
            parts.append(f"fc{lower_tag}_{upper_tag}")

    return "_".join(str(p) for p in parts)


def resolve_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    repo_root = get_repo_root()
    base_dir = repo_root / "results" / "local_runs"

    experiment_name = config.get("experiment_name")
    if not experiment_name:
        experiment_name = build_experiment_name(config)

    run_dir = base_dir / experiment_name
    ensure_dir(run_dir)

    return {
        "run_dir": run_dir,
        "summary_path": run_dir / "summary.json",
        "results_path": run_dir / "results.json",
        "indices_path": run_dir / "indices.json",
        "config_snapshot_path": run_dir / "config.json",
    }


def find_gold_answer(example: Dict[str, Any]) -> str:
    if "gold_answer" in example and example["gold_answer"] is not None:
        return str(example["gold_answer"])

    if "answer" in example and example["answer"] is not None:
        return str(example["answer"])

    if "answers" in example and example["answers"]:
        answers = example["answers"]
        if isinstance(answers, list):
            return str(answers[0])

    return ""


def find_question(example: Dict[str, Any]) -> str:
    if "question" in example and example["question"] is not None:
        return str(example["question"])
    return ""


def normalize_hallucination_labels(example: Dict[str, Any]) -> Any:
    return example.get("hallucination_labels")


def build_short_answer_prompt(question: str, instruction: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]
