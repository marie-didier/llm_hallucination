from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inside.utils import find_question, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the structure of an annotated dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Short dataset name used for output naming, for example qa820 or triviaqa.",
    )
    return parser.parse_args()


def load_examples(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "examples", "items", "rows"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"unsupported dataset structure in {path}")


def safe_len(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    return None


def audit_examples(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    key_counter = Counter()
    generation_len_counter = Counter()
    num_generations_declared_counter = Counter()
    num_generations_mismatch = 0
    examples_with_generations = 0
    examples_with_labels = 0
    examples_with_gold = 0
    missing_question_count = 0

    sample_rows = []

    for idx, example in enumerate(examples):
        for key in example.keys():
            key_counter[key] += 1

        question = find_question(example)
        if not question.strip():
            missing_question_count += 1

        generations = example.get("generations")
        gen_len = safe_len(generations)
        if gen_len is not None:
            examples_with_generations += 1
            generation_len_counter[gen_len] += 1

        if example.get("hallucination_labels") is not None:
            examples_with_labels += 1

        if example.get("gold_answer") is not None or example.get("answer") is not None or example.get("answers"):
            examples_with_gold += 1

        declared = example.get("num_generations")
        if declared is not None:
            num_generations_declared_counter[declared] += 1
            if gen_len is not None and declared != gen_len:
                num_generations_mismatch += 1

        if idx < 5:
            sample_rows.append(
                {
                    "id": example.get("id"),
                    "question_preview": question[:120],
                    "declared_num_generations": declared,
                    "actual_num_generations": gen_len,
                    "has_gold_answer": (
                        example.get("gold_answer") is not None
                        or example.get("answer") is not None
                        or bool(example.get("answers"))
                    ),
                    "has_hallucination_labels": example.get("hallucination_labels") is not None,
                    "keys": sorted(example.keys()),
                }
            )

    return {
        "num_examples": len(examples),
        "examples_with_generations": examples_with_generations,
        "examples_with_gold": examples_with_gold,
        "examples_with_hallucination_labels": examples_with_labels,
        "missing_question_count": missing_question_count,
        "declared_num_generations_counts": dict(sorted(num_generations_declared_counter.items())),
        "actual_generation_length_counts": dict(sorted(generation_len_counter.items())),
        "num_generations_mismatch_count": num_generations_mismatch,
        "key_presence_counts": dict(sorted(key_counter.items())),
        "sample_rows": sample_rows,
    }


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    output_path = REPO_ROOT / "data" / "audit" / f"{args.name}_audit.json"

    examples = load_examples(dataset_path)
    audit = audit_examples(examples)

    audit["dataset_name"] = args.name
    audit["dataset_path"] = str(dataset_path)
    save_json(audit, output_path)

    print(f"dataset: {dataset_path}")
    print(f"name: {args.name}")
    print(f"num_examples: {audit['num_examples']}")
    print(f"mismatch_count: {audit['num_generations_mismatch_count']}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
