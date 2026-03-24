from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inside.pipeline import run_experiment_suite
from src.inside.utils import build_experiment_name, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run INSIDE experiments from a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"config at {path} must define a dictionary")

    return config


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    config = load_yaml_config(config_path)

    if "experiment_name" not in config or not config["experiment_name"]:
        config["experiment_name"] = build_experiment_name(config)

    summary = run_experiment_suite(config)

    final_experiment_name = config.get("experiment_name") or build_experiment_name(config)

    print()
    print(f"config: {config_path}")
    print(f"experiment_name: {final_experiment_name}")
    print("finished successfully")

    summary_path = Path("results") / "local_runs" / final_experiment_name / "summary.json"
    print(f"summary_path: {summary_path}")

    latest_summary_path = Path("results") / "local_runs" / "_latest_summary.json"
    save_json(summary, latest_summary_path)
    print(f"latest_summary_path: {latest_summary_path}")


if __name__ == "__main__":
    main()
