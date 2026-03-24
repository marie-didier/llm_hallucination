#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ACTIVE_PATHS=(
  "$ROOT_DIR/results/local_runs"
  "$ROOT_DIR/results/final/tables"
  "$ROOT_DIR/results/final/summaries"
  "$ROOT_DIR/results/final/cases"
  "$ROOT_DIR/results/final/figures"
)

DEEP_PATHS=(
  "$ROOT_DIR/inside_smoketest/outputs"
  "$ROOT_DIR/inside_fc/inside_fc/outputs"
)

FULL_REPRO_EXTRA=(
  "$ROOT_DIR/results/local_runs/_shared_fc_thresholds"
)

usage() {
  echo "Usage:"
  echo "  bash scripts/clean_outputs.sh --active"
  echo "  bash scripts/clean_outputs.sh --full-repro"
  echo "  bash scripts/clean_outputs.sh --deep"
  exit 1
}

remove_contents() {
  local target="$1"
  if [ -d "$target" ]; then
    echo "[clean] removing contents of: $target"
    find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  else
    echo "[clean] skipping missing dir: $target"
  fi
}

MODE="${1:-}"

case "$MODE" in
  --active)
    echo "[mode] active clean"
    for path in "${ACTIVE_PATHS[@]}"; do
      mkdir -p "$path"
      remove_contents "$path"
    done
    ;;

  --full-repro)
    echo "[mode] full reproducibility clean"
    for path in "${ACTIVE_PATHS[@]}"; do
      mkdir -p "$path"
      remove_contents "$path"
    done
    for path in "${FULL_REPRO_EXTRA[@]}"; do
      mkdir -p "$path"
      remove_contents "$path"
    done
    ;;

  --deep)
    echo "[mode] deep clean"
    for path in "${ACTIVE_PATHS[@]}"; do
      mkdir -p "$path"
      remove_contents "$path"
    done
    for path in "${FULL_REPRO_EXTRA[@]}"; do
      mkdir -p "$path"
      remove_contents "$path"
    done
    for path in "${DEEP_PATHS[@]}"; do
      mkdir -p "$path"
      remove_contents "$path"
    done
    ;;

  *)
    usage
    ;;
esac

mkdir -p \
  "$ROOT_DIR/results/final/tables" \
  "$ROOT_DIR/results/final/summaries" \
  "$ROOT_DIR/results/final/cases" \
  "$ROOT_DIR/results/final/figures" \
  "$ROOT_DIR/results/local_runs"

echo "[done] cleanup completed."
