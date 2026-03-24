#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-core}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
REQ_STAMP_FILE="$VENV_DIR/.requirements.sha256"

cd "$ROOT_DIR"

print_msg() {
  echo
  echo "$1"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "missing directory: $path"
    exit 1
  fi
}

pick_python_for_venv() {
  if command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"
    return
  fi
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo "no suitable python executable found"
  exit 1
}

ensure_venv() {
  if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    print_msg "venv found at $VENV_DIR"
    return
  fi

  local pybin
  pybin="$(pick_python_for_venv)"

  print_msg "creating venv with $pybin"
  "$pybin" -m venv "$VENV_DIR"
}

activate_venv() {
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    print_msg "activating venv"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
  else
    print_msg "venv already active: $VIRTUAL_ENV"
  fi
}

check_python_version() {
  python - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"python {sys.version.split()[0]} is too old, need >= 3.11")
print(f"python version ok: {sys.version.split()[0]}")
PY
}

compute_requirements_hash() {
  python - "$REQUIREMENTS_FILE" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
content = path.read_bytes()
print(hashlib.sha256(content).hexdigest())
PY
}

ensure_requirements_installed() {
  require_file "$REQUIREMENTS_FILE"

  local current_hash
  current_hash="$(compute_requirements_hash)"

  if [[ -f "$REQ_STAMP_FILE" ]]; then
    local installed_hash
    installed_hash="$(cat "$REQ_STAMP_FILE")"
    if [[ "$installed_hash" == "$current_hash" ]]; then
      print_msg "requirements already up to date"
      return
    fi
  fi

  print_msg "installing requirements"
  python -m pip install --upgrade pip
  python -m pip install -r "$REQUIREMENTS_FILE"
  echo "$current_hash" > "$REQ_STAMP_FILE"
}

check_hf_login_soft() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    if huggingface-cli whoami >/dev/null 2>&1; then
      print_msg "huggingface login detected"
    else
      print_msg "warning: huggingface-cli is installed but no login was detected"
      echo "if the llama model is gated and not cached on the server, run: huggingface-cli login"
    fi
  else
    print_msg "warning: huggingface-cli not found"
    echo "this is not always a problem if the model is already cached, but gated model access may fail"
  fi
}

run_config() {
  local config_name="$1"
  print_msg "running config: $config_name"
  python scripts/run_experiment.py --config "configs/$config_name"
}

get_experiment_name() {
  local config_path="$1"
  python - "$config_path" <<'PY'
import sys
from pathlib import Path
import yaml

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root))

from src.inside.utils import build_experiment_name

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg.get("experiment_name") or build_experiment_name(cfg))
PY
}

run_audit_qa820() {
  print_msg "auditing qa820 dataset"
  python scripts/audit_dataset.py \
    --dataset data/raw/qa_generations_820_12_annotated.json \
    --name qa820
}

run_audit_triviaqa() {
  print_msg "auditing triviaqa dataset"
  python scripts/audit_dataset.py \
    --dataset data/raw/triviaqa_annotated.json \
    --name triviaqa
}

run_compare_fc_n300() {
  local baseline_name
  local fc_name

  baseline_name="$(get_experiment_name "configs/baseline_qa820_n300.yaml")"
  fc_name="$(get_experiment_name "configs/fc_qa820_n300_p05.yaml")"

  print_msg "comparing baseline vs fc on n300"
  python scripts/compare_fc.py \
    --baseline "results/local_runs/$baseline_name" \
    --fc "results/local_runs/$fc_name" \
    --output-name "comparison_baseline_vs_fc_n300"
}

run_build_tables() {
  print_msg "building final tables"
  python scripts/build_tables.py
  python scripts/build_figures.py
}

check_common_project_files() {
  require_dir "configs"
  require_dir "scripts"
  require_dir "src"
  require_file "requirements.txt"

  require_file "scripts/run_experiment.py"
  require_file "scripts/audit_dataset.py"
  require_file "scripts/compare_fc.py"
  require_file "scripts/build_tables.py"
}

check_smoke_files() {
  require_file "configs/baseline_smoke.yaml"
  require_file "data/raw/qa_generations_820_12_annotated.json"
}

check_core_files() {
  require_file "data/raw/qa_generations_820_12_annotated.json"
  require_file "data/raw/triviaqa_annotated.json"

  require_file "configs/baseline_ablation_k_k5.yaml"
  require_file "configs/baseline_ablation_k_k10.yaml"
  require_file "configs/baseline_ablation_k_k20.yaml"
  require_file "configs/baseline_qa820_n100.yaml"
  require_file "configs/baseline_qa820_n300.yaml"
  require_file "configs/baseline_triviaqa_n100.yaml"
  require_file "configs/fc_qa820_n100_p01.yaml"
  require_file "configs/fc_qa820_n100_p02.yaml"
  require_file "configs/fc_qa820_n100_p05.yaml"
  require_file "configs/fc_qa820_n300_p05.yaml"
}

check_full_files() {
  check_core_files
  require_file "configs/baseline_qa820_n500.yaml"
  require_file "configs/baseline_qa820_n820.yaml"
}

check_extended_files() {
  check_full_files
  require_file "configs/fc_qa820_nall_p05.yaml"
  require_file "configs/fc_triviaqa_nall_p05.yaml"
}

run_mode_smoke() {
  check_smoke_files
  run_audit_qa820
  run_config "baseline_smoke.yaml"
}

run_mode_core() {
  check_core_files

  run_audit_qa820
  run_audit_triviaqa

  run_config "baseline_ablation_k_k5.yaml"
  run_config "baseline_ablation_k_k10.yaml"
  run_config "baseline_ablation_k_k20.yaml"

  run_config "baseline_qa820_n100.yaml"
  run_config "baseline_qa820_n300.yaml"
  run_config "baseline_triviaqa_n100.yaml"

  run_config "fc_qa820_n100_p01.yaml"
  run_config "fc_qa820_n100_p02.yaml"
  run_config "fc_qa820_n100_p05.yaml"
  run_config "fc_qa820_n300_p05.yaml"

  run_compare_fc_n300
}

run_mode_full() {
  check_full_files

  run_audit_qa820
  run_audit_triviaqa

  run_config "baseline_ablation_k_k5.yaml"
  run_config "baseline_ablation_k_k10.yaml"
  run_config "baseline_ablation_k_k20.yaml"

  run_config "baseline_qa820_n100.yaml"
  run_config "baseline_qa820_n300.yaml"
  run_config "baseline_qa820_n500.yaml"
  run_config "baseline_qa820_n820.yaml"
  run_config "baseline_triviaqa_n100.yaml"

  run_config "fc_qa820_n100_p01.yaml"
  run_config "fc_qa820_n100_p02.yaml"
  run_config "fc_qa820_n100_p05.yaml"
  run_config "fc_qa820_n300_p05.yaml"

  run_compare_fc_n300
  run_build_tables
}

run_mode_extended() {
  check_extended_files

  run_audit_qa820
  run_audit_triviaqa

  run_config "baseline_ablation_k_k5.yaml"
  run_config "baseline_ablation_k_k10.yaml"
  run_config "baseline_ablation_k_k20.yaml"

  run_config "baseline_qa820_n100.yaml"
  run_config "baseline_qa820_n300.yaml"
  run_config "baseline_qa820_n500.yaml"
  run_config "baseline_qa820_n820.yaml"
  run_config "baseline_triviaqa_n100.yaml"

  run_config "fc_qa820_n100_p01.yaml"
  run_config "fc_qa820_n100_p02.yaml"
  run_config "fc_qa820_n100_p05.yaml"
  run_config "fc_qa820_n300_p05.yaml"
  run_config "fc_qa820_nall_p05.yaml"
  run_config "fc_triviaqa_nall_p05.yaml"

  run_compare_fc_n300
  run_build_tables
}

print_msg "mode: $MODE"
print_msg "root: $ROOT_DIR"

check_common_project_files
ensure_venv
activate_venv
check_python_version
ensure_requirements_installed
check_hf_login_soft

case "$MODE" in
  smoke)
    run_mode_smoke
    ;;
  core)
    run_mode_core
    ;;
  full)
    run_mode_full
    ;;
  extended)
    run_mode_extended
    ;;
  *)
    echo "unknown mode: $MODE"
    echo "usage: ./run_all.sh [smoke|core|full|extended]"
    exit 1
    ;;
esac

print_msg "done"
