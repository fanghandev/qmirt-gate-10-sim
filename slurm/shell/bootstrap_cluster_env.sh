#!/usr/bin/env bash
set -euo pipefail

check_python_imports() {
  local python_bin="${1:-python}"
  "$python_bin" - <<'PY'
import importlib
import sys

required = ["opengate", "scipy", "pandas"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if missing:
    print("Missing Python dependencies:")
    for item in missing:
        print(f"  - {item}")
    sys.exit(1)

print("Python environment check passed:")
for name in required:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"  {name}: {version}")
PY
}

list_missing_packages() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import importlib

required = ["opengate", "scipy", "pandas"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

print(" ".join(missing))
PY
}

install_python_packages() {
  local python_bin="$1"
  echo "Installing Python packages into $($python_bin -c 'import sys; print(sys.executable)')"
  "$python_bin" -m pip install --upgrade pip

  # Prefer binary wheels for numeric stack to avoid source builds on older GCC toolchains.
  "$python_bin" -m pip install --only-binary=:all: "numpy<2.0" "scipy<1.13" "pandas<2.3"
  "$python_bin" -m pip install opengate
}

acquire_bootstrap_lock() {
  local lock_root="${OPENGATE_BOOTSTRAP_LOCK_DIR:-${HOME}/.cache}"
  local lock_dir="${lock_root}/opengate-bootstrap-${CONDA_ENV_NAME:-opengate}"
  local lock_pid_file="${lock_dir}/pid"

  mkdir -p "$lock_root"

  while true; do
    if mkdir "$lock_dir" 2>/dev/null; then
      printf '%s\n' "$$" > "$lock_pid_file"
      trap "rm -rf '$lock_dir'" EXIT INT TERM
      return 0
    fi

    if [[ -f "$lock_pid_file" ]]; then
      local lock_pid
      lock_pid="$(cat "$lock_pid_file" 2>/dev/null || true)"
      if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
        rm -rf "$lock_dir"
        continue
      fi
    fi

    echo "Waiting for bootstrap lock at $lock_dir..."
    sleep 2
  done
}

if [[ -n "${SLURM_ENV_SCRIPT:-}" ]]; then
  if [[ -f "$SLURM_ENV_SCRIPT" ]]; then
    # shellcheck disable=SC1090
    source "$SLURM_ENV_SCRIPT"
  else
    echo "Error: SLURM_ENV_SCRIPT points to a missing file: $SLURM_ENV_SCRIPT" >&2
    exit 1
  fi
fi

export PATH="/PHShome/${USER}/.local/bin:${PATH}"

if ! command -v module >/dev/null 2>&1; then
  echo "Error: the 'module' command is unavailable; ERISTwo jobs should load miniforge3 via modules." >&2
  exit 1
fi

module load miniforge3

gcc_module="${SLURM_GCC_MODULE:-gcc}"
if [[ -n "$gcc_module" ]]; then
  module load "$gcc_module"
fi

if [[ -n "${SLURM_MODULES:-}" ]]; then
  for module_name in ${SLURM_MODULES}; do
    if [[ "$module_name" == "miniforge3" ]]; then
      continue
    fi
    module load "$module_name"
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: miniforge3 loaded, but conda is still unavailable." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

target_env="${CONDA_ENV_NAME:-opengate}"
python_version="${CONDA_PYTHON_VERSION:-3.12}"

acquire_bootstrap_lock

env_path="$(conda env list | awk -v env="$target_env" '$1==env {print $NF; exit}')"

if [[ -z "$env_path" ]]; then
  echo "Conda environment '$target_env' was not found. Creating it under miniforge3 with Python $python_version..."
  conda create -y -n "$target_env" -c conda-forge "python=${python_version}" pip
  env_path="$(conda env list | awk -v env="$target_env" '$1==env {print $NF; exit}')"
  if [[ -z "$env_path" ]]; then
    echo "Error: failed to locate conda environment '$target_env' after creation." >&2
    exit 1
  fi
fi

export PYTHON_BIN="$env_path/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: expected python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

missing_packages="$(list_missing_packages "$PYTHON_BIN")"
if [[ -n "$missing_packages" ]]; then
  echo "Environment '$target_env' is missing packages: $missing_packages"
  install_python_packages "$PYTHON_BIN"
fi

check_python_imports "$PYTHON_BIN"
conda activate "$target_env"

echo "Using PYTHON_BIN=$PYTHON_BIN"