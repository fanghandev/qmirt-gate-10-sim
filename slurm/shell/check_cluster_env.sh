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

if [[ $# -gt 0 ]]; then
  check_python_imports "$1"
else
  check_python_imports python
fi
