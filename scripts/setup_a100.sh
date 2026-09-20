#!/usr/bin/env bash
# One-time environment setup on the A100 box (or a SOL login node).
#
#   bash scripts/setup_a100.sh            # creates .venv with vllm==0.29.0 + this package
#   VENV=/scratch/$USER/aclkv-venv bash scripts/setup_a100.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel
pip install "vllm==0.29.0"
pip install -e "$REPO_ROOT[dev]"

echo "--- checking the vLLM plugin entry point is registered"
python - <<'PY'
from importlib.metadata import entry_points
eps = [e for e in entry_points(group="vllm.general_plugins") if e.name == "aclkv_scoped_hash"]
assert eps, "aclkv plugin entry point missing (pip install -e . again?)"
print("ok:", eps[0])
PY

echo "--- preparing data + workloads (HotpotQA download ~27MB)"
cd "$REPO_ROOT"
python scripts/prepare_data.py --max-questions 3000 --tokenizer Qwen/Qwen2.5-7B-Instruct
python scripts/gen_workloads.py
echo "setup complete. Next: bash scripts/start_vllm.sh 2   (in one shell)  then  python scripts/smoke_test_plugin.py"
