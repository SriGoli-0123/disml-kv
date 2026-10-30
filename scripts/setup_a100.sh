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
if [ -d "/scratch/${USER:-}" ]; then
  export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
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

echo "--- preparing data + workloads (HotpotQA ~27MB, MuSiQue ~11MB)"
cd "$REPO_ROOT"
# v1 (synthetic-injection) data, kept so the committed v1 results stay reproducible
python scripts/prepare_data.py --max-questions 3000 --tokenizer Qwen/Qwen2.5-7B-Instruct
python scripts/gen_workloads.py
# v2: full HotpotQA dev + MuSiQue-Ans dev, retrieval-driven multi-tenant workloads
python scripts/prepare_data.py --dataset hotpotqa --max-questions 0 --out-dir data/prepared_hotpotqa_full
python scripts/prepare_data.py --dataset musique --max-questions 0
python scripts/gen_workloads_v2.py
python scripts/make_manifest.py
echo "setup complete. v2 GPU matrix: sbatch --array=0-7 scripts/sol_array.sbatch   (see README section 3)"
