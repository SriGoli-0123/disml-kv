#!/usr/bin/env bash
#SBATCH --job-name=aclkv-probe
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
module load mamba/latest 2>/dev/null || true
module load cuda-12.1.1-gcc-12.1.0 2>/dev/null || true
source "${VENV:-/scratch/$USER/aclkv-venv}/bin/activate"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
WL=data/workloads/w_share0.50_mix-default_u20_s0.json
for kv in 1 2 4; do
  pkill -f vllm 2>/dev/null || true; sleep 3
  log=logs/vllm_probe_kv${kv}_$(date +%s).log
  bash scripts/start_vllm.sh "$kv" > "$log" 2>&1 &
  PID=$!
  bash scripts/wait_for_server.sh http://127.0.0.1:8000 1800 "$PID" "$log"
  python scripts/security_probe.py --workload "$WL" --base-url http://127.0.0.1:8000 \
    --policies B1 B2 B6 --n 20 --out results/security_probe_B1B2B6_kv${kv}.json \
    || echo "probe flagged a problem at kv=${kv}"
  kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true
  pkill -f vllm 2>/dev/null || true; sleep 3
done
python scripts/summarize.py