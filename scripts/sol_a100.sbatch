#!/usr/bin/env bash
# SLURM job for ASU SOL: one A100, runs the live benchmark matrix.
#
#   sbatch scripts/sol_a100.sbatch                 # full matrix (~6-8 h)
#   QUICK=1 sbatch --export=ALL,QUICK=1 -t 1:30:00 scripts/sol_a100.sbatch
#
#SBATCH --job-name=aclkv
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/aclkv_%j.out
#SBATCH --error=logs/aclkv_%j.err
set -euo pipefail
# submit from the repo root:  cd disml-kv && sbatch scripts/sol_a100.sbatch
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
module load mamba/latest 2>/dev/null || true
module load cuda-12.1.1-gcc-12.1.0 2>/dev/null || true
# shellcheck disable=SC1091
source "${VENV:-.venv}/bin/activate"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
nvidia-smi
python scripts/run_sim_matrix.py --tag matrix
bash scripts/run_bench_matrix.sh
python scripts/plot_results.py
