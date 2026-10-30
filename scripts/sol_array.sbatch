#!/usr/bin/env bash
# SLURM job array for the v2 live matrix (one A100 per (model, KV budget) cell).
#
#   python scripts/make_manifest.py
#   sbatch --array=0-7 scripts/sol_array.sbatch                    # all cells
#   sbatch --array=0 --export=ALL,WHAT=quality scripts/sol_array.sbatch
#   JUDGE_URL=https://openai.rc.asu.edu/v1 JUDGE_MODEL=qwen3-235b-a22b-instruct-2507 \
#     JUDGE_API_KEY=$SOL_API_KEY sbatch --array=0 --export=ALL scripts/sol_array.sbatch
#
# 32B cells need 2 GPUs (or an INT8 checkpoint): submit them with --gres=gpu:a100:2
# and TP=2 (start_vllm.sh passes extra args through VLLM_EXTRA_ARGS).
#
#SBATCH --job-name=aclkv-v2
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/aclkv_v2_%A_%a.out
#SBATCH --error=logs/aclkv_v2_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
module load mamba/latest 2>/dev/null || true
module load cuda-12.1.1-gcc-12.1.0 2>/dev/null || true
# shellcheck disable=SC1091
source "${VENV:-.venv}/bin/activate"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
nvidia-smi
bash scripts/run_cell.sh "${SLURM_ARRAY_TASK_ID:-0}" "${WHAT:-all}"
