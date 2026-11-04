#!/usr/bin/env bash
# Extra seeds and larger user counts, simulator only (CPU, no GPU / no vLLM).
#
# Prerequisite: data/prepared exists, i.e. you already ran
#   python scripts/prepare_data.py --max-questions 3000 --tokenizer Qwen/Qwen2.5-7B-Instruct
#
# Usage:   bash scripts/run_extra_sim.sh
# Tunable: SEEDS="1 2 3 4" USERS="50 100" MIX=default bash scripts/run_extra_sim.sh
#
# Outputs go to results/sim/extra/ and results/sim_seed_consistency.{csv,md};
# nothing in results/sim/matrix or results/bench is touched.
set -euo pipefail

SEEDS="${SEEDS:-1 2 3 4}"
USERS="${USERS:-50 100}"
MIX="${MIX:-default}"

# 20 users, seed 0 plus the extra seeds (seed 0 regenerates the committed workloads)
python scripts/gen_workloads.py --acl-mix "$MIX" --seeds 0 $SEEDS

# larger user populations, seed 0
for u in $USERS; do
  python scripts/gen_workloads.py --acl-mix "$MIX" --n-users "$u" --seeds 0
done

python scripts/run_sim_matrix.py --workloads data/workloads/w_share*_mix-"${MIX}"_u*_s*.json --tag extra
python scripts/aggregate_seeds.py --sim-csv results/sim/extra/summary.csv
