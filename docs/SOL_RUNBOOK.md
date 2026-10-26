# ASU SOL Supercomputer Interactive Runbook

This guide contains the step-by-step instructions for running the Access-Aware KV-Cache Sharing experiments on an interactive A100 GPU node on ASU SOL.

---

## Prerequisites & Interactive GPU Allocation

When logged into a SOL login node, allocate an interactive A100 GPU session. Do not run heavy vLLM inference or benchmarks on the login nodes.

```bash
# Request an interactive A100 GPU session (4 hours)
interactive -p general -q public -g gpu:a100:1 -c 8 -m 64G -t 4:00:00

# Alternative using srun:
# srun -p general -q public --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=4:00:00 --pty bash
```

Once on the compute node, verify your GPU:
```bash
nvidia-smi
```

---

## Phase 1: Repository & Environment Setup

Always work in `/scratch/$USER` to avoid hitting home directory storage quotas.

```bash
# 1. Navigate to scratch space
cd /scratch/$USER

# 2. Clone repository (if not already cloned) and checkout branch 'ved'
git clone https://github.com/SriGoli-0123/disml-kv.git
cd disml-kv
git checkout ved
git pull origin ved

# 3. Load required modules
module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0

# 4. Set environment variables
export HF_HOME=/scratch/$USER/hf_cache
export VENV=/scratch/$USER/aclkv-venv
mkdir -p "$HF_HOME"

# 5. Run the one-time A100 setup script
# This creates the virtual environment, installs vLLM 0.29.0, registers the plugin,
# downloads HotpotQA, and generates the synthetic matrix workloads.
bash scripts/setup_a100.sh

# 6. Activate virtual environment
source "$VENV/bin/activate"

# 7. Run unit test suite (verifies 42 tests, including Go/No-Go checkpoint)
pytest -v
```

---

## Phase 2: Verify Data Preparation & Workload Matrix

If `scripts/setup_a100.sh` ran successfully, the data and workloads are already generated under `data/prepared/` and `data/workloads/`. You can verify or regenerate them manually at any time:

```bash
# Prepare HotpotQA corpus + BM25 rankings
python scripts/prepare_data.py --max-questions 3000 --tokenizer Qwen/Qwen2.5-7B-Instruct

# Generate the 12 matrix workloads (4 share rates x 3 ACL mixes)
python scripts/gen_workloads.py

# Quick check of generated workloads
ls -la data/workloads/
```

---

## Phase 3: Start vLLM Server & Smoke Test Plugin

Open two tmux panes or terminal shells on your allocated compute node.

### Shell 1: Start vLLM Serving Engine
```bash
cd /scratch/$USER/disml-kv
source /scratch/$USER/aclkv-venv/bin/activate
export HF_HOME=/scratch/$USER/hf_cache

# Start vLLM with 2 GiB KV cache on port 8000
bash scripts/start_vllm.sh 2
```
*Wait until you see:*
`aclkv: scoped prefix-cache hashing installed` and `Application startup complete.`

### Shell 2: Run Plugin Smoke Test
```bash
cd /scratch/$USER/disml-kv
source /scratch/$USER/aclkv-venv/bin/activate

# Smoke test multi-barrier block hashing
python scripts/smoke_test_plugin.py --base-url http://localhost:8000
```
*Expected Output:*
`PLUGIN OK: multi-scope hashing is active and stock salts still work.`

---

## Phase 4: Security & Timing Side-Channel Probe

With the server running in Shell 1, run the timing probe in Shell 2:

```bash
cd /scratch/$USER/disml-kv
source /scratch/$USER/aclkv-venv/bin/activate

# Tests whether an attacker without the server-derived HMAC salt can leak victim cache state
python scripts/security_probe.py \
  --workload data/workloads/w_share0.50_mix-default_u20_s0.json \
  --base-url http://localhost:8000 \
  --n 20 \
  --out results/security_probe_kv2.json
```
*Expected Result:*
- Under `UB` (Insecure Global Sharing): Attacker hits victim prefix blocks and observes warm TTFT.
- Under `B3`, `B4`, `B5`, `B6` (Scoped Hashing): `attacker[no_salt] hits = 0/20`, `attacker[forged] hits = 0/20`, TTFT is cold. Zero leaks.

---

## Phase 5: Run Benchmark Matrix (Performance & QA Quality)

You can run either the automated runner script or targeted individual benchmarks.

### Option A: Fast Run (~25 min)
Covers 2 GiB cache budget, share rates 0.0 and 0.5, concurrency 16, policies B0, B1, B3, B5, B6, UB, plus the security probe:
```bash
QUICK=1 bash scripts/run_bench_matrix.sh
```

### Option B: Full Matrix Run (All budgets: 1, 2, 4 GiB; all share rates: 0, 0.25, 0.5, 0.75; concurrency 4, 16)
```bash
bash scripts/run_bench_matrix.sh
```

### Option C: RQ2 Ordering vs. QA Quality Ablations
Evaluate answer quality (EM, F1) on HotpotQA across orderings, anchor depths, and displacement limits:

```bash
# 1. Compare B3 (retrieval), B4 (access-aware), B5 (reuse-aware), B6 (anchor k=2)
python -m aclkv.bench --workload data/workloads/w_share0.50_mix-default_u20_s0.json \
  --policies B3 B4 B5 B6 --tag ordering --concurrency 16 --kv-gib 2

# 2. Sweep anchor depth k in {0, 1, 2, 3}
for k in 0 1 2 3; do
  python -m aclkv.bench --workload data/workloads/w_share0.50_mix-default_u20_s0.json \
    --policies B5 --anchor-k $k --tag anchor$k --concurrency 16 --kv-gib 2
done

# 3. Sweep bounded displacement max_shift in {0, 2, 4}
for s in 0 2 4; do
  python -m aclkv.bench --workload data/workloads/w_share0.50_mix-default_u20_s0.json \
    --policies B5 B6 --max-shift $s --tag shift$s --concurrency 16 --kv-gib 2
done
```

---

## Phase 6: Summarize Results & Generate Figures

Aggregate benchmark logs into summary tables and generate publication-ready plots:

```bash
# 1. Generate Markdown tables (combines simulator baselines + live GPU runs)
python scripts/summarize.py

# View the generated summary table
cat results/summary.md

# 2. Generate plots
python scripts/plot_results.py --out-dir results/plots

# The resulting figures will be saved in results/plots/
ls -lh results/plots/
```
