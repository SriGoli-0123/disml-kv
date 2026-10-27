#!/usr/bin/env bash
# Full live benchmark matrix on one GPU: for each KV budget start a server,
# run every (workload x concurrency x policy) with aclkv.bench, stop the server.
#
#   bash scripts/run_bench_matrix.sh                 # full matrix from configs/matrix.json (hours)
#   QUICK=1 bash scripts/run_bench_matrix.sh         # 2 GiB, share {0,0.5}, C=16, ~25 min
#   KV_GIBS="1 4" SHARES="0.5" CONCS="16" POLICIES="B1 B3 B5" bash scripts/run_bench_matrix.sh
#   MIX=restrictive bash scripts/run_bench_matrix.sh
#   TRUE_B0=1 ...                                    # also run B0 on a server with prefix caching disabled
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CFG=configs/matrix.json
# Ensure modern libstdc++ from mamba is prioritized over older system /lib64
PY_BASE_LIB="$(python -c "import sys; print(sys.base_prefix)" 2>/dev/null || true)/lib"
if [ -d "$PY_BASE_LIB" ]; then
  export LD_LIBRARY_PATH="$PY_BASE_LIB:${LD_LIBRARY_PATH:-}"
fi
if [ -d "/packages/apps/mamba/2.9.0/lib" ]; then
  export LD_LIBRARY_PATH="/packages/apps/mamba/2.9.0/lib:${LD_LIBRARY_PATH:-}"
fi

# Ensure CUDA toolkit (nvcc) and CUDA_HOME are available
if ! command -v nvcc >/dev/null 2>&1; then
  for mod_init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
    if [ -f "$mod_init" ]; then
      source "$mod_init" 2>/dev/null || true
      break
    fi
  done
  if command -v module >/dev/null 2>&1; then
    module load cuda-12.1.1-gcc-12.1.0 2>/dev/null || module load cuda-12.1 2>/dev/null || module load cuda 2>/dev/null || true
  fi
fi

if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  else
    for p in /packages/apps/cuda/12.1.1 /packages/apps/cuda/12.4 /packages/apps/cuda/12.8 /packages/apps/cuda/*; do
      if [ -x "$p/bin/nvcc" ]; then
        export CUDA_HOME="$p"
        export PATH="$p/bin:$PATH"
        export LD_LIBRARY_PATH="$p/lib64:${LD_LIBRARY_PATH:-}"
        break
      fi
    done
  fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# Disable FlashInfer JIT sampler by default to prevent nvcc compilation errors (e.g. unknown option --compress-mode on CUDA <=12.3)
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

jq_get() { python -c "import json,sys; c=json.load(open('$CFG')); print($1)"; }

MODEL="${MODEL:-$(jq_get "c['model']")}"
export MODEL
TOKENIZER="${TOKENIZER:-$(jq_get "c['tokenizer']")}"
CHAT_FORMAT="${CHAT_FORMAT:-$(jq_get "c['chat_format']")}"
MAX_TOKENS="${MAX_TOKENS:-$(jq_get "c['max_tokens']")}"
MIX="${MIX:-default}"
SEED="${SEED:-0}"
NUSERS="${NUSERS:-$(jq_get "c['workload']['n_users']")}"
if [ "${QUICK:-0}" = "1" ]; then
  KV_GIBS="${KV_GIBS:-$(jq_get "' '.join(map(str,c['quick']['kv_gib']))")}"
  SHARES="${SHARES:-$(jq_get "' '.join(map(str,c['quick']['share_rates']))")}"
  CONCS="${CONCS:-$(jq_get "' '.join(map(str,c['quick']['concurrency']))")}"
  POLICIES="${POLICIES:-$(jq_get "' '.join(c['quick']['policies'])")}"
else
  KV_GIBS="${KV_GIBS:-$(jq_get "' '.join(map(str,c['kv_gib']))")}"
  SHARES="${SHARES:-$(jq_get "' '.join(map(str,c['share_rates']))")}"
  CONCS="${CONCS:-$(jq_get "' '.join(map(str,c['concurrency']))")}"
  POLICIES="${POLICIES:-$(jq_get "' '.join(c['policies'])")}"
fi
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:$PORT"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
OUT_DIR="${OUT_DIR:-results/bench}"
LOG_DIR="${LOG_DIR:-logs}"; mkdir -p "$LOG_DIR" "$OUT_DIR"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

echo "model=$MODEL kv_gibs=[$KV_GIBS] shares=[$SHARES] concs=[$CONCS] policies=[$POLICIES] mix=$MIX"

start_server() {  # $1 = kv gib, $2.. = extra vllm args
  local kv="$1"; shift
  local log_file="$LOG_DIR/vllm_kv${kv}_$(date +%s).log"
  echo "--- Starting vLLM server in background: KV=${kv}GiB (log: $log_file)"
  bash scripts/start_vllm.sh "$kv" "$@" > "$log_file" 2>&1 &
  SERVER_PID=$!
  bash scripts/wait_for_server.sh "$BASE_URL" 1800 "$SERVER_PID" "$log_file"
}
stop_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 5
  fi
}
trap stop_server EXIT

FORCE="${FORCE:-0}"
if [ "$FORCE" = "1" ]; then
  EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS} --force"
fi

bench_file_done() {
  local tag="$1" wl="$2" c="$3" kv="$4" pol="$5"
  local wl_stem; wl_stem="$(basename "$wl" .json)"
  local prefix=""
  [ -n "$tag" ] && prefix="${tag}_"
  local target="$OUT_DIR/${prefix}${wl_stem}_c${c}_kv${kv}_${pol}.json"
  if [ "$FORCE" != "1" ] && [ -s "$target" ]; then
    return 0
  else
    return 1
  fi
}

workload_concurrency_done() {
  local wl="$1" c="$2" kv="$3"
  for pol in $POLICIES; do
    if ! bench_file_done "" "$wl" "$c" "$kv" "$pol"; then
      return 1
    fi
  done
  return 0
}

kv_budget_done() {
  local kv="$1"
  for share in $SHARES; do
    local wl; wl=$(printf "data/workloads/w_share%.2f_mix-%s_u%s_s%s.json" "$share" "$MIX" "$NUSERS" "$SEED")
    for c in $CONCS; do
      if ! workload_concurrency_done "$wl" "$c" "$kv"; then
        return 1
      fi
      if [ "${TRUE_B0:-0}" = "1" ]; then
        if ! bench_file_done "trueB0" "$wl" "$c" "$kv" "B0"; then
          return 1
        fi
      fi
    done
  done
  if [ "${SECURITY_PROBE:-1}" = "1" ]; then
    local probe_out="results/security_probe_kv${kv}.json"
    if [ "$FORCE" != "1" ] && [ ! -s "$probe_out" ]; then
      return 1
    fi
  fi
  return 0
}

for kv in $KV_GIBS; do
  if [ "$FORCE" != "1" ] && kv_budget_done "$kv"; then
    echo "=== [checkpoint] All benchmarks and probes for kv=${kv}GiB are already completed. Skipping server start."
    continue
  fi

  start_server "$kv"
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then   # SKIP_SMOKE=1 when benchmarking --scope-mode marker on a stock vLLM
    python scripts/smoke_test_plugin.py --base-url "$BASE_URL" --tokenizer "$TOKENIZER"
  fi
  for share in $SHARES; do
    wl=$(printf "data/workloads/w_share%.2f_mix-%s_u%s_s%s.json" "$share" "$MIX" "$NUSERS" "$SEED")
    [ -f "$wl" ] || { echo "missing workload $wl (run scripts/gen_workloads.py)"; exit 1; }
    for c in $CONCS; do
      if [ "$FORCE" != "1" ] && workload_concurrency_done "$wl" "$c" "$kv"; then
        echo "=== [checkpoint] kv=${kv}GiB share=$share concurrency=$c all policies completed. Skipping."
        continue
      fi
      echo "=== kv=${kv}GiB share=$share concurrency=$c"
      # shellcheck disable=SC2086
      python -m aclkv.bench --workload "$wl" --policies $POLICIES --base-url "$BASE_URL" \
        --tokenizer "$TOKENIZER" --chat-format "$CHAT_FORMAT" --concurrency "$c" --max-tokens "$MAX_TOKENS" \
        --kv-gib "$kv" --out-dir "$OUT_DIR" $EXTRA_BENCH_ARGS
    done
  done
  if [ "${SECURITY_PROBE:-1}" = "1" ]; then
    probe_out="results/security_probe_kv${kv}.json"
    if [ "$FORCE" != "1" ] && [ -s "$probe_out" ]; then
      echo "=== [checkpoint] security probe for kv=${kv}GiB already exists ($probe_out). Skipping."
    else
      wl=$(printf "data/workloads/w_share0.50_mix-%s_u%s_s%s.json" "$MIX" "$NUSERS" "$SEED")
      python scripts/security_probe.py --workload "$wl" --base-url "$BASE_URL" --tokenizer "$TOKENIZER" \
        --chat-format "$CHAT_FORMAT" --out "$probe_out" || echo "security probe reported a problem"
    fi
  fi
  stop_server

  if [ "${TRUE_B0:-0}" = "1" ]; then
    true_b0_needed=0
    for share in $SHARES; do
      wl=$(printf "data/workloads/w_share%.2f_mix-%s_u%s_s%s.json" "$share" "$MIX" "$NUSERS" "$SEED")
      for c in $CONCS; do
        if ! bench_file_done "trueB0" "$wl" "$c" "$kv" "B0"; then
          true_b0_needed=1
          break 2
        fi
      done
    done

    if [ "$true_b0_needed" = "1" ] || [ "$FORCE" = "1" ]; then
      start_server "$kv" --no-enable-prefix-caching
      for share in $SHARES; do
        wl=$(printf "data/workloads/w_share%.2f_mix-%s_u%s_s%s.json" "$share" "$MIX" "$NUSERS" "$SEED")
        for c in $CONCS; do
          if [ "$FORCE" != "1" ] && bench_file_done "trueB0" "$wl" "$c" "$kv" "B0"; then
            echo "=== [checkpoint] trueB0 kv=${kv}GiB share=$share c=$c already completed. Skipping."
            continue
          fi
          python -m aclkv.bench --workload "$wl" --policies B0 --tag trueB0 --base-url "$BASE_URL" \
            --tokenizer "$TOKENIZER" --chat-format "$CHAT_FORMAT" --concurrency "$c" --max-tokens "$MAX_TOKENS" \
            --kv-gib "$kv" --out-dir "$OUT_DIR" --no-reset $EXTRA_BENCH_ARGS
        done
      done
      stop_server
    else
      echo "=== [checkpoint] All trueB0 benchmarks for kv=${kv}GiB already completed. Skipping server start."
    fi
  fi
done
python scripts/summarize.py --bench-dir "$OUT_DIR" --sim-csv results/sim/matrix/summary.csv || true
echo "done. results in $OUT_DIR; tables in results/summary.md"
