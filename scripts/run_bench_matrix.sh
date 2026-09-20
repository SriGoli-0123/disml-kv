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
BASE_URL="http://localhost:$PORT"
OUT_DIR="${OUT_DIR:-results/bench}"
LOG_DIR="${LOG_DIR:-logs}"; mkdir -p "$LOG_DIR" "$OUT_DIR"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

echo "model=$MODEL kv_gibs=[$KV_GIBS] shares=[$SHARES] concs=[$CONCS] policies=[$POLICIES] mix=$MIX"

start_server() {  # $1 = kv gib, $2.. = extra vllm args
  local kv="$1"; shift
  bash scripts/start_vllm.sh "$kv" "$@" > "$LOG_DIR/vllm_kv${kv}_$(date +%s).log" 2>&1 &
  SERVER_PID=$!
  bash scripts/wait_for_server.sh "$BASE_URL" 1800
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

for kv in $KV_GIBS; do
  start_server "$kv"
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then   # SKIP_SMOKE=1 when benchmarking --scope-mode marker on a stock vLLM
    python scripts/smoke_test_plugin.py --base-url "$BASE_URL" --tokenizer "$TOKENIZER"
  fi
  for share in $SHARES; do
    wl=$(printf "data/workloads/w_share%.2f_mix-%s_u%s_s%s.json" "$share" "$MIX" "$NUSERS" "$SEED")
    [ -f "$wl" ] || { echo "missing workload $wl (run scripts/gen_workloads.py)"; exit 1; }
    for c in $CONCS; do
      echo "=== kv=${kv}GiB share=$share concurrency=$c"
      # shellcheck disable=SC2086
      python -m aclkv.bench --workload "$wl" --policies $POLICIES --base-url "$BASE_URL" \
        --tokenizer "$TOKENIZER" --chat-format "$CHAT_FORMAT" --concurrency "$c" --max-tokens "$MAX_TOKENS" \
        --kv-gib "$kv" --out-dir "$OUT_DIR" $EXTRA_BENCH_ARGS
    done
  done
  if [ "${SECURITY_PROBE:-1}" = "1" ]; then
    wl=$(printf "data/workloads/w_share0.50_mix-%s_u%s_s%s.json" "$MIX" "$NUSERS" "$SEED")
    python scripts/security_probe.py --workload "$wl" --base-url "$BASE_URL" --tokenizer "$TOKENIZER" \
      --chat-format "$CHAT_FORMAT" --out "results/security_probe_kv${kv}.json" || echo "security probe reported a problem"
  fi
  stop_server

  if [ "${TRUE_B0:-0}" = "1" ]; then
    start_server "$kv" --no-enable-prefix-caching
    for share in $SHARES; do
      wl=$(printf "data/workloads/w_share%.2f_mix-%s_u%s_s%s.json" "$share" "$MIX" "$NUSERS" "$SEED")
      for c in $CONCS; do
        python -m aclkv.bench --workload "$wl" --policies B0 --tag trueB0 --base-url "$BASE_URL" \
          --tokenizer "$TOKENIZER" --chat-format "$CHAT_FORMAT" --concurrency "$c" --max-tokens "$MAX_TOKENS" \
          --kv-gib "$kv" --out-dir "$OUT_DIR" --no-reset $EXTRA_BENCH_ARGS
      done
    done
    stop_server
  fi
done
python scripts/summarize.py --bench-dir "$OUT_DIR" --sim-csv results/sim/matrix/summary.csv || true
echo "done. results in $OUT_DIR; tables in results/summary.md"
