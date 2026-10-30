#!/usr/bin/env bash
# Run one (model, KV budget) cell of the v2 live matrix.
#
#   bash scripts/run_cell.sh <manifest-line-index>        # reads configs/gpu_manifest.tsv
#   bash scripts/run_cell.sh 0 quality                    # only the answer-quality runs
#
# Per cell: start vLLM (+ aclkv plugin) -> smoke test -> for each workload:
#   closed loop C=16 (all policies), open-loop Poisson rate sweep with TTFT SLO,
#   answer-quality runs (+ LLM judge when JUDGE_URL/JUDGE_MODEL are set),
#   live security probe with the insider attacker -> stop vLLM.
# Every bench call is checkpointed (skips finished policy files), so a
# resubmitted array task resumes where it stopped.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
IDX="${1:?usage: run_cell.sh <manifest index> [all|quality|load]}"
WHAT="${2:-all}"
LINE=$(sed -n "$((IDX + 1))p" configs/gpu_manifest.tsv)
MODEL=$(echo "$LINE" | cut -f1); CHAT=$(echo "$LINE" | cut -f2); KV=$(echo "$LINE" | cut -f3)
[ -n "$MODEL" ] || { echo "no manifest line $IDX"; exit 1; }
cfg() { python -c "import json; c=json.load(open('configs/v2.json')); print($1)"; }
WORKLOADS=$(cfg "' '.join(c['gpu']['workloads'])")
POLICIES=$(cfg "' '.join(c['gpu']['policies'])")
QPOL=$(cfg "' '.join(c['gpu']['quality_policies'])")
RATES=$(cfg "' '.join(map(str, c['gpu']['rates']))")
SLO=$(cfg "c['gpu']['slo_ttft_ms']")
NLIVE=$(cfg "c['gpu']['n_requests_live']")
MTAG=$(basename "$MODEL")
OUT="results/v2_live/${MTAG}_kv${KV}"
PORT="${PORT:-8000}"; BASE="http://127.0.0.1:$PORT"
mkdir -p "$OUT" logs
export MODEL MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"

# smaller live copies of the workloads (deterministic; regenerated if missing)
for wl in $WORKLOADS; do
  [ -f "data/workloads_v2/${wl}_live.json" ] || python scripts/gen_workloads_v2.py "$wl" --n-requests "$NLIVE" --suffix _live
done

bash scripts/start_vllm.sh "$KV" > "logs/vllm_${MTAG}_kv${KV}_$(date +%s).log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT
bash scripts/wait_for_server.sh "$BASE" 2400
python scripts/smoke_test_plugin.py --base-url "$BASE" --tokenizer "$MODEL"

JUDGE_ARGS=()
if [ -n "${JUDGE_URL:-}" ] && [ -n "${JUDGE_MODEL:-}" ]; then JUDGE_ARGS=(--judge-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL"); fi
common=(--tokenizer "$MODEL" --chat-format "$CHAT" --kv-gib "$KV" --base-url "$BASE" --max-tokens 48)

for wl in $WORKLOADS; do
  W="data/workloads_v2/${wl}_live.json"
  case "$wl" in mq_*) CORPUS=data/prepared_musique/corpus.jsonl ;; *) CORPUS=data/prepared_hotpotqa_full/corpus.jsonl ;; esac
  if [ "$WHAT" = "all" ] || [ "$WHAT" = "load" ]; then
    # shellcheck disable=SC2086
    python -m aclkv.bench --workload "$W" --corpus "$CORPUS" --policies $POLICIES --concurrency 16 \
      --out-dir "$OUT/closed" "${common[@]}"
    for r in $RATES; do
      # shellcheck disable=SC2086
      python -m aclkv.bench --workload "$W" --corpus "$CORPUS" --policies $POLICIES --arrival poisson --rate "$r" \
        --slo-ttft-ms "$SLO" --out-dir "$OUT/open" "${common[@]}"
    done
  fi
  if [ "$WHAT" = "all" ] || [ "$WHAT" = "quality" ]; then
    # shellcheck disable=SC2086
    python -m aclkv.bench --workload "$W" --corpus "$CORPUS" --policies $QPOL --concurrency 16 \
      --out-dir "$OUT/quality" "${common[@]}" "${JUDGE_ARGS[@]}"
  fi
done
if [ "$WHAT" = "all" ]; then
  python scripts/security_probe.py --workload "data/workloads_v2/hq_main_s0_live.json" \
    --corpus data/prepared_hotpotqa_full/corpus.jsonl --base-url "$BASE" --tokenizer "$MODEL" --chat-format "$CHAT" \
    --insider --policies Global PerUser ACLScope ACLKV B3 --out "$OUT/security_probe.json" || echo "probe reported a problem"
fi
echo "cell $IDX ($MODEL, ${KV} GiB) done -> $OUT"
