#!/usr/bin/env bash
# Start vLLM with a fixed KV-cache budget and the aclkv scoped-hash plugin.
#
#   bash scripts/start_vllm.sh 2                      # 2 GiB KV cache
#   bash scripts/start_vllm.sh 1 --no-enable-prefix-caching   # "true B0" server
#   MODEL=Qwen/Qwen3-8B bash scripts/start_vllm.sh 4
#
# Env: MODEL, PORT (8000), MAX_MODEL_LEN (6144), MAX_NUM_SEQS (32), GPU_UTIL (0.90)
set -euo pipefail
KV_GIB="${1:?usage: start_vllm.sh <kv_gib> [extra vllm args]}"; shift || true
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_UTIL="${GPU_UTIL:-0.90}"
BYTES=$(python -c "print(int(float('$KV_GIB')*1024**3))")

# /reset_prefix_cache lives behind the dev router
export VLLM_SERVER_DEV_MODE=1
# make sure our plugin is not filtered out if VLLM_PLUGINS is set elsewhere
if [ -n "${VLLM_PLUGINS:-}" ]; then export VLLM_PLUGINS="${VLLM_PLUGINS},aclkv_scoped_hash"; fi

echo "starting vLLM: model=$MODEL kv=${KV_GIB}GiB (${BYTES} bytes) port=$PORT max_num_seqs=$MAX_NUM_SEQS"
exec vllm serve "$MODEL" \
  --port "$PORT" \
  --kv-cache-memory-bytes "$BYTES" \
  --block-size 16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --dtype bfloat16 \
  "$@"
