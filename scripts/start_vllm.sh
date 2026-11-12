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
GPU_UTIL="${GPU_UTIL:-0.75}"
BYTES=$(python -c "print(int(float('$KV_GIB')*1024**3))")

# /reset_prefix_cache lives behind the dev router
export VLLM_SERVER_DEV_MODE=1
if [ -d "/scratch/${USER:-}" ]; then
  export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
fi

# Ensure modern libstdc++ from mamba is prioritized over older system /lib64
PY_BASE_LIB="$(python -c "import sys; print(sys.base_prefix)" 2>/dev/null || true)/lib"
if [ -d "$PY_BASE_LIB" ]; then
  export LD_LIBRARY_PATH="$PY_BASE_LIB:${LD_LIBRARY_PATH:-}"
fi
if [ -d "/packages/apps/mamba/2.9.0/lib" ]; then
  export LD_LIBRARY_PATH="/packages/apps/mamba/2.9.0/lib:${LD_LIBRARY_PATH:-}"
fi

# Ensure CUDA toolkit (nvcc) and CUDA_HOME are available for JIT compilation (FlashInfer, deep_gemm)
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
  ${VLLM_EXTRA_ARGS:-} \
  "$@"
