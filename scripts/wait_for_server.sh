#!/usr/bin/env bash
# Block until the vLLM server answers /health (default: 30 min).
set -euo pipefail
URL="${1:-http://localhost:8000}"; TIMEOUT="${2:-1800}"
echo "Waiting for vLLM server at $URL to become ready..."
for ((i=0; i<TIMEOUT; i+=5)); do
  if curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "vLLM server is healthy and ready at $URL (took ${i}s)."
    exit 0
  fi
  if (( i > 0 && i % 30 == 0 )); then
    echo "  still initializing vLLM engine at $URL (${i}s elapsed, compiling & warming up)..."
  fi
  sleep 5
done
echo "server at $URL not ready after ${TIMEOUT}s" >&2; exit 1
