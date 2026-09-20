#!/usr/bin/env bash
# Block until the vLLM server answers /health (default: 30 min).
set -euo pipefail
URL="${1:-http://localhost:8000}"; TIMEOUT="${2:-1800}"
for ((i=0; i<TIMEOUT; i+=5)); do
  if curl -sf "$URL/health" >/dev/null 2>&1; then echo "server ready at $URL"; exit 0; fi
  sleep 5
done
echo "server at $URL not ready after ${TIMEOUT}s" >&2; exit 1
