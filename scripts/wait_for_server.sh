#!/usr/bin/env bash
# Block until the vLLM server answers /health (default: 30 min).
set -euo pipefail
URL="${1:-http://localhost:8000}"
TIMEOUT="${2:-1800}"
PID="${3:-}"
LOG_FILE="${4:-}"

check_health() {
  # 1. curl IPv4 with proxy bypassed
  curl -4 -sf --noproxy "*" "$URL/health" >/dev/null 2>&1 && return 0
  curl -4 -sf --noproxy "*" "${URL/localhost/127.0.0.1}/health" >/dev/null 2>&1 && return 0
  # 2. Python urllib fallback
  python3 -c "import urllib.request; urllib.request.urlopen('$URL/health', timeout=1)" >/dev/null 2>&1 && return 0
  python3 -c "import urllib.request; urllib.request.urlopen('${URL/localhost/127.0.0.1}/health', timeout=1)" >/dev/null 2>&1 && return 0
  # 3. Server log readiness detection
  if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

echo "Waiting for vLLM server at $URL to become ready..."
for ((i=0; i<TIMEOUT; i+=5)); do
  if check_health; then
    echo "vLLM server is healthy and ready at $URL (took ${i}s)."
    exit 0
  fi
  if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: vLLM server process (PID $PID) died unexpectedly!" >&2
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
      echo "--- Last 30 lines of $LOG_FILE ---" >&2
      tail -n 30 "$LOG_FILE" >&2
      echo "-----------------------------------" >&2
    fi
    exit 1
  fi
  if (( i > 0 && i % 20 == 0 )); then
    echo "  ... still initializing vLLM engine at $URL (${i}s elapsed, weights & torch.compile) ..."
  fi
  sleep 5
done
echo "server at $URL not ready after ${TIMEOUT}s" >&2
if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
  tail -n 30 "$LOG_FILE" >&2
fi
exit 1
