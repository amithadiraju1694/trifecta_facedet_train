#!/usr/bin/env bash
set -euo pipefail

port="${1:-7860}"

if ! [[ "$port" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [port]" >&2
  exit 2
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "Error: lsof not found on PATH" >&2
  exit 127
fi

pids="$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"

if [[ -z "${pids}" ]]; then
  echo "No listening processes found on port ${port}"
  exit 0
fi

echo "Killing PIDs on port ${port}: ${pids//$'\n'/ }"
while IFS= read -r pid; do
  [[ -z "$pid" ]] && continue
  kill -9 "$pid" 2>/dev/null || true
done <<< "$pids"

echo "Done."
