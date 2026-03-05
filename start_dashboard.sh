#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT_CANDIDATES=(8501 8502 8503 8601)
SELECTED_PORT=""

for p in "${PORT_CANDIDATES[@]}"; do
  if ! lsof -i ":$p" >/dev/null 2>&1; then
    SELECTED_PORT="$p"
    break
  fi
done

if [[ -z "$SELECTED_PORT" ]]; then
  echo "No free port found in: ${PORT_CANDIDATES[*]}"
  exit 1
fi

echo "Starting dashboard on port $SELECTED_PORT"
echo "If VS Code says 'Error forwarding port', forward this exact port manually."

auto_cmd=(streamlit run dashboard_app.py --server.address 0.0.0.0 --server.port "$SELECTED_PORT" --server.headless true)
"${auto_cmd[@]}"
