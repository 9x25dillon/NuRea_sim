#!/usr/bin/env bash
set -euo pipefail

# Usage: ./entangle.sh [http|sql]
# Default: http
# Requires: $DATABASE_URL and (for http) $CHAOS_URL (default http://localhost:8081)

MODE="${1:-http}"

if [[ "$MODE" == "http" ]]; then
  echo "Entangling via HTTP route at ${CHAOS_URL:-http://localhost:8081} ..."
  julia entangle_augmented_http.jl
elif [[ "$MODE" == "sql" ]]; then
  echo "Entangling via direct SQL ..."
  julia entangle_augmented_sql.jl
else
  echo "Unknown mode: $MODE"
  exit 1
fi

echo "Entanglement complete."
