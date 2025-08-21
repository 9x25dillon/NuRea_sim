#!/usr/bin/env bash
set -euo pipefail

# Usage: ./load_augmented.sh
# Requires: Julia, PostgreSQL reachable via $DATABASE_URL, ChaosRAGJulia schema initialized.

echo "Loading augmented CSVs into hd_nodes..."
julia ingest_augmented.jl
echo "All set."
