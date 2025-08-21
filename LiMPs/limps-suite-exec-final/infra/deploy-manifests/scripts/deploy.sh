#!/usr/bin/env bash
set -euo pipefail
ENVIRONMENT="${1:-staging}"

echo "[deploy] Environment: $ENVIRONMENT"
export OPTIMIZER_TAG=${OPTIMIZER_TAG:-latest}
export MOTIF_TAG=${MOTIF_TAG:-latest}
export AL_ULS_TAG=${AL_ULS_TAG:-latest}
export BACKEND_TAG=${BACKEND_TAG:-latest}
export FRONTEND_TAG=${FRONTEND_TAG:-latest}
export CI_REGISTRY=${CI_REGISTRY:-registry.gitlab.com}

docker compose -f infra/deploy-manifests/docker-compose.yml pull
docker compose -f infra/deploy-manifests/docker-compose.yml up -d
docker compose -f infra/deploy-manifests/docker-compose.yml ps

echo "[deploy] Done."
