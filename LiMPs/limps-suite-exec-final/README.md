
# limps-suite (execution-ready CI/CD scaffolding)

This package contains **ready-to-run GitLab CI/CD** configs, Dockerfiles, and a simple Compose-based deployment.

## Projects (expected GitLab paths)
- limps-suite/core/limps-matrix-optimizer
- limps-suite/core/symbolic-polynomial-svc
- limps-suite/core/entropy-engine
- limps-suite/services/motif-detection
- limps-suite/services/poly-optimizer-client
- limps-suite/services/al-uls-orchestrator
- limps-suite/apps/choppy-backend
- limps-suite/apps/choppy-frontend
- limps-suite/infra (contains orchestrator + deploy manifests)

## How to use
1. Create the GitLab group/subgroups and import your repos to the above paths.
2. Copy the corresponding `.gitlab-ci.yml` and `Dockerfile` from each folder into each project.
3. Put everything inside `infra` into your `limps-suite/infra` project.
4. Set CI/CD variables at the **group** level:
   - `PYPI_TOKEN` (for publishing the Python client on tags)
   - `STAGING_URL` (e.g., https://staging.example.com)
5. Ensure GitLab Container Registry is enabled for the group.
6. Register a Docker runner (and optional GPU runner).
7. Push to any project → the pipeline builds/tests and publishes an image.
8. Trigger the **orchestrator** (in infra) to build in the correct dependency order and deploy to Compose.

## Deployment
- Uses `infra/deploy-manifests/docker-compose.yml` with image coordinates anchored to your GitLab registry.
- `infra/deploy-manifests/scripts/deploy.sh staging` will pull and run containers on the host with Docker.
