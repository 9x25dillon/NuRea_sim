
#!/usr/bin/env bash
set -euo pipefail

echo "=== limps-suite bootstrap ==="
echo "1) Create GitLab group 'limps-suite' and subgroups: core, services, apps, infra"
echo "2) Import each GitHub repo into the matching GitLab project path"
echo "3) In group CI/CD > Variables, set:"
echo "   - CI_REGISTRY_USER / CI_REGISTRY_PASSWORD (already provided by GitLab if registry enabled)"
echo "   - PYPI_TOKEN (for client publish)"
echo "   - STAGING_URL (for deploy link)"
echo "4) Register at least one Docker runner (and a GPU runner if needed)."
echo "5) In infra project, add the orchestrator pipeline to the default branch."
echo "6) Trigger pipeline in infra to orchestrate builds across projects."
