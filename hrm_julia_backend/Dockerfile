# hrm_julia_backend/Dockerfile
FROM julia:1.10-bullseye

# curl for healthcheck/debug
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps globally and PRECOMPILE. Fail the build if it can't resolve.
RUN julia --color=yes -e 'import Pkg; Pkg.add(["HTTP", "JSON3"]); Pkg.precompile()'

# --- Now copy the backend sources (server.jl etc)
COPY . /app

# Healthcheck endpoint
HEALTHCHECK CMD curl -fsS http://localhost:9000/healthz || exit 1

# Run the server directly
CMD ["julia", "/app/server.jl"]
