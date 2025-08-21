FROM julia:1.10-bullseye

# healthcheck needs curl
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt-lists/*

WORKDIR /app

# If you have a Project.toml, we respect it
COPY Project.toml* Manifest.toml* /app/
RUN julia -e 'import Pkg; Pkg.activate("."); try Pkg.instantiate() catch e; @warn "Pkg.instantiate failed (ok if no Project.toml)"; end'

# Copy server code (or rely on volume for live-editing)
COPY . /app

ENV JULIA_HOST=0.0.0.0
ENV JULIA_PORT=9000
ENV JULIA_NUM_THREADS=auto

# If you already have a server entrypoint, this will just include it.
# Otherwise, see fallback below.
CMD ["julia", "--project=.", "-e", "include(\"server.jl\")"]
