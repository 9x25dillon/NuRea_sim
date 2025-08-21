FROM julia:1.10-bullseye

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Project.toml and install dependencies
COPY backends/julia/Project.toml /app/Project.toml
RUN julia -e 'import Pkg; Pkg.activate("."); Pkg.instantiate()'

# Copy server code
COPY backends/julia/server.jl /app/server.jl

# Set environment variables
ENV JULIA_NUM_THREADS=auto

# Start the server
CMD ["julia", "--project=.", "/app/server.jl"]
