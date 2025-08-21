
FROM julia:1.10
WORKDIR /app
COPY . /app
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
EXPOSE 8081
HEALTHCHECK CMD wget -qO- http://localhost:8081/health || exit 1
CMD ["julia", "--project=.", "poly_optimizer_server.jl"]
