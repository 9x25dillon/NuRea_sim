
FROM julia:1.10
WORKDIR /app
COPY . /app
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
EXPOSE 8080
CMD ["julia", "--project=.", "main.jl"]
