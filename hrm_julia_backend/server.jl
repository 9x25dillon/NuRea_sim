# NuRea Julia Backend Server
# Packages are pre-installed in Docker build

using HTTP, JSON3, Statistics, LinearAlgebra

# Simple matrix optimization functions
function optimize_matrix(M::Vector{Vector{Float64}}; sparsity::Float64=0.0)
    M2 = [row .* 0.95 for row in M] # demo projection
    if sparsity <= 0
        return M2
    end
    
    allvals = reduce(vcat, M2)
    n = length(allvals)
    k = max(1, min(n, round(Int, sparsity * n)))
    cutoff = sort!(abs.(copy(allvals)))[k]
    
    return [map(x -> (abs(x) < cutoff ? 0.0 : x), row) for row in M2]
end

# Health check handler
function healthz_handler(req)
    return HTTP.Response(200, "ok")
end

# Health check handler (detailed)
function health_handler(req)
    return HTTP.Response(200, JSON3.write(Dict("status" => "healthy", "backend" => "julia")))
end

# Optimization handler
function optimize_handler(req)
    body = JSON3.read(String(req.body))
    M = haskey(body, "matrix") ? [Float64.(r) for r in body["matrix"]] : [randn(10) for _ in 1:10]
    s = get(body, "sparsity", 0.0)
    
    Mopt = optimize_matrix(M; sparsity=Float64(s))
    
    resp = Dict(
        "optimized_matrix" => Mopt,
        "rows" => length(Mopt),
        "cols" => length(Mopt[1]),
        "method" => "simple_projection",
        "sparsity" => s
    )
    
    return HTTP.Response(200, JSON3.write(resp), ["Content-Type" => "application/json"])
end

# Router setup
router = HTTP.Router()
HTTP.register!(router, "GET", "/healthz", healthz_handler)
HTTP.register!(router, "GET", "/health", health_handler)
HTTP.register!(router, "POST", "/optimize", optimize_handler)

# Main server
function main()
    host = get(ENV, "JULIA_HOST", "0.0.0.0")
    port = parse(Int, get(ENV, "JULIA_PORT", "9000"))
    
    println("NuRea Julia backend listening on $host:$port")
    println("Available endpoints:")
    println("  GET  /healthz  - Docker health check")
    println("  GET  /health   - Detailed health status")
    println("  POST /optimize - Matrix optimization")
    
    HTTP.serve(router, host, port; verbose=false)
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
