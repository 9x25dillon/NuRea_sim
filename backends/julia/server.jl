#!/usr/bin/env julia
"""
NuRea Julia Backend - Matrix Optimization Server
Provides high-performance numerical optimization for the NuRea orchestrator
"""

using HTTP
using JSON3
using Statistics
using LinearAlgebra

# Simple demo transform: scale and prune small values
function optimize_matrix(M::Vector{Vector{Float64}}; method::String="chebyshev_projection", params::Dict=Dict())
    if method == "chebyshev_projection"
        # Simulate Chebyshev projection with scaling and sparsity
        sparsity = get(params, "sparsity", 0.0)
        
        # Scale matrix (simulate projection)
        M2 = [row .* 0.95 for row in M]
        
        if sparsity > 0
            # Apply sparsity by zeroing small values
            allvals = reduce(vcat, M2)
            n = length(allvals)
            k = max(1, min(n, round(Int, sparsity * n)))
            cutoff = sort!(abs.(copy(allvals)))[k]
            M2 = [map(x -> (abs(x) < cutoff ? 0.0 : x), row) for row in M2]
        end
        
        return M2
        
    elseif method == "sparsity"
        # L1 regularization simulation
        lambda = get(params, "lambda", 1.0)
        M2 = [map(x -> sign(x) * max(0, abs(x) - lambda), row) for row in M]
        return M2
        
    elseif method == "rank"
        # Nuclear norm regularization simulation
        tau = get(params, "tau", 1.0)
        # Simple SVD-based rank reduction simulation
        M2 = copy(M)
        for i in 1:length(M2)
            for j in 1:length(M2[i])
                if abs(M2[i][j]) < tau
                    M2[i][j] = 0.0
                end
            end
        end
        return M2
        
    else
        # Default: just return scaled matrix
        return [row .* 0.9 for row in M]
    end
end

function healthz(_req)
    HTTP.Response(200, "ok")
end

function optimize(req)
    try
        body = String(take!(req.body))
        data = JSON3.read(body)
        
        # Extract matrix and parameters
        M = haskey(data, :matrix) ? [Float64.(r) for r in data[:matrix]] : [randn(10) for _ in 1:10]
        method = get(data, :method, "chebyshev_projection")
        params = get(data, :params, Dict())
        
        # Optimize matrix
        Mopt = optimize_matrix(M; method=method, params=params)
        
        # Prepare response
        resp = Dict(
            "optimized_matrix" => Mopt,
            "rows" => length(Mopt),
            "cols" => length(Mopt[1]),
            "method" => method,
            "params" => params
        )
        
        HTTP.Response(200, JSON3.write(resp), ["Content-Type" => "application/json"])
        
    catch e
        error_msg = "Optimization failed: $(e)"
        @error error_msg exception=(e, catch_backtrace())
        HTTP.Response(500, JSON3.write(Dict("error" => error_msg)), ["Content-Type" => "application/json"])
    end
end

# Set up router
router = HTTP.Router()
HTTP.register!(router, "GET",  "/healthz", healthz)
HTTP.register!(router, "POST", "/optimize", optimize)

# Get configuration from environment
host = get(ENV, "JULIA_HOST", "0.0.0.0")
port = parse(Int, get(ENV, "JULIA_PORT", "9000"))

@info "NuRea Julia backend starting" host port
@info "Available endpoints:" "/healthz" "/optimize"

# Start server
HTTP.serve(router, host, port; verbose=false)
