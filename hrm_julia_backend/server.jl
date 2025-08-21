using HTTP
using JSON3
using JuMP
using OSQP
using Convex
using SCS
using LinearAlgebra
using SparseArrays

# Existing sparsity solver
function opt_sparsity(A::Matrix{Float64}; λ::Float64=1.0)
    n, m = size(A)
    model = Model(optimizer_with_attributes(OSQP.Optimizer, "verbose" => false))
    
    @variable(model, X[1:n, 1:m])
    @objective(model, Min, (λ/2) * sum((X[i,j] - A[i,j])^2 for i=1:n, j=1:m) + sum(abs(X[i,j]) for i=1:n, j=1:m))
    
    optimize!(model)
    X̂ = value.(X)
    obj = objective_value(model)
    iters = MOI.get(model, MOI.IterationCount())
    return (objective = obj, matrix_opt = X̂, iterations = iters)
end

# Nuclear-norm "rank" solver (Convex.jl + SCS)
function opt_rank_nuclear(A::Matrix{Float64}; τ::Float64=1.0, λ::Float64=1.0, maxiters::Int=10_000)
    n, m = size(A)
    X = Convex.Variable(n, m)
    problem = minimize( τ * nuclearnorm(X) + (λ/2) * sumsquares(X - A) )
    Convex.solve!(problem, SCS.Optimizer; silent_solver = true, max_iters = maxiters)
    X̂ = evaluate(X)
    obj = problem.optval
    iters = problem.solver_result.info[:iter]  # SCS iteration count
    return (objective = obj, matrix_opt = X̂, iterations = iters)
end

# Build symmetric graph Laplacian from (possibly directed) adjacency
function laplacian_from_adj(A::AbstractMatrix{<:Real})
    A = Array{Float64}(A)
    A .= max.(A, 0.0)               # no negative edges
    @inbounds for i in 1:size(A,1)  # zero self-loops
        A[i,i] = 0.0
    end
    A = (A .+ A') ./ 2              # symmetrize
    d = sum(A, dims=2)
    L = Diagonal(vec(d)) - A        # L = D - A
    return sparse(L)
end

# Laplacian "structure" solver (JuMP + OSQP)
"""
min_X   (λ/2)||X - A||_F^2 + (β/2) * tr(X' * L * X)
This stays a convex QP if L is PSD (graph Laplacian is PSD).
"""
function opt_structured(A::Matrix{Float64}, L::SparseMatrixCSC{Float64,Int};
                        λ::Float64=1.0, β::Float64=1.0)
    n, m = size(A)
    model = Model(optimizer_with_attributes(OSQP.Optimizer, "verbose" => false))

    @variable(model, X[1:n, 1:m])

    # Quadratic objective components:
    # (λ/2)*||X-A||^2_F = (λ/2) * sum (X - A).^2
    # (β/2)*tr(X' L X)  = (β/2) * sum_k X[:,k]' * L * X[:,k]
    @expression(model, recon, (λ/2) * sum((X[i,j] - A[i,j])^2 for i=1:n, j=1:m))
    @expression(model, smooth, (β/2) * sum( sum( X[p,k]*L[p,q]*X[q,k] for p=1:n, q=1:n ) for k=1:m ))
    @objective(model, Min, recon + smooth)

    optimize!(model)
    X̂ = value.(X)
    obj = objective_value(model)
    iters = MOI.get(model, MOI.IterationCount())
    return (objective = obj, matrix_opt = X̂, iterations = iters)
end

# HTTP handler for optimization requests
function optimize_handler(req)
    body = JSON3.read(String(req.body))
    A = Matrix{Float64}(body["matrix"])
    method = String(get(body, "method", "sparsity"))
    params = haskey(body, "params") ? body["params"] : JSON3.Object()

    if method == "sparsity"
        λ = haskey(params, "lambda") ? Float64(params["lambda"]) : 1.0
        res = opt_sparsity(A; λ=λ)
        return HTTP.Response(200, JSON3.write(Dict(
            "objective"=>res.objective,
            "matrix_opt"=>res.matrix_opt,
            "iterations"=>res.iterations,
            "meta"=>Dict("solver"=>"OSQP","method"=>"sparsity")
        )))

    elseif method == "rank"
        τ = haskey(params, "tau") ? Float64(params["tau"]) : 1.0
        λ = haskey(params, "lambda") ? Float64(params["lambda"]) : 1.0
        maxiters = haskey(params, "max_iters") ? Int(params["max_iters"]) : 10_000
        res = opt_rank_nuclear(A; τ=τ, λ=λ, maxiters=maxiters)
        return HTTP.Response(200, JSON3.write(Dict(
            "objective"=>res.objective,
            "matrix_opt"=>res.matrix_opt,
            "iterations"=>res.iterations,
            "meta"=>Dict("solver"=>"SCS","method"=>"rank","tau"=>τ,"lambda"=>λ)
        )))

    elseif method == "structure"
        if !haskey(body, "adjacency")
            return HTTP.Response(400, "{\"error\":\"structure requires adjacency\"}")
        end
        adj = body["adjacency"]
        β = haskey(adj, "beta") ? Float64(adj["beta"]) : 1.0
        adjmat = Matrix{Float64}(adj["adjacency"])
        L = laplacian_from_adj(adjmat)

        λ = haskey(params, "lambda") ? Float64(params["lambda"]) : 1.0
        res = opt_structured(A, L; λ=λ, β=β)
        return HTTP.Response(200, JSON3.write(Dict(
            "objective"=>res.objective,
            "matrix_opt"=>res.matrix_opt,
            "iterations"=>res.iterations,
            "meta"=>Dict("solver"=>"OSQP","method"=>"structure","beta"=>β,"lambda"=>λ)
        )))

    else
        return HTTP.Response(400, "{\"error\":\"method not implemented\"}")
    end
end

# Health check endpoint
function health_handler(req)
    return HTTP.Response(200, "{\"status\":\"healthy\",\"backend\":\"julia\"}")
end

# Router setup
function router(req)
    if req.method == "POST" && req.target == "/optimize"
        return optimize_handler(req)
    elseif req.method == "GET" && req.target == "/health"
        return health_handler(req)
    else
        return HTTP.Response(404, "{\"error\":\"not found\"}")
    end
end

# Main server
function main()
    println("Starting Julia optimization server on port 9000...")
    println("Available methods: sparsity, rank, structure")
    println("Health check: GET /health")
    println("Optimize: POST /optimize")
    
    HTTP.serve(router, "127.0.0.1", 9000)
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
