println("Testing basic Julia functionality...")

# Test basic matrix operations
A = [1.0 2.0; 3.0 4.0]
println("Matrix A:")
println(A)

# Test basic optimization (without external packages)
function simple_opt(A)
    n, m = size(A)
    X = zeros(n, m)
    for i in 1:n
        for j in 1:m
            X[i,j] = A[i,j] * 0.5  # Simple scaling
        end
    end
    return X
end

result = simple_opt(A)
println("Simple optimization result:")
println(result)

println("Basic Julia test completed successfully!")
