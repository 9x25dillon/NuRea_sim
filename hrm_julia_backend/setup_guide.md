# Julia Backend Setup Guide

## Current Status
The Julia backend code is complete and ready, but there are package installation issues that need to be resolved.

## What We Have
✅ Complete server.jl with all three solvers (sparsity, rank, structure)  
✅ Project.toml with correct dependencies  
✅ Basic Julia functionality working  
✅ Mock implementations for testing  

## Package Installation Issues
The following packages are failing to install:
- HTTP
- JSON3  
- JuMP
- OSQP
- Convex
- SCS

## Troubleshooting Steps

### 1. Check Julia Environment
```bash
julia --version  # Should be 1.8+
which julia      # Should be /usr/bin/julia
```

### 2. Clear Julia Cache (if needed)
```bash
rm -rf ~/.julia/registries
rm -rf ~/.julia/environments
```

### 3. Try Alternative Installation Methods

#### Method A: Global Installation
```bash
julia -e 'using Pkg; Pkg.add("HTTP")'
julia -e 'using Pkg; Pkg.add("JSON3")'
julia -e 'using Pkg; Pkg.add("JuMP")'
julia -e 'using Pkg; Pkg.add("OSQP")'
julia -e 'using Pkg; Pkg.add("Convex")'
julia -e 'using Pkg; Pkg.add("SCS")'
```

#### Method B: Project-based Installation
```bash
cd hrm_julia_backend
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

#### Method C: Manual Package Addition
```bash
julia --project=. -e 'using Pkg; Pkg.add("HTTP"); Pkg.add("JSON3"); Pkg.add("JuMP"); Pkg.add("OSQP"); Pkg.add("Convex"); Pkg.add("SCS")'
```

### 4. Check System Resources
```bash
df -h                    # Check disk space
free -h                  # Check memory
ps aux | grep julia      # Check for hanging Julia processes
```

### 5. Alternative: Use System Package Manager
If available on your system:
```bash
# Arch Linux (Garuda)
sudo pacman -S julia-http julia-json3 julia-jump julia-osqp julia-convex julia-scs

# Ubuntu/Debian
sudo apt install julia-http julia-json3 julia-jump julia-osqp julia-convex julia-scs
```

## Testing Without Full Packages

### Run Basic Tests
```bash
julia test_simple.jl    # Basic Julia functionality
julia test_basic.jl     # Mock optimization functions
```

### Expected Output
```
Testing basic optimization functions...
Input matrix:
[1.0 2.0 3.0; 4.0 5.0 6.0]

=== Testing Sparsity Solver ===
Objective: 9.75
Result matrix:
[0.5 1.5 2.5; 3.5 4.5 5.5]

=== Testing Rank Solver ===
Objective: 21.875
Result matrix:
[0.5 1.0 1.5; 2.0 2.5 3.0]

=== Testing Structure Solver ===
Objective: 6.75
Result matrix:
[1.0 2.0 3.0; 4.0 5.0 6.0]

All basic tests completed successfully!
```

## Once Packages Are Working

### Start the Server
```bash
julia --project=. server.jl
```

### Test the Server
```bash
# Health check
curl -X GET http://127.0.0.1:9000/health

# Test rank solver
curl -X POST http://127.0.0.1:9000/optimize \
  -H "Content-Type: application/json" \
  -d '{"matrix": [[1,2,3],[4,5,6]], "method": "rank", "params": {"tau": 2.0, "lambda": 1.0}}'

# Test structure solver  
curl -X POST http://127.0.0.1:9000/optimize \
  -H "Content-Type: application/json" \
  -d '{"matrix": [[1,2,3],[4,5,6]], "method": "structure", "params": {"lambda": 1.0}, "adjacency": {"labels": ["n1","n2"], "beta": 0.8, "adjacency": [[0,1],[1,0]]}}'
```

## Integration with Python

Set these environment variables in your Python code:
```python
import os
os.environ["MATRIX_BACKEND"] = "julia"
os.environ["MATRIX_JULIA_URL"] = "http://127.0.0.1:9000"
```

## Next Steps
1. Resolve package installation issues using the troubleshooting steps above
2. Test the server with the provided curl commands
3. Integrate with your Python orchestrator
4. Run full optimization tests

## Support
If package installation continues to fail, consider:
- Using a different Julia version
- Installing packages via system package manager
- Using a containerized environment
- Running the mock implementations for development/testing
