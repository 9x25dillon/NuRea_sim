# Julia Backend for Matrix Optimization

This Julia backend provides three optimization methods:
1. **Sparsity** - L1 regularization using OSQP
2. **Rank** - Nuclear norm regularization using Convex.jl + SCS
3. **Structure** - Graph Laplacian regularization using JuMP + OSQP

## Installation

### Prerequisites
- Julia 1.8+ installed and accessible via `julia` command
- Sufficient disk space (~2-3GB for packages)

### Step 1: Install Dependencies
```bash
cd hrm_julia_backend
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

If the above fails, try installing packages globally:
```bash
julia -e 'using Pkg; Pkg.add("HTTP"); Pkg.add("JSON3"); Pkg.add("JuMP"); Pkg.add("OSQP"); Pkg.add("Convex"); Pkg.add("SCS")'
```

### Step 2: Test Installation
```bash
julia --project=. -e 'using HTTP; using JSON3; using JuMP; using OSQP; using Convex; using SCS; println("All packages loaded!")'
```

### Step 3: Start Server
```bash
julia --project=. server.jl
```

## Usage

### Health Check
```bash
curl -X GET http://127.0.0.1:9000/health
```

### Test Rank Solver
```bash
curl -X POST http://127.0.0.1:9000/optimize \
  -H "Content-Type: application/json" \
  -d '{"matrix": [[1,2,3],[4,5,6]], "method": "rank", "params": {"tau": 2.0, "lambda": 1.0}}'
```

### Test Structure Solver
```bash
curl -X POST http://127.0.0.1:9000/optimize \
  -H "Content-Type: application/json" \
  -d '{"matrix": [[1,2,3],[4,5,6]], "method": "structure", "params": {"lambda": 1.0}, "adjacency": {"labels": ["n1","n2"], "beta": 0.8, "adjacency": [[0,1],[1,0]]}}'
```

## Troubleshooting

### Package Installation Issues
- Check disk space: `df -h`
- Clear Julia cache: `rm -rf ~/.julia/registries`
- Try installing packages one by one
- Check Julia version: `julia --version`

### Server Won't Start
- Check if port 9000 is available: `netstat -tlnp | grep 9000`
- Verify all packages load: `julia --project=. -e 'using HTTP; println("OK")'`
- Check server.jl syntax: `julia --project=. -c server.jl`

### Performance Issues
- For large matrices, increase `max_iters` in rank solver
- Use sparse adjacency matrices for structure solver
- Monitor memory usage during optimization

## Integration with Python

Set environment variables in your Python code:
```python
import os
os.environ["MATRIX_BACKEND"] = "julia"
os.environ["MATRIX_JULIA_URL"] = "http://127.0.0.1:9000"
```

## API Reference

### POST /optimize
- **matrix**: Input matrix (2D array)
- **method**: "sparsity", "rank", or "structure"
- **params**: Solver-specific parameters
- **adjacency**: Required for "structure" method

### Response Format
```json
{
  "objective": 123.45,
  "matrix_opt": [[...], [...]],
  "iterations": 42,
  "meta": {"solver": "SCS", "method": "rank"}
}
```
