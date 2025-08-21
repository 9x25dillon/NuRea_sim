# 🚀 NuRea Simulation Setup Guide

**Complete Setup for Fractal Cascade Simulation with Docker & Clean Architecture**

## 🎯 What This Guide Covers

This guide will help you set up the **NuRea Simulation platform** with:
- **Clean Architecture**: Organized into `orchestrator/`, `backends/`, `data/`, `artifacts/`
- **Docker Integration**: One-command startup with `docker-compose`
- **Integration Story**: Python ↔ Julia ↔ ChaosRAGJulia workflow
- **End-to-End Examples**: From matrix optimization to vector analysis

## 🏗️ New Architecture Overview

```
NuRea_sim/
├── docker-compose.yml          # Main orchestration
├── plan.json                   # Single source of truth for pipeline
├── orchestrator/               # Python orchestration engine
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       └── cli.py             # Main CLI entry point
├── backends/                   # Service backends
│   ├── julia/                 # Matrix optimization
│   │   ├── Dockerfile
│   │   ├── Project.toml
│   │   └── server.jl
│   ├── chaos-rag/             # Vector database & RAG
│   └── postgres/              # Database initialization
├── data/                       # Input data (small samples in git)
├── artifacts/                  # Output results
└── README.md                   # Project documentation
```

## 🚀 Quick Start (One-Command Setup)

### **Prerequisites**
- Docker and Docker Compose installed
- Git LFS for large files
- 4GB+ RAM available

### **1. Clone and Setup**
```bash
# Clone the repository
git clone https://github.com/9x25dillon/NuRea_sim.git
cd NuRea_sim

# Install Git LFS
git lfs install

# Pull large files
git lfs pull
```

### **2. Start All Services**
```bash
# One command to start everything
docker compose up --build
```

**What happens:**
1. **PostgreSQL** starts with pgvector extension
2. **Julia Backend** starts on port 9000 (matrix optimization)
3. **ChaosRAGJulia** starts on port 8081 (vector database)
4. **Python Orchestrator** runs the pipeline from `plan.json`

### **3. Verify Everything Works**
```bash
# Check Julia backend
curl http://localhost:9000/healthz

# Check ChaosRAGJulia
curl http://localhost:8081/health

# Check orchestrator logs
docker compose logs orchestrator
```

## 🔬 End-to-End Examples

### **Example 1: Matrix Optimization Pipeline**

**What it does:**
- Loads/generates a 64×64 matrix
- Optimizes via Julia backend (Chebyshev projection)
- Vectorizes via ChaosRAGJulia (L2 normalization)
- Saves results to artifacts/

**Run it:**
```bash
# The pipeline runs automatically when you start services
# Check results:
ls -la artifacts/
head -n 5 artifacts/optimized_matrix.csv
```

**Expected Output:**
```
[orchestrator] Pipeline step 1/4: load_matrix
[orchestrator] Generating 64x64 synthetic matrix with seed 42
[orchestrator] Loaded matrix: 64x64
[orchestrator] Pipeline step 2/4: optimize
[orchestrator] Optimizing matrix via Julia backend: chebyshev_projection
[orchestrator] Julia optimization complete: 64x64
[orchestrator] Pipeline step 3/4: vectorize
[orchestrator] Vectorizing matrix via ChaosRAGJulia: l2_normalize -> 1536D
[orchestrator] Pipeline step 4/4: save
[orchestrator] Saved matrix to artifacts/optimized_matrix.csv
[orchestrator] Pipeline execution complete!
```

### **Example 2: Custom Pipeline**

**Create custom plan.json:**
```json
{
  "version": "1.0",
  "run_id": "custom-optimization",
  "backend": { 
    "julia_url": "http://julia-backend:9000"
  },
  "pipeline": [
    {
      "op": "load_matrix",
      "source": "data/my_matrix.csv",
      "shape": [128, 128],
      "seed": 123
    },
    {
      "op": "optimize",
      "method": "sparsity",
      "backend": "julia",
      "lambda": 0.5
    },
    {
      "op": "save",
      "target": "artifacts/my_optimized_matrix.csv"
    }
  ]
}
```

**Run custom pipeline:**
```bash
# Set custom plan path
export PLAN_PATH=/app/custom_plan.json

# Restart orchestrator with new plan
docker compose restart orchestrator
```

## 🛠️ Development Setup

### **Local Development (Without Docker)**

#### **Julia Backend**
```bash
cd backends/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia server.jl
```

#### **Python Orchestrator**
```bash
cd orchestrator
pip install -r requirements.txt
python -m app.cli --plan ../plan.json
```

#### **ChaosRAGJulia**
```bash
cd chaos_rag_single
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia server_vector.jl
```

### **Testing the Integration**

#### **Test Julia Backend**
```bash
# Test health endpoint
curl http://localhost:9000/healthz

# Test optimization
curl -X POST http://localhost:9000/optimize \
  -H "Content-Type: application/json" \
  -d '{"matrix": [[1,2],[3,4]], "method": "chebyshev_projection", "params": {"sparsity": 0.5}}'
```

#### **Test Python Orchestrator**
```bash
# Test with sample data
python -m app.cli --plan plan.json --verbose

# Test with custom plan
python -m app.cli --plan custom_plan.json
```

## 🔧 Configuration

### **Environment Variables (.env)**
```bash
# Copy and customize
cp .env.example .env

# Key variables:
JULIA_PORT=9000              # Julia backend port
CHAOS_RAG_PORT=8081          # ChaosRAGJulia port
ORCH_LOG_LEVEL=INFO          # Orchestrator logging
DATABASE_URL=postgres://...   # Database connection
```

### **Plan.json Configuration**
The `plan.json` file is your **single source of truth** for:
- **Pipeline Operations**: load_matrix, optimize, vectorize, save
- **Backend Selection**: julia, chaos_rag
- **Parameters**: method, sparsity, dimensions, etc.
- **Data Flow**: source files, target outputs

## 📊 Data Management

### **Data Directory Structure**
```
data/
├── sample_matrix.csv          # Small test data (checked into git)
├── matrix.csv                 # Generated if missing
└── README.md                  # Data provenance documentation
```

### **Large Files (Git LFS)**
- **Raw simulation data**: Stored via Git LFS
- **Generated artifacts**: Stored in `artifacts/` directory
- **Data provenance**: Documented in `data/README.md`

### **Fetching Full Datasets**
```bash
# For large datasets not in git
make fetch-data

# Or manually
./scripts/fetch_simulation_data.sh
```

## 🧪 Testing & Validation

### **Run Tests**
```bash
# Python tests
cd orchestrator
python -m pytest tests/

# Julia tests
cd backends/julia
julia --project=. -e 'using Pkg; Pkg.test()'

# Integration tests
python -m pytest tests/integration/
```

### **Performance Benchmarks**
```bash
# Benchmark matrix optimization
python scripts/benchmark_matrix_ops.py

# Benchmark vector operations
python scripts/benchmark_vector_ops.py
```

## 🚀 Production Deployment

### **Docker Production**
```bash
# Production build
docker compose -f docker-compose.prod.yml up --build

# With custom environment
docker compose -f docker-compose.prod.yml --env-file .env.prod up
```

### **Kubernetes Deployment**
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n nurea-sim
```

## 🔍 Troubleshooting

### **Common Issues**

#### **Julia Backend Won't Start**
```bash
# Check Julia installation
julia --version

# Check dependencies
cd backends/julia
julia --project=. -e 'using Pkg; Pkg.status()'
```

#### **PostgreSQL Connection Issues**
```bash
# Check if PostgreSQL is running
docker compose ps postgres

# Check logs
docker compose logs postgres

# Test connection
docker compose exec postgres psql -U chaos_user -d chaos -c "SELECT 1;"
```

#### **Orchestrator Pipeline Fails**
```bash
# Check plan.json syntax
python -c "import json; json.load(open('plan.json'))"

# Check backend connectivity
curl http://localhost:9000/healthz
curl http://localhost:8081/health

# Run with verbose logging
docker compose run --rm orchestrator python -m app.cli --plan /app/plan.json --verbose
```

### **Debug Mode**
```bash
# Enable debug logging
export ORCH_LOG_LEVEL=DEBUG

# Run with debug output
docker compose run --rm orchestrator python -m app.cli --plan /app/plan.json --verbose
```

## 🌟 Next Steps

### **Immediate Improvements**
1. **Add more optimization methods** to Julia backend
2. **Implement real Chebyshev projection** (replace demo transform)
3. **Add entropy-guided optimization** parameters
4. **Expand vectorization methods** in ChaosRAGJulia

### **Advanced Features**
1. **ARC puzzle integration** for cognitive tasks
2. **LIMPS entropy routing** for adaptive optimization
3. **Real-time reactor telemetry** processing
4. **Multi-physics coupling** algorithms

### **Research Directions**
1. **Classical vs. entropy-guided optimization** comparison
2. **PWR dataset as domain-specific benchmark**
3. **Cascade failure prediction** validation
4. **Real-time safety analysis** performance

---

## 🎉 You're Ready!

Your NuRea Simulation platform is now set up with:
- ✅ **Clean, organized architecture**
- ✅ **Docker-based deployment**
- ✅ **Integration story**: Python ↔ Julia ↔ ChaosRAGJulia
- ✅ **End-to-end examples**
- ✅ **Comprehensive testing**
- ✅ **Production-ready configuration**

**Start exploring the future of nuclear physics simulation!** 🌊⚛️

For questions or contributions, see our [Contributing Guidelines](CONTRIBUTING.md) or open an issue on GitHub.
