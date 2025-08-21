# 🌊 Fractal Cascade Simulation (FCS) with ChaosRAGJulia

**Advanced Nuclear Physics Simulation & AI-Powered Analysis Platform**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Julia](https://img.shields.io/badge/Julia-1.8+-purple.svg)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-green.svg)](https://postgresql.org/)

## 🎯 What This Project Is For

**Fractal Cascade Simulation (FCS)** is a cutting-edge platform that combines **nuclear physics simulation** with **AI-powered analysis** to solve complex problems in:

- **Nuclear Reactor Design & Safety**
- **Lattice Physics Calculations** (PWR fuel assembly neutronics)
- **Temporal Cascade Analysis** of reactor states
- **Multi-Physics Optimization** using advanced mathematical techniques
- **Real-time Reactor Monitoring** and predictive analytics

## 🚀 Core Capabilities

### 🔬 **Nuclear Physics Simulation**
- **Lattice Physics Calculations**: Advanced neutron transport simulations for PWR fuel assemblies
- **Vector-Augmented Physics**: Combines traditional numerical data with AI-ready vector representations
- **Temporal Causality Modeling**: Creates explicit time-based relationships between simulation states
- **Cascade Failure Prediction**: Analyzes how reactor states evolve and cascade over time

### 🤖 **AI-Powered Analysis (ChaosRAGJulia)**
- **Vector Similarity Search**: Uses PostgreSQL + pgvector for efficient similarity matching
- **Retrieval-Augmented Generation (RAG)**: Intelligent querying of simulation results
- **Temporal Graph Construction**: Builds causal relationships between simulation states
- **Real-time Telemetry Analysis**: Processes live reactor data for immediate insights

### 🧮 **Mathematical Optimization**
- **Matrix Orchestrator**: Python-based pipeline for mathematical operations
- **Julia Backend**: High-performance optimization algorithms (OSQP, Convex.jl, SCS)
- **Sparsity Optimization**: L1 regularization for efficient matrix operations
- **Rank Optimization**: Nuclear norm regularization for dimensionality reduction
- **Structure Optimization**: Graph Laplacian regularization for spatial relationships

### 🌡️ **Entropy & Information Theory**
- **Thermodynamic Analysis**: Monitors system stability and energy flow
- **Information Entropy**: Measures uncertainty and information content in data
- **Oscillation Detection**: Identifies periodic patterns in reactor behavior
- **Thermostatic Control**: Adaptive transformations based on entropy levels

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Fractal Cascade Simulation               │
├─────────────────────────────────────────────────────────────┤
│  🌊 Nuclear Physics Engine                                 │
│  ├── Lattice Physics Calculations                          │
│  ├── Neutron Transport Simulation                          │
│  └── Fuel Assembly Analysis                                │
├─────────────────────────────────────────────────────────────┤
│  🤖 ChaosRAGJulia (AI Backend)                            │
│  ├── Vector Database (PostgreSQL + pgvector)               │
│  ├── RAG System for Intelligent Querying                   │
│  ├── Temporal Graph Construction                           │
│  └── Real-time Data Ingestion                             │
├─────────────────────────────────────────────────────────────┤
│  🧮 Matrix Optimization Engine                             │
│  ├── Python Orchestrator                                   │
│  ├── Julia Backend (OSQP, Convex.jl, SCS)                 │
│  └── Multi-method Optimization                             │
├─────────────────────────────────────────────────────────────┤
│  🌡️ Entropy Engine                                         │
│  ├── Thermodynamic Analysis                                │
│  ├── Information Theory                                    │
│  └── Adaptive Control Systems                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Technology Stack

### **Backend Services**
- **Julia**: High-performance numerical computing and optimization
- **Python**: Orchestration, data processing, and ML pipelines
- **PostgreSQL**: Primary database with pgvector extension for vector operations
- **Ruby**: Additional integration layer and API gateway

### **AI & ML**
- **Vector Embeddings**: 1536-dimensional representations of simulation data
- **FFT Features**: Fast Fourier Transform magnitude features for pattern recognition
- **L2 Normalization**: Standardized vector representations for similarity search
- **Entropy Analysis**: Information-theoretic measures for system stability

### **Infrastructure**
- **Docker**: Containerized deployment for all services
- **Git LFS**: Large file storage for simulation data (>100MB files)
- **CI/CD**: Automated testing and deployment pipelines
- **Microservices**: Modular architecture for scalability

## 📊 What It Solves

### **Nuclear Engineering Challenges**
1. **Fuel Assembly Optimization**: Find optimal fuel rod configurations
2. **Reactor Safety Analysis**: Predict cascade failures and critical states
3. **Real-time Monitoring**: Continuous analysis of reactor telemetry
4. **Historical Pattern Recognition**: Learn from past simulation data

### **Scientific Research Applications**
1. **Parameter Space Exploration**: Efficiently explore vast simulation parameter spaces
2. **Cross-Simulation Correlation**: Find relationships between different simulation runs
3. **Anomaly Detection**: Identify unusual reactor behavior patterns
4. **Reproducibility Tracking**: Maintain temporal causality in simulation results

### **Industrial Use Cases**
1. **Reactor Design**: Optimize new reactor designs using historical data
2. **Predictive Maintenance**: Anticipate equipment failures and maintenance needs
3. **Quality Assurance**: Validate simulation results against known patterns
4. **Regulatory Compliance**: Document and track simulation methodologies

## 🚀 Quick Start

### **Prerequisites**
```bash
# System requirements
- Julia 1.8+
- Python 3.8+
- PostgreSQL 13+ with pgvector extension
- Git LFS for large files
```

### **Installation**
```bash
# Clone the repository
git clone https://github.com/9x25dillon/NuRea_sim.git
cd NuRea_sim

# Install Git LFS
git lfs install

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials

# Start ChaosRAGJulia server
cd chaos_rag_single
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia server_vector.jl

# Start Matrix Orchestrator
python matrix_orchestrator.py --plan plan.json
```

### **Data Processing Pipeline**
```bash
# Process lattice physics data
cd "lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)(1)"
python process_vectors.py

# Ingest data into database
./load_augmented.sh

# Create temporal edges
./entangle.sh http  # or ./entangle.sh sql
```

## 📁 Project Structure

```
Fractal_cascade_simulation/
├── chaos_rag_single/           # ChaosRAGJulia server implementations
├── lattice-physics*/           # Nuclear physics simulation data & scripts
├── matrix_orchestrator.py      # Python-based optimization orchestration
├── hrm_julia_backend/          # Julia mathematical optimization backend
├── entropy engine/             # Thermodynamic & information theory analysis
├── LiMPs/                      # Enterprise infrastructure & deployment
├── carryon/                    # Memory management & soulpack system
├── bridge.rb                   # Ruby integration layer
└── config/                     # Configuration files & settings
```

## 🔬 Key Features

### **Vector-Augmented Physics**
- **L2 Normalization**: Standardized vector representations
- **FFT Magnitude Features**: Pattern recognition capabilities
- **Entropy Augmentation**: Information-theoretic measures
- **1536-Dimensional Vectors**: Optimized for similarity search

### **Temporal Graph Construction**
- **Step-1 Edges**: i → i+1 temporal relationships (weight 1.0)
- **Step-5 Edges**: i → i+5 temporal relationships (weight 0.6)
- **Causal Inference**: Understanding of reactor state evolution
- **Cascade Analysis**: Prediction of failure propagation

### **Real-time Capabilities**
- **Live Data Ingestion**: Process streaming reactor telemetry
- **Immediate Analysis**: Real-time vector similarity search
- **Predictive Alerts**: Early warning of potential issues
- **Adaptive Responses**: Dynamic system adjustments

## 🌟 Why This Matters

### **Nuclear Safety**
- **Prevent Cascade Failures**: Understand how reactor states evolve
- **Real-time Monitoring**: Continuous safety analysis
- **Predictive Analytics**: Anticipate problems before they occur
- **Regulatory Compliance**: Comprehensive documentation and tracking

### **Scientific Advancement**
- **Multi-Physics Integration**: Combine different simulation domains
- **AI-Powered Insights**: Discover patterns humans might miss
- **Scalable Analysis**: Handle massive simulation datasets
- **Reproducible Research**: Maintain causal relationships in data

### **Industrial Applications**
- **Cost Reduction**: Optimize reactor designs and operations
- **Efficiency Gains**: Better fuel utilization and power output
- **Risk Mitigation**: Proactive identification of potential issues
- **Knowledge Preservation**: Maintain institutional knowledge

## 🤝 Contributing

This project welcomes contributions! See our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Areas for Contribution**
- **New Optimization Algorithms**: Extend the mathematical optimization capabilities
- **Additional Physics Models**: Support for different reactor types
- **Performance Improvements**: Optimize vector operations and database queries
- **Documentation**: Improve user guides and API documentation
- **Testing**: Expand test coverage and validation

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 9x25dillon**

## 🙏 Acknowledgments

- **Nuclear Engineering Community**: For domain expertise and validation
- **Julia Language**: For high-performance numerical computing
- **PostgreSQL + pgvector**: For efficient vector similarity search
- **Open Source Community**: For the tools and libraries that make this possible

---

**🚀 Ready to revolutionize nuclear physics simulation? Start exploring the Fractal Cascade Simulation today!**

*For questions, support, or collaboration, please open an issue or reach out to the maintainers.*