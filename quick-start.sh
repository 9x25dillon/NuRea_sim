#!/bin/bash
# NuRea Simulation Quick Start Script
# One-command setup and demo

set -e

echo "🚀 NuRea Simulation Quick Start"
echo "================================"

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Prerequisites met"

# Create necessary directories
echo "📁 Setting up directory structure..."
mkdir -p data artifacts
echo "✅ Directories created"

# Check if we have a plan.json
if [ ! -f "plan.json" ]; then
    echo "❌ plan.json not found. Please ensure you're in the NuRea_sim directory."
    exit 1
fi

echo "✅ Configuration files found"

# Start services
echo "🐳 Starting NuRea services..."
echo "   This will start:"
echo "   - PostgreSQL with pgvector"
echo "   - Julia Backend (matrix optimization)"
echo "   - ChaosRAGJulia (vector database)"
echo "   - Python Orchestrator (pipeline execution)"
echo ""

# Use docker compose (newer) or docker-compose (older)
if docker compose version &> /dev/null; then
    docker compose up --build -d
else
    docker-compose up --build -d
fi

echo "✅ Services started in background"

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check Julia backend
if curl -s http://localhost:9000/healthz > /dev/null; then
    echo "✅ Julia Backend: Healthy"
else
    echo "❌ Julia Backend: Not responding"
fi

# Check ChaosRAGJulia
if curl -s http://localhost:8081/health > /dev/null; then
    echo "✅ ChaosRAGJulia: Healthy"
else
    echo "❌ ChaosRAGJulia: Not responding"
fi

# Check orchestrator logs
echo "📊 Checking orchestrator status..."
if docker compose logs orchestrator 2>/dev/null | grep -q "Pipeline execution complete"; then
    echo "✅ Orchestrator: Pipeline completed successfully"
else
    echo "⏳ Orchestrator: Still running or not started"
fi

echo ""
echo "🎉 Quick start complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Check results: ls -la artifacts/"
echo "   2. View logs: docker compose logs -f"
echo "   3. Stop services: docker compose down"
echo "   4. Customize: Edit plan.json for different pipelines"
echo ""
echo "🌐 Service URLs:"
echo "   - Julia Backend: http://localhost:9000"
echo "   - ChaosRAGJulia: http://localhost:8081"
echo "   - PostgreSQL: localhost:5432"
echo ""
echo "📚 Documentation: See SETUP_GUIDE.md for detailed instructions"
echo ""
echo "🚀 Ready to revolutionize nuclear physics simulation!"
