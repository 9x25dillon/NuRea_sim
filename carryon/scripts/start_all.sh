#!/bin/bash

echo "🚀 Starting Kiln Orchestrator Services"
echo "======================================"

# Check if we're in the kiln directory
if [ ! -f "bin/kiln" ]; then
    echo "❌ Please run this script from the kiln directory"
    exit 1
fi

# Start PostgreSQL first (if using Docker)
echo -e "\n🗄️  Starting PostgreSQL..."
if command -v docker &> /dev/null; then
    docker run -d --name kiln-postgres \
        -e POSTGRES_DB=chaos \
        -e POSTGRES_USER=user \
        -e POSTGRES_PASSWORD=pass \
        -p 5432:5432 \
        postgres:15
    echo "✅ PostgreSQL started in Docker"
else
    echo "⚠️  Docker not found, assuming PostgreSQL is running locally"
fi

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 5

# Start Chaos RAG service
echo -e "\n🔮 Starting Chaos RAG Service..."
./bin/kiln service start chaos_rag &
CHAOS_PID=$!
echo "✅ Chaos RAG started (PID: $CHAOS_PID)"

# Wait for Chaos RAG to be healthy
echo "⏳ Waiting for Chaos RAG to be healthy..."
sleep 10

# Start Suno API service
echo -e "\n🎵 Starting Suno API Service..."
./bin/kiln service start suno_api &
SUNO_PID=$!
echo "✅ Suno API started (PID: $SUNO_PID)"

# Wait for Suno API to be healthy
echo "⏳ Waiting for Suno API to be healthy..."
sleep 10

# Start Desktop App service
echo -e "\n🖥️  Starting Desktop App Service..."
./bin/kiln service start desktop_app &
DESKTOP_PID=$!
echo "✅ Desktop App started (PID: $DESKTOP_PID)"

# Wait for all services to be ready
echo -e "\n⏳ Waiting for all services to be ready..."
sleep 15

# Test all services
echo -e "\n🧪 Testing all services..."
./scripts/test_services.sh

echo -e "\n🎯 All services started!"
echo "Service PIDs:"
echo "  Chaos RAG: $CHAOS_PID"
echo "  Suno API:  $SUNO_PID"
echo "  Desktop:   $DESKTOP_PID"
echo ""
echo "Use './bin/kiln service health <service_name>' to check health"
echo "Use './bin/kiln <command> --help' for usage information"
