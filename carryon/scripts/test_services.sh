#!/bin/bash

echo "🧪 Testing Kiln Orchestrator Services"
echo "====================================="

# Test chaos RAG service
echo -e "\n🔮 Testing Chaos RAG Service..."
if curl -s http://localhost:8081/chaos/rag/query > /dev/null 2>&1; then
    echo "✅ Chaos RAG service is running on port 8081"
else
    echo "❌ Chaos RAG service is not responding on port 8081"
fi

# Test Suno API service
echo -e "\n🎵 Testing Suno API Service..."
if curl -s http://localhost:8001/cookies > /dev/null 2>&1; then
    echo "✅ Suno API service is running on port 8001"
else
    echo "❌ Suno API service is not responding on port 8001"
fi

# Test desktop app service
echo -e "\n🖥️  Testing Desktop App Service..."
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "✅ Desktop app service is running on port 5173"
else
    echo "❌ Desktop app service is not responding on port 5173"
fi

# Test PostgreSQL
echo -e "\n🗄️  Testing PostgreSQL..."
if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "✅ PostgreSQL is running on port 5432"
else
    echo "❌ PostgreSQL is not responding on port 5432"
fi

echo -e "\n🎯 Test Summary:"
echo "Run './bin/kiln service health <service_name>' for detailed health checks"
echo "Run './bin/kiln <command> --help' for usage information"
