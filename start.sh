#!/bin/bash
# Start script for Self-Morphing Adaptive Recursion Engine (Docker Compose)

set -e

echo "🚀 Starting Self-Morphing Adaptive Recursion Engine stack..."

# Create data directory if it doesn't exist
mkdir -p data

# Start services
docker-compose up --build -d

echo "✅ Stack started!"
echo "   - API: http://localhost:8000 (Swagger: http://localhost:8000/docs)"
echo "   - Neo4j Browser: http://localhost:7474"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"