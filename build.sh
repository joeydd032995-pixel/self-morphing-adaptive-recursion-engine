#!/bin/bash
# Build script for Self-Morphing Adaptive Recursion Engine

set -e

echo "🔨 Building Self-Morphing Adaptive Recursion Engine Docker image..."

docker-compose build --no-cache

echo "✅ Build complete!"
echo "You can now run: ./start.sh or docker-compose up"