# Dockerfile for Self-Morphing Adaptive Recursion Engine
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (e.g., for torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY organized_self_morphing_engine.py fastapi_server.py cli.py .

# API port (default 8000 in code)
EXPOSE 8000

CMD ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
