# ═════════════════════════════════════════════════════════════════
# Multi-Agent Research Assistant — Production Dockerfile
# Supports dual-backend runtime: Local (GPU) or Gemini (CPU/Cloud)
# ═════════════════════════════════════════════════════════════════

FROM python:3.12-slim AS base

# Install system dependencies required for FAISS, PDF parsing, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    LLM_BACKEND=local

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configs
COPY config.yaml .
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user for container security
RUN useradd -m -u 1000 appuser && \
    mkdir -p data models logs && \
    chown -R appuser:appuser /app

USER appuser

# Expose API and Web Dashboard port (defaults to 8080, avoiding Redis port 8001)
EXPOSE 8080

# Health check to monitor service availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Launch the FastAPI application
CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8080"]
