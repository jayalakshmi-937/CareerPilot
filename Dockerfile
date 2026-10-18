# ==============================================================================
# CareerPilot: Agentic AI for Personalized Placement Preparation
# Production Dockerfile — Multi-Service Container (FastAPI + Streamlit)
# ==============================================================================

FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install essential system packages and healthcheck dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set standard application workspace
WORKDIR /app

# Copy dependency definition and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application modules and placement assets
COPY agent/ ./agent/
COPY backend/ ./backend/
COPY data/ ./data/
COPY database/ ./database/
COPY frontend/ ./frontend/
COPY interview/ ./interview/
COPY placement/ ./placement/
COPY rag/ ./rag/
COPY pytest.ini ./pytest.ini
COPY README.md ./README.md

# Create persistent storage mount directory for SQLite and ChromaDB
RUN mkdir -p /app/data_storage/chroma_db /app/data_storage/documents

# Expose API port (FastAPI: 8000) and Web Dashboard port (Streamlit: 8501)
EXPOSE 8000 8501

# Default execution: Launch FastAPI Backend
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
