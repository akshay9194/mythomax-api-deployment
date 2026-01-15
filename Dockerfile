# Base CUDA image for GPU inference
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy app
COPY app/ /app

# Install Python deps
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
