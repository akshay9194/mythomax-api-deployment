# MythoMax API — Pod Mode (GGUF + llama-cpp-python)
# Uses pre-built CUDA wheel, no compilation needed
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/workspace/models

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Install pre-built llama-cpp-python with CUDA 12.1
RUN pip install --no-cache-dir \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Install FastAPI + deps
RUN pip install --no-cache-dir fastapi uvicorn[standard] boto3

# Copy application
COPY app/model_loader.py .
COPY app/server.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
