# Base CUDA image for GPU inference
FROM nvidia/cuda:12.1.105-base-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy app
COPY app/ /app

# Install Python deps
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
