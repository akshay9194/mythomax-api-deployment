"""
Shared model loader for MythoMax GGUF.

Used by both server.py (Pod mode) and handler.py (Serverless mode).

Logic:
  1. Check MODEL_PATH for .gguf file
  2. If missing → download from Cloudflare R2
  3. Load with llama-cpp-python (GPU accelerated)

Env vars:
  MODEL_FILENAME       - GGUF file name (default: mythomax-l2-13b.Q4_K_M.gguf)
  MODEL_DIR            - Directory to store model (default: /workspace/models)
  GPU_LAYERS           - Number of layers on GPU, -1 = all (default: -1)
  R2_ACCOUNT_ID        - Cloudflare account ID
  R2_ACCESS_KEY_ID     - R2 access key
  R2_SECRET_ACCESS_KEY - R2 secret key
  R2_BUCKET_NAME       - R2 bucket name
  R2_MODEL_KEY         - Object key in bucket
"""

import os
import sys
import logging
import boto3
from botocore.config import Config
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "mythomax-l2-13b.Q4_K_M.gguf")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/workspace/models"))
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "-1"))

# R2 config
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.getenv("R2_BUCKET_NAME", "mythomax")
R2_MODEL_KEY = os.getenv("R2_MODEL_KEY", "models/mythomax-l2-13b.Q4_K_M.gguf")


def download_from_r2():
    """Download GGUF model from Cloudflare R2."""
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET]):
        missing = [k for k, v in {
            "R2_ACCOUNT_ID": R2_ACCOUNT_ID,
            "R2_ACCESS_KEY_ID": R2_ACCESS_KEY,
            "R2_SECRET_ACCESS_KEY": R2_SECRET_KEY,
            "R2_BUCKET_NAME": R2_BUCKET,
        }.items() if not v]
        raise RuntimeError(f"R2 credentials missing: {', '.join(missing)}")

    logger.info(f"Downloading model from R2: {R2_BUCKET}/{R2_MODEL_KEY}")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    head = s3.head_object(Bucket=R2_BUCKET, Key=R2_MODEL_KEY)
    total_size = head["ContentLength"]
    logger.info(f"Model size: {total_size / (1024**3):.2f} GB")

    downloaded = 0
    last_pct = -1

    def progress(bytes_transferred):
        nonlocal downloaded, last_pct
        downloaded += bytes_transferred
        pct = int((downloaded / total_size) * 100)
        if pct >= last_pct + 10:
            last_pct = pct
            logger.info(f"Download: {pct}% ({downloaded // (1024**2)} MB / {total_size // (1024**2)} MB)")

    s3.download_file(R2_BUCKET, R2_MODEL_KEY, str(MODEL_PATH), Callback=progress)
    logger.info(f"Model downloaded: {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / (1024**3):.2f} GB)")


def load_model():
    """Check for model, download if missing, load with llama.cpp."""
    from llama_cpp import Llama

    # Step 1: Check local
    if MODEL_PATH.exists():
        size = os.path.getsize(MODEL_PATH)
        logger.info(f"Model found: {MODEL_PATH} ({size / (1024**3):.2f} GB)")
    else:
        # Step 2: Download from R2
        logger.info(f"Model not found at {MODEL_PATH}, downloading from R2...")
        download_from_r2()

    # Step 3: Load
    logger.info(f"Loading model with {GPU_LAYERS} GPU layers...")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=GPU_LAYERS,
            n_ctx=4096,
            n_batch=512,
            verbose=True,  # Show llama.cpp internal logs for debugging
        )
    except Exception as e:
        logger.error(f"GPU load failed: {e}")
        logger.info("Retrying with CPU only (n_gpu_layers=0)...")
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=0,
            n_ctx=4096,
            n_batch=512,
            verbose=True,
        )
    logger.info("Model loaded and ready!")
    return llm
