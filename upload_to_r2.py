"""
Upload large files to Cloudflare R2 using S3-compatible multipart upload.

Usage:
  python upload_to_r2.py

Required environment variables:
  R2_ACCOUNT_ID      - Your Cloudflare account ID
  R2_ACCESS_KEY_ID   - R2 API token access key
  R2_SECRET_ACCESS_KEY - R2 API token secret key
  R2_BUCKET_NAME     - Target R2 bucket name

The file will be uploaded with multipart (handles files > 300MB).
"""

import os
import sys
import boto3
from botocore.config import Config

# ── Config from env ──────────────────────────────────────────
ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID", "")
SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
BUCKET = "mythomax"  # os.getenv("R2_BUCKET_NAME", "")

# File to upload
FILE_PATH = os.path.join(os.path.dirname(__file__), "models", "mythomax-l2-13b.Q4_K_M.gguf")
R2_KEY = "models/mythomax-l2-13b.Q4_K_M.gguf"  # path inside the bucket

def main():
    # Validate
    missing = []
    if not ACCOUNT_ID: missing.append("R2_ACCOUNT_ID")
    if not ACCESS_KEY: missing.append("R2_ACCESS_KEY_ID")
    if not SECRET_KEY: missing.append("R2_SECRET_ACCESS_KEY")
    if not BUCKET: missing.append("R2_BUCKET_NAME")

    if missing:
        print(f"ERROR: Set these environment variables first: {', '.join(missing)}")
        print()
        print("  $env:R2_ACCOUNT_ID = 'your_cloudflare_account_id'")
        print("  $env:R2_ACCESS_KEY_ID = 'your_r2_access_key'")
        print("  $env:R2_SECRET_ACCESS_KEY = 'your_r2_secret_key'")
        print("  $env:R2_BUCKET_NAME = 'your_bucket_name'")
        sys.exit(1)

    if not os.path.exists(FILE_PATH):
        print(f"ERROR: File not found: {FILE_PATH}")
        print("  Download it first:")
        print("  python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('TheBloke/MythoMax-L2-13B-GGUF', 'mythomax-l2-13b.Q4_K_M.gguf', local_dir='./models')\"")
        sys.exit(1)

    file_size = os.path.getsize(FILE_PATH)
    print(f"File: {FILE_PATH}")
    print(f"Size: {file_size / (1024**3):.2f} GB")
    print(f"Bucket: {BUCKET}")
    print(f"Key: {R2_KEY}")
    print()

    # Create S3 client for R2
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
        region_name="auto",
    )

    # Multipart upload with progress
    from boto3.s3.transfer import TransferConfig

    config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,  # 100MB
        multipart_chunksize=100 * 1024 * 1024,   # 100MB chunks
        max_concurrency=4,
    )

    class ProgressCallback:
        def __init__(self, total):
            self.total = total
            self.uploaded = 0

        def __call__(self, bytes_transferred):
            self.uploaded += bytes_transferred
            pct = (self.uploaded / self.total) * 100
            bar = "=" * int(pct // 2) + ">" + " " * (50 - int(pct // 2))
            mb = self.uploaded / (1024**2)
            total_mb = self.total / (1024**2)
            sys.stdout.write(f"\r  [{bar}] {pct:.1f}% ({mb:.0f}/{total_mb:.0f} MB)")
            sys.stdout.flush()
            if self.uploaded >= self.total:
                print()

    print("Uploading to Cloudflare R2...")
    s3.upload_file(
        FILE_PATH,
        BUCKET,
        R2_KEY,
        Config=config,
        Callback=ProgressCallback(file_size),
    )

    print()
    print(f"Done! File uploaded to: r2://{BUCKET}/{R2_KEY}")
    print()

    # Generate the public URL (if bucket has public access)
    print(f"R2 URL: https://{BUCKET}.{ACCOUNT_ID}.r2.cloudflarestorage.com/{R2_KEY}")
    print()
    print("Use this R2_MODEL_URL in your handler's env vars on RunPod.")


if __name__ == "__main__":
    main()
