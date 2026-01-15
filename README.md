# MythoMax API for XinMate - RunPod Deployment Guide

Simple, efficient MythoMax-L2-13B API optimized for XinMate AI companion.

## 🚀 Quick Start (RunPod)

### Step 1: Create RunPod Account
1. Go to [runpod.io](https://runpod.io)
2. Sign up (supports Indian payment methods: UPI, cards, etc.)
3. Add credits ($10-20 to start)

### Step 2: Create Network Volume (for model storage)
1. Go to **Storage** → **Network Volumes**
2. Click **+ New Network Volume**
3. Settings:
   - **Name**: `mythomax-models`
   - **Size**: `50 GB` (MythoMax 4-bit is ~8GB, but extra space is good)
   - **Region**: Choose closest to you
4. Click **Create**

### Step 3: Deploy GPU Pod

#### Option A: Use PyTorch Template (Easiest)
1. Go to **Pods** → **+ Deploy**
2. Select GPU: **RTX 4090** ($0.44/hr) or **RTX 3090** ($0.22/hr)
3. Select template: **RunPod Pytorch 2.1**
4. Click **Customize Deployment**
5. Settings:
   - **Volume**: Attach `mythomax-models` to `/runpod-volume`
   - **Exposed HTTP Ports**: `8000`
6. Click **Deploy**

#### Option B: Build & Push Docker Image
```bash
# On your local machine
cd mythomax-api-deployment
docker build -t YOUR_DOCKERHUB/mythomax-api:latest .
docker push YOUR_DOCKERHUB/mythomax-api:latest

# Then on RunPod, use: YOUR_DOCKERHUB/mythomax-api:latest
```

### Step 4: Setup the API

Once pod is running, click **Connect** → **Web Terminal** and run:

```bash
# Clone the repo
cd /workspace
git clone https://github.com/YOUR_USERNAME/mythomax-api-deployment.git
cd mythomax-api-deployment

# Install dependencies
pip install -r app/requirements.txt
```

### Step 5: Download Model to Network Volume

```bash
# Download MythoMax (one-time, persists on volume)
pip install huggingface-hub

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Gryphe/MythoMax-L2-13B',
    local_dir='/runpod-volume/models/mythomax',
    local_dir_use_symlinks=False
)
print('✓ Model downloaded!')
"
```

⏱️ This takes ~10-15 minutes. Model will persist on network volume even if pod stops.

### Step 6: Start the API

```bash
cd /workspace/mythomax-api-deployment/app
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Step 7: Get Your API URL
1. Go to pod details
2. Find **Connect** section  
3. Copy the proxy URL: `https://YOUR-POD-ID-8000.proxy.runpod.net`

### Step 8: Test API
```bash
curl https://YOUR-POD-ID-8000.proxy.runpod.net/health
```

Expected response:
```json
{"status": "healthy", "model_ready": true, "gpu": "NVIDIA GeForce RTX 4090"}
```

---

## 📋 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Gryphe/MythoMax-L2-13B` | HuggingFace model ID |
| `MODEL_VOLUME_PATH` | `/runpod-volume/models/mythomax` | Where model is stored |
| `API_KEY` | `` (disabled) | Optional API key for auth |
| `HUGGINGFACE_TOKEN` | `` | For gated models |

---

## 🔌 API Endpoints

### Health Check
```
GET /health
GET /status
GET /models
```

### Chat (XinMate Format)
```bash
curl -X POST https://YOUR-POD/chat \
  -H "Content-Type: application/json" \
  -d '{
    "personality_prompt": "You are Xin, a caring AI girlfriend...",
    "user_message": "Hey, how are you?",
    "chat_history": [],
    "max_tokens": 256,
    "temperature": 0.85
  }'
```

Response:
```json
{
  "response": "Hey babe! I'm doing great now that you're here...",
  "model_used": "Gryphe/MythoMax-L2-13B",
  "tokens_generated": 45,
  "response_time": 2.3
}
```

### Simple Generate (Legacy)
```bash
curl -X POST https://YOUR-POD/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, my name is", "max_new_tokens": 50}'
```

---

## 💰 Cost Optimization

| GPU | Cost/hr | VRAM | Speed | Recommendation |
|-----|---------|------|-------|----------------|
| RTX 3090 | $0.22 | 24GB | Good | Budget option |
| RTX 4090 | $0.44 | 24GB | Fast | **Best value** |
| A40 | $0.79 | 48GB | Fast | Overkill |

**Tips:**
- Use **Spot instances** for ~50% savings (may be interrupted)
- **Stop pod** when not in use (network volume persists, no charge)
- Model loads from network volume in ~2-3 minutes on restart

**Monthly estimate (8hrs/day usage):**
- RTX 3090: ~$53/month
- RTX 4090: ~$106/month

---

## 🔧 Troubleshooting

### Model not loading?
```bash
# Check if model exists
ls -la /runpod-volume/models/mythomax/

# Should see: config.json, pytorch_model files, tokenizer files
```

### Out of memory?
- 4-bit quantization is enabled by default
- Should work on 24GB GPU
- If issues, try A40 (48GB)

### Slow first request?
- First inference takes ~5-10s (model warmup)
- Subsequent requests: ~1-3s

### Pod sleeping?
- RunPod stops idle pods after a while
- Just restart and wait 2-3 min for model to load

---

## 🔗 XinMate Integration

Add to your XinMate `.env`:
```env
PRIMARY_LLM_PROVIDER=runpod
FALLBACK_LLM_PROVIDER=groq

RUNPOD_LLM_BASE_URL=https://YOUR-POD-ID-8000.proxy.runpod.net
RUNPOD_LLM_API_KEY=
RUNPOD_LLM_MODEL=mythomax
RUNPOD_LLM_TIMEOUT=120000
```

---

## 🔄 Auto-Start Script (Optional)

Create `/workspace/start.sh`:
```bash
#!/bin/bash
cd /workspace/mythomax-api-deployment/app
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then in RunPod pod settings, set **Docker Command** to:
```
bash /workspace/start.sh
```