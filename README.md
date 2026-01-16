# MythoMax LLM API

Configurable MythoMax-L2-13B API for AI chat applications. Optimized for RunPod GPU deployment.

## Quick Deploy (RunPod)

### Start Command
```bash
pip install transformers accelerate bitsandbytes safetensors fastapi uvicorn protobuf sentencepiece && git clone https://github.com/akshay9194/mythomax-api-deployment.git /workspace/app && cd /workspace/app/app && uvicorn server:app --host 0.0.0.0 --port 8000
```

### Environment Variables (all optional)
| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | _(none)_ | Bearer token for auth (empty = no auth) |
| `MODEL_NAME` | `Gryphe/MythoMax-L2-13B` | HuggingFace model ID |
| `DEFAULT_MAX_TOKENS` | `512` | Max generation length |
| `DEFAULT_TEMPERATURE` | `0.85` | Creativity (0.0-1.0) |

### RunPod Setup
1. Create Network Volume (50GB) in your region
2. Deploy Pod with PyTorch template (RTX 4090 recommended)
3. Attach volume to `/workspace`
4. Set environment variables
5. Paste Start Command
6. Expose port 8000

## API Endpoints

### Health Check
```bash
GET /health
```

### Simple Chat
```bash
POST /chat
{
  "prompt": "Hello!",
  "system_prompt": "You are a friendly assistant.",
  "max_tokens": 256,
  "temperature": 0.8
}
```

### Chat with History (XinMate format)
```bash
POST /v1/chat
{
  "personality_prompt": "You are Scarlett, a flirty companion...",
  "user_message": "Hey there!",
  "chat_history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
  ]
}
```

### Response Format
```json
{
  "response": "Hey! How are you doing today?",
  "model": "Gryphe/MythoMax-L2-13B",
  "tokens_generated": 42
}
```

## Authentication

If `API_KEY` is set, include Bearer token:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}' \
  https://your-pod-8000.proxy.runpod.net/chat
```

## Local Development

```bash
# With Docker
docker build -t mythomax-api .
docker run --gpus all -p 8000:8000 -e API_KEY=test mythomax-api

# Without Docker (requires CUDA)
pip install -r app/requirements.txt
cd app && uvicorn server:app --reload
```

## Cost Estimates (RunPod)

| GPU | $/hour | 4-bit Model | Notes |
|-----|--------|-------------|-------|
| RTX 4090 | $0.44 | ✅ Fast | Best performance |
| RTX 4090 Spot | $0.29 | ✅ Fast | Can be interrupted |
| RTX 3090 | $0.22 | ✅ Good | Budget option |

Model download: ~15 min first time, then cached on network volume.

## License

MIT
