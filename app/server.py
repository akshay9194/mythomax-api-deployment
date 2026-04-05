"""
MythoMax API Server (GGUF) — Pod Mode

FastAPI server for RunPod GPU Pods.
Loads pre-quantized GGUF model from network volume (or downloads from R2).
Uses llama-cpp-python for GPU-accelerated inference.

Endpoints:
  GET  /health   — health check
  GET  /status   — worker status
  POST /chat     — generate text from prompt
  POST /generate — alias for /chat

Environment Variables:
  MODEL_DIR            - Model directory (default: /workspace/models)
  MODEL_FILENAME       - GGUF filename (default: mythomax-l2-13b.Q4_K_M.gguf)
  API_KEY              - Bearer token for auth (optional, empty = no auth)
  DEFAULT_MAX_TOKENS   - Max generation length (default: 512)
  DEFAULT_TEMPERATURE  - Creativity 0.0-1.0 (default: 0.85)
  DEFAULT_TOP_P        - Nucleus sampling (default: 0.9)
  REPETITION_PENALTY   - Repetition penalty (default: 1.1)
  GPU_LAYERS           - GPU layers, -1 = all (default: -1)
  R2_*                 - Cloudflare R2 credentials (for auto-download if model missing)
"""

import os
import re
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="MythoMax API", version="3.0.0")
security = HTTPBearer(auto_error=False)

# ── Config ───────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.85"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# ── Request/Response ─────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_generated: int

# ── Auth ─────────────────────────────────────────────────────

async def verify_api_key(creds: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEY:
        return True
    if not creds or creds.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# ── Model Loading (uses shared model_loader.py) ─────────────

@app.on_event("startup")
async def startup():
    try:
        from model_loader import load_model
        app.state.llm = load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        app.state.llm = None

# ── Response Cleaning ────────────────────────────────────────

def clean_response(text: str) -> str:
    if not text:
        return ""
    lines = text.strip().split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[A-Z][a-z]+:\s*', '', line)
        line = re.sub(r'^(User|Human|Assistant|Response|Input|Instruction):\s*', '', line, flags=re.IGNORECASE)
        if not line.strip():
            continue
        if cleaned and line.strip() == cleaned[-1].strip():
            continue
        cleaned.append(line)
    result = '\n'.join(cleaned).strip()
    for marker in ["### Input:", "### Response:", "### Instruction:", "### Human:", "### Assistant:", "###"]:
        if marker in result:
            result = result.split(marker)[0].strip()
    return result

# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "MythoMax API",
        "version": "3.0.0",
        "format": "GGUF (Q4_K_M)",
        "endpoints": ["/health", "/chat", "/generate"],
    }

@app.get("/health")
async def health():
    model_ready = hasattr(app.state, 'llm') and app.state.llm is not None
    return {
        "status": "healthy" if model_ready else "loading",
        "model": "MythoMax-L2-13B-Q4_K_M",
        "model_ready": model_ready,
    }

@app.get("/status")
async def status():
    return {
        "status": "ready" if (hasattr(app.state, 'llm') and app.state.llm) else "loading",
        "activeInferences": 0,
        "queueLength": 0,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: bool = Depends(verify_api_key)):
    if not hasattr(app.state, 'llm') or not app.state.llm:
        raise HTTPException(status_code=503, detail="Model not loaded")

    max_tokens = req.max_tokens or DEFAULT_MAX_TOKENS
    temperature = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
    top_p = req.top_p if req.top_p is not None else DEFAULT_TOP_P

    output = app.state.llm(
        req.prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=REPETITION_PENALTY,
        stop=["### Input:", "### Instruction:", "### Response:", "User:", "Human:"],
    )

    raw = output["choices"][0]["text"]
    token_count = output["usage"]["completion_tokens"]
    cleaned = clean_response(raw)

    return ChatResponse(
        response=cleaned,
        model="MythoMax-L2-13B-Q4_K_M",
        tokens_generated=token_count,
    )

@app.post("/generate")
async def generate(req: ChatRequest, _: bool = Depends(verify_api_key)):
    return await chat(req, _)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
