import os
import torch
import asyncio
import logging
import traceback
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path

app = FastAPI(title="MythoMax API for XinMate")

# -------------------------------
# Configuration
# -------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Gryphe/MythoMax-L2-13B")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
API_KEY = os.environ.get("API_KEY", "")  # Optional: leave empty to disable auth
MODEL_DIR_ENV = os.environ.get("MODEL_DIR", None)

# Persistent volume mount (default path used in RunPod templates)
MODEL_VOLUME = Path(os.environ.get("MODEL_VOLUME_PATH", "/runpod-volume/models/mythomax"))
try:
    MODEL_VOLUME.mkdir(parents=True, exist_ok=True)
except Exception:
    pass  # May fail if path doesn't exist yet

# Application state
app.state.pipe = None
app.state.tokenizer = None
app.state.model_ready = False
app.state.load_error = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# -------------------------------
# Request Models
# -------------------------------
class GenRequest(BaseModel):
    """Simple generation request (legacy)"""
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

class ChatRequest(BaseModel):
    """XinMate-compatible chat request"""
    bot_id: str = "xinmate"
    personality_prompt: str  # System prompt with persona
    user_message: str  # Latest user message
    chat_history: List[Dict[str, str]] = []  # Previous messages
    max_tokens: int = 512
    temperature: float = 0.85
    top_p: float = 0.9
    model_name: Optional[str] = None

# -------------------------------
# Security Middleware (Optional)
# -------------------------------
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Skip auth if no API_KEY is set
    if not API_KEY:
        return await call_next(request)
    
    # Skip auth for health/docs endpoints
    if request.url.path in ["/health", "/", "/docs", "/openapi.json", "/status", "/models"]:
        return await call_next(request)
    
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return await call_next(request)

# -------------------------------
# Helper: determine model path
# -------------------------------
def resolve_model_path():
    # Priority: explicit MODEL_DIR env -> mounted volume if non-empty -> MODEL_NAME (remote)
    if MODEL_DIR_ENV:
        p = Path(MODEL_DIR_ENV)
        logging.info("Using MODEL_DIR env: %s", p)
        return str(p)
    try:
        if MODEL_VOLUME.exists() and any(MODEL_VOLUME.iterdir()):
            logging.info("Using mounted model volume: %s", MODEL_VOLUME)
            return str(MODEL_VOLUME)
    except Exception:
        logging.warning("Unable to read MODEL_VOLUME, falling back to MODEL_NAME")
    logging.info("Using remote MODEL_NAME: %s", MODEL_NAME)
    return MODEL_NAME

# -------------------------------
# Helper: Build prompt from chat
# -------------------------------
def build_chat_prompt(personality_prompt: str, user_message: str, chat_history: List[Dict[str, str]]) -> str:
    """Build Alpaca-style prompt for MythoMax"""
    prompt_parts = []
    
    # System/personality prompt
    if personality_prompt:
        prompt_parts.append(f"### Instruction:\n{personality_prompt}\n")
    
    # Add chat history (limit to prevent context overflow)
    max_history = 10
    recent_history = chat_history[-max_history:] if len(chat_history) > max_history else chat_history
    
    for msg in recent_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user":
            prompt_parts.append(f"### Input:\n{content}\n")
        elif role == "assistant":
            prompt_parts.append(f"### Response:\n{content}\n")
    
    # Current user message
    prompt_parts.append(f"### Input:\n{user_message}\n")
    prompt_parts.append("### Response:\n")
    
    return "\n".join(prompt_parts)

# -------------------------------
# Background model loader (non-blocking startup)
# -------------------------------
async def _load_model_async(retries: int = 1, retry_delay: int = 10):
    """Load model in background. Sets app.state.model_ready and app.state.load_error."""
    for attempt in range(1, retries + 2):
        try:
            logging.info("Starting model load attempt %d: %s", attempt, MODEL_NAME)

            # Prefer bitsandbytes quantization if available
            bnb_config = None
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logging.info("BitsAndBytesConfig created, will attempt 4-bit load")
            except Exception as e:
                logging.warning("bitsandbytes / BitsAndBytesConfig not available or failed: %s", e)
                bnb_config = None

            hf_kwargs = {}
            if HF_TOKEN:
                hf_kwargs["use_auth_token"] = HF_TOKEN

            model_path = resolve_model_path()

            # Load tokenizer first (works for local dirs or remote repos)
            tokenizer = AutoTokenizer.from_pretrained(model_path, **hf_kwargs, use_fast=True)
            app.state.tokenizer = tokenizer
            logging.info("Tokenizer loaded")

            # Try quantized load if possible, otherwise fall back
            if bnb_config is not None:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        quantization_config=bnb_config,
                        trust_remote_code=True,
                        **hf_kwargs,
                    )
                    logging.info("Loaded model with quantization")
                except Exception as e:
                    logging.warning("Quantized load failed, falling back to non-quantized load: %s", e)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        trust_remote_code=True,
                        **hf_kwargs,
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    **hf_kwargs,
                )

            # Determine pipeline device (-1 for CPU, 0 for first CUDA device)
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
            app.state.pipe = pipe
            app.state.model_ready = True
            app.state.load_error = None
            
            # Log GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logging.info(f"✓ Model loaded on {gpu_name} ({gpu_mem:.1f}GB)")
            else:
                logging.info("✓ Model loaded on CPU")
            return

        except Exception as e:
            tb = traceback.format_exc()
            app.state.load_error = tb
            app.state.model_ready = False
            logging.exception("Model load attempt %d failed", attempt)
            if attempt <= retries:
                logging.info("Retrying in %d seconds...", retry_delay)
                await asyncio.sleep(retry_delay)
            else:
                logging.error("All model load attempts failed")
                return

# -------------------------------
# Startup Event: schedule background load
# -------------------------------
@app.on_event("startup")
def schedule_model_load():
    # Schedule the async loader without blocking startup
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_load_model_async(retries=1, retry_delay=10))
        logging.info("Scheduled background model load task")
    except RuntimeError:
        # If no running loop (e.g., when running sync tests), run directly
        logging.info("No running loop, running loader synchronously")
        asyncio.run(_load_model_async(retries=1, retry_delay=10))

# -------------------------------
# Health / Status Endpoints
# -------------------------------
@app.get("/")
async def root():
    return {
        "service": "MythoMax API for XinMate",
        "model": MODEL_NAME,
        "status": "ready" if app.state.model_ready else "loading",
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if app.state.model_ready else "loading",
        "model_ready": app.state.model_ready,
        "error": app.state.load_error,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

@app.get("/status")
async def status():
    return {
        "status": "ready" if app.state.model_ready else "loading",
        "activeInferences": 0,
        "queueLength": 0,
    }

@app.get("/models")
async def list_models():
    return {"models": [MODEL_NAME], "default": MODEL_NAME}

# -------------------------------
# Chat Endpoint (XinMate format)
# -------------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    """XinMate-compatible chat endpoint"""
    if not app.state.model_ready:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not ready: {app.state.load_error or 'still loading'}"
        )

    pipe = app.state.pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model pipeline unavailable")

    try:
        # Build prompt from chat format
        prompt = build_chat_prompt(
            req.personality_prompt,
            req.user_message,
            req.chat_history,
        )
        
        logging.debug(f"Generated prompt length: {len(prompt)} chars")

        start_time = time.time()
        
        output = pipe(
            prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            return_full_text=False,
            truncation=True,
            pad_token_id=app.state.tokenizer.eos_token_id if app.state.tokenizer else None,
        )

        response_time = time.time() - start_time
        
        # Extract and clean generated text
        generated_text = output[0]["generated_text"] if isinstance(output, list) else str(output)
        generated_text = generated_text.strip()
        
        # Remove any prompt markers that might leak through
        for marker in ["### Input:", "### Response:", "### Instruction:", "###"]:
            if marker in generated_text:
                generated_text = generated_text.split(marker)[0].strip()

        tokens_generated = int(len(generated_text.split()) * 1.3)

        logging.info(f"Generated ~{tokens_generated} tokens in {response_time:.2f}s")

        return {
            "response": generated_text,
            "model_used": MODEL_NAME,
            "tokens_generated": tokens_generated,
            "response_time": round(response_time, 2),
        }

    except Exception as e:
        logging.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Generation Endpoint (Legacy)
# -------------------------------
@app.post("/generate")
async def generate(req: GenRequest):
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet: %s" % ("no error" if not app.state.load_error else "check /health"))

    pipe = app.state.pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model pipeline unavailable")

    out = pipe(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=True,
        return_full_text=False,
        truncation=True,
    )

    text = out[0]["generated_text"] if isinstance(out, list) else str(out)
    return {"response": text}
