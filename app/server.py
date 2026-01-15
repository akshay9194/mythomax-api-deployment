"""
MythoMax API Server - Configurable LLM API for XinMate and general use

All settings configurable via environment variables.
Supports both simple chat format and XinMate's format.

Environment Variables:
  MODEL_NAME         - HuggingFace model ID (default: Gryphe/MythoMax-L2-13B)
  MODEL_VOLUME_PATH  - Path to cache models (default: /workspace/models)
  API_KEY            - Bearer token for auth (optional, empty = no auth)
  MAX_HISTORY        - Max chat history messages to include (default: 10)
  DEFAULT_MAX_TOKENS - Default max tokens (default: 512)
  DEFAULT_TEMPERATURE- Default temperature (default: 0.85)
  DEFAULT_TOP_P      - Default top_p (default: 0.9)
  REPETITION_PENALTY - Repetition penalty (default: 1.1)
"""

import os
import re
import logging
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="MythoMax API", version="2.0.0")
security = HTTPBearer(auto_error=False)

# ═══════════════════════════════════════════════════════════════
# Configuration from Environment Variables
# ═══════════════════════════════════════════════════════════════
MODEL_NAME = os.getenv("MODEL_NAME", "Gryphe/MythoMax-L2-13B")
MODEL_VOLUME = Path(os.getenv("MODEL_VOLUME_PATH", "/workspace/models"))
API_KEY = os.getenv("API_KEY", "")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.85"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# ═══════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════

class SimpleChatRequest(BaseModel):
    """Simple chat request format"""
    prompt: str
    system_prompt: Optional[str] = "You are a helpful assistant."
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

class XinMateChatRequest(BaseModel):
    """XinMate-compatible chat request with history"""
    bot_id: str = "xinmate"
    personality_prompt: str
    user_message: str
    chat_history: List[Dict[str, str]] = []
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

class ChatResponse(BaseModel):
    """Unified chat response"""
    response: str
    model: str
    tokens_generated: int

# ═══════════════════════════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════════════════════════

async def verify_api_key(creds: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token if API_KEY is configured"""
    if not API_KEY:
        return True  # No auth required
    if not creds or creds.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# ═══════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        MODEL_VOLUME.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading {MODEL_NAME}...")
        logger.info(f"Cache directory: {MODEL_VOLUME}")
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        app.state.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(MODEL_VOLUME),
            trust_remote_code=True,
        )
        logger.info("Tokenizer loaded")
        
        # Load model with quantization
        app.state.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=str(MODEL_VOLUME),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True,
        )
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Model loaded on {gpu_name}")
        else:
            logger.info("Model loaded on CPU")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        app.state.model = None
        app.state.tokenizer = None

# ═══════════════════════════════════════════════════════════════
# Response Cleaning
# ═══════════════════════════════════════════════════════════════

def clean_response(text: str) -> str:
    """Clean up model response - remove artifacts, name prefixes, and self-dialogue"""
    if not text:
        return ""
    
    # Split into lines for processing
    lines = text.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Remove character name prefix at start (e.g., "Scarlett:", "Kira:", "Emma:")
        line = re.sub(r'^[A-Z][a-z]+:\s*', '', line)
        
        # Remove User/Human/Assistant markers
        line = re.sub(r'^(User|Human|Assistant|Response|Input|Instruction):\s*', '', line, flags=re.IGNORECASE)
        
        # Skip if line is now empty
        if not line.strip():
            continue
            
        # Skip if this looks like the model asking itself a question then answering
        # (lines that start with common question patterns after a previous response)
        if cleaned_lines and re.match(r'^(How|What|Why|When|Where|Do you|Have you|Are you|Can you|Would you)', line):
            # Check if previous line was a complete thought (ends with . ! ? or emoji)
            prev = cleaned_lines[-1].strip()
            if prev and (prev[-1] in '.!?😊😘💕🥰😏' or prev.endswith(')')):
                # This might be a self-asked question - skip it
                continue
        
        # Skip duplicate content
        if cleaned_lines and line.strip() == cleaned_lines[-1].strip():
            continue
            
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    # Final cleanup: remove any trailing prompt markers
    for marker in ["### Input:", "### Response:", "### Instruction:", "### Human:", "### Assistant:", "###"]:
        if marker in result:
            result = result.split(marker)[0].strip()
    
    # If we ended up with nothing, return the original first line cleaned
    if not result and text.strip():
        first_line = text.strip().split('\n')[0]
        result = re.sub(r'^[A-Z][a-z]+:\s*', '', first_line).strip()
    
    return result

# ═══════════════════════════════════════════════════════════════
# Generation Logic
# ═══════════════════════════════════════════════════════════════

def generate_response(prompt: str, max_tokens: int, temperature: float, top_p: float):
    """Generate response from prompt, returns (text, token_count)"""
    
    inputs = app.state.tokenizer(prompt, return_tensors="pt").to(app.state.model.device)
    
    with torch.no_grad():
        outputs = app.state.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=app.state.tokenizer.eos_token_id,
            repetition_penalty=REPETITION_PENALTY,
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    raw_response = app.state.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return raw_response, len(new_tokens)

# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Service info"""
    return {
        "service": "MythoMax API",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "endpoints": ["/health", "/chat", "/v1/chat", "/generate"],
    }

@app.get("/health")
async def health():
    """Health check"""
    model_ready = hasattr(app.state, 'model') and app.state.model is not None
    return {
        "status": "healthy" if model_ready else "loading",
        "model": MODEL_NAME,
        "model_ready": model_ready,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

@app.get("/status")
async def status():
    """Status for scaling checks"""
    return {
        "status": "ready" if (hasattr(app.state, 'model') and app.state.model) else "loading",
        "activeInferences": 0,
        "queueLength": 0,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: SimpleChatRequest, _: bool = Depends(verify_api_key)):
    """
    Simple chat endpoint - receives ready-to-use prompt
    
    Request body:
    - prompt: Full formatted prompt (built by caller)
    - system_prompt: Optional additional context
    - max_tokens, temperature, top_p: Generation params (optional)
    """
    if not hasattr(app.state, 'model') or not app.state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # If system_prompt provided, prepend it; otherwise use prompt as-is
    if req.system_prompt and req.system_prompt != "You are a helpful assistant.":
        full_prompt = f"{req.system_prompt}\n\n{req.prompt}"
    else:
        full_prompt = req.prompt
    
    # Use defaults from env if not provided
    max_tokens = req.max_tokens or DEFAULT_MAX_TOKENS
    temperature = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
    top_p = req.top_p if req.top_p is not None else DEFAULT_TOP_P
    
    raw_response, token_count = generate_response(full_prompt, max_tokens, temperature, top_p)
    cleaned = clean_response(raw_response)
    
    return ChatResponse(
        response=cleaned,
        model=MODEL_NAME,
        tokens_generated=token_count,
    )

@app.post("/v1/chat", response_model=ChatResponse)
async def xinmate_chat(req: XinMateChatRequest, _: bool = Depends(verify_api_key)):
    """
    XinMate-compatible chat endpoint with history
    
    Request body:
    - bot_id: Bot identifier
    - personality_prompt: System/character prompt
    - user_message: Current user message
    - chat_history: List of {role, content} dicts
    - max_tokens, temperature, top_p: Generation params (optional)
    """
    if not hasattr(app.state, 'model') or not app.state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Build prompt with history
    prompt_parts = [req.personality_prompt, ""]
    
    # Include limited history
    history = req.chat_history[-MAX_HISTORY:] if len(req.chat_history) > MAX_HISTORY else req.chat_history
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Response: {content}")
    
    # Add current message
    prompt_parts.append(f"User: {req.user_message}")
    prompt_parts.append("Response:")
    
    full_prompt = "\n".join(prompt_parts)
    
    # Use defaults from env if not provided
    max_tokens = req.max_tokens or DEFAULT_MAX_TOKENS
    temperature = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
    top_p = req.top_p if req.top_p is not None else DEFAULT_TOP_P
    
    raw_response, token_count = generate_response(full_prompt, max_tokens, temperature, top_p)
    cleaned = clean_response(raw_response)
    
    return ChatResponse(
        response=cleaned,
        model=MODEL_NAME,
        tokens_generated=token_count,
    )

@app.post("/generate")
async def generate(req: SimpleChatRequest, _: bool = Depends(verify_api_key)):
    """Legacy generate endpoint - alias for /chat"""
    return await chat(req, _)

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
