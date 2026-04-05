"""
RunPod Serverless Handler for MythoMax (GGUF)

Generic text generation handler for RunPod Serverless.
Downloads pre-quantized GGUF model from Cloudflare R2 on cold start,
caches in /tmp, loads with llama-cpp-python for GPU inference.

Uses shared model_loader.py for model download + loading logic.

Expected input:
  {
    "input": {
      "prompt": "your full prompt here",
      "max_tokens": 180,
      "temperature": 0.9,
      "top_p": 0.92
    }
  }

Returns:
  {
    "response": "generated text",
    "model": "MythoMax-L2-13B-Q4_K_M",
    "tokens_generated": 42
  }
"""

import os
import re
import logging
import runpod

# Serverless uses /tmp (no network volume)
os.environ.setdefault("MODEL_DIR", "/tmp/models")

from model_loader import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.85"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# ── Global model ref ─────────────────────────────────────────
llm = None


def clean_response(text: str) -> str:
    """Clean model output — remove artifacts, name prefixes, self-dialogue."""
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


def handler(event):
    """
    RunPod Serverless handler.

    event["input"] = {
        "prompt": str,
        "max_tokens": int (optional),
        "temperature": float (optional),
        "top_p": float (optional),
    }
    """
    inp = event.get("input", {})
    prompt = inp.get("prompt", "")

    if not prompt:
        return {"error": "No prompt provided"}

    max_tokens = inp.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = inp.get("temperature", DEFAULT_TEMPERATURE)
    top_p = inp.get("top_p", DEFAULT_TOP_P)

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=REPETITION_PENALTY,
        stop=["### Input:", "### Instruction:", "### Response:", "User:", "Human:"],
    )

    raw = output["choices"][0]["text"]
    token_count = output["usage"]["completion_tokens"]
    cleaned = clean_response(raw)

    return {
        "response": cleaned,
        "model": MODEL_NAME,
        "tokens_generated": token_count,
    }


# ── Load model at cold start, then serve ─────────────────────
# Try loading at startup (works if DNS is ready)
# If it fails (e.g., DNS not ready in serverless), load on first request
try:
    llm = load_model()
except Exception as e:
    logger.warning(f"Startup model load failed ({e}), will retry on first request")
    llm = None


def handler_wrapper(event):
    """Wrapper that ensures model is loaded before handling."""
    global llm
    if llm is None:
        logger.info("Loading model on first request...")
        llm = load_model()
    return handler(event)


runpod.serverless.start({"handler": handler_wrapper})
