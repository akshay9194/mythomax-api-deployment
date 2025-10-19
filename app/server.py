import os
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from pathlib import Path

app = FastAPI(title="MythoMax API")

# -------------------------------
# Configuration
# -------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "TheBloke/MythoMax-L2-13B-GPTQ")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
API_KEY = os.environ.get("API_KEY", "my_super_secret_key")  # header-based auth

# Persistent volume mount
MODEL_VOLUME = Path("/runpod-volume/models/mythomax")
MODEL_VOLUME.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Request Model
# -------------------------------
class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

# -------------------------------
# Security Middleware
# -------------------------------
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return await call_next(request)

# -------------------------------
# Startup Event: Load Model
# -------------------------------
@app.on_event("startup")
def load_model():
    global pipe
    print("Starting model load:", MODEL_NAME)

    # BitsAndBytesConfig for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # HF token if required
    hf_kwargs = {}
    if HF_TOKEN:
        hf_kwargs["use_auth_token"] = HF_TOKEN

    # Check if model already exists in mounted volume
    model_path = MODEL_VOLUME if any(MODEL_VOLUME.iterdir()) else MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_path, **hf_kwargs, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        **hf_kwargs
    )

    # Pipeline on GPU 0
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0
    )
    app.state.pipe = pipe
    print("Model loaded successfully.")

# -------------------------------
# Generation Endpoint
# -------------------------------
@app.post("/generate")
async def generate(req: GenRequest):
    pipe = app.state.pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    out = pipe(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=True,
        return_full_text=False,
        truncation=True
    )

    text = out[0]["generated_text"] if isinstance(out, list) else str(out)
    return {"response": text}
