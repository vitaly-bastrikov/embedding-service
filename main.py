import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- ENV CONFIG ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2").strip()
logger.info(f"Using embedding model: {EMBEDDING_MODEL}")

# --- MODEL LOADING ---
try:
    if EMBEDDING_MODEL == "all-MiniLM-L6-v2":
        model = SentenceTransformer(EMBEDDING_MODEL)
    elif EMBEDDING_MODEL == "intfloat/e5-base-v2":
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported EMBEDDING_MODEL: {EMBEDDING_MODEL}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# --- ENCODING METHODS ---

def encode_with_minilm(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()

def encode_with_e5(text: str) -> list[float]:
    text = f"query: {text}"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs)
        cls_embedding = output.last_hidden_state[:, 0]
        return F.normalize(cls_embedding, p=2, dim=1)[0].tolist()

# --- SELECT ACTIVE ENCODER FUNCTION ---
if EMBEDDING_MODEL == "all-MiniLM-L6-v2":
    encode = encode_with_minilm
else:
    encode = encode_with_e5

# --- API ---

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(request: TextRequest):
    try:
        embedding = encode(request.text)
        return {"embedding": embedding}
    except Exception as e:
        logger.error(f"Error processing embedding request: {str(e)}")
        raise
