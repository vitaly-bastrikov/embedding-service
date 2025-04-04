import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- MODEL LOADING ---
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Successfully loaded all-MiniLM-L6-v2 model")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# --- API ---

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(request: TextRequest):
    try:
        embedding = model.encode(request.text, normalize_embeddings=True).tolist()
        return {"embedding": embedding}
    except Exception as e:
        logger.error(f"Error processing embedding request: {str(e)}")
        raise
