from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + decent quality

class TextInput(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(input: TextInput):
    embedding = model.encode(input.text).tolist()
    return {"embedding": embedding}
