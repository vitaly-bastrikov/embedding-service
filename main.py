from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(request: TextRequest):
    embedding = model.encode(request.text).tolist()
    return {"embedding": embedding}
