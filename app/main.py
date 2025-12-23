import os
from fastapi import FastAPI, HTTPException
from app.embedding_service import (
    embed_text,
    embed_image,
    TextEmbedRequest,
    ImageEmbedRequest,
    EmbeddingResponse,
    default_model,
    MODEL_NAME, OpenAIEmbeddingResponse, OpenAIEmbeddingRequest, open_ai_embed_image
)

app = FastAPI(
    title="Multimodal Embedding Service",
    description=f"HTTP service for generating text and image embeddings using {MODEL_NAME}.",
    version="1.0.0"
)

# --- Endpoints ---

@app.get("/health")
async def health_check():
    if default_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok", "default_model": MODEL_NAME}

@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text_endpoint(request: TextEmbedRequest):
    """Generates embeddings for a list of text strings."""
    try:
        return embed_text(request.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image_endpoint(request: ImageEmbedRequest):
    """Generates embeddings for a list of image URLs."""
    try:
        return embed_image(request.image_urls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def openai_embedding_endpoint(request: OpenAIEmbeddingRequest):
    """Generates embeddings for a list of image URLs."""
    try:
        return open_ai_embed_image(image_urls=request.input, model_name=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))