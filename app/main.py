import os
import io
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME", "clip-ViT-B-32")
# TRANSFORMERS_CACHE is set via environment variable in compose.yml
# and defaults to /app/model_cache in the Dockerfile
MODEL_CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", "/app/model_cache")

# --- Initialization ---
# Initialize the model globally to load it once on startup
try:
    # The model will be downloaded to MODEL_CACHE_DIR if not present
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR)
    print(f"Successfully loaded model: {MODEL_NAME} from {MODEL_CACHE_DIR}")
except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    # In a real service, you might want to exit or raise an error here
    model = None

app = FastAPI(
    title="Multimodal Embedding Service",
    description=f"HTTP service for generating text and image embeddings using {MODEL_NAME}.",
    version="1.0.0"
)

# --- Pydantic Schemas ---
class TextEmbedRequest(BaseModel):
    texts: list[str]

class ImageEmbedRequest(BaseModel):
    image_urls: list[str]

class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str

# --- Utility Functions ---
def get_image_from_url(url: str) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image object."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from {url}: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image from {url}: {e}")

# --- Endpoints ---

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(request: TextEmbedRequest):
    """Generates embeddings for a list of text strings."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    if not request.texts:
        return EmbeddingResponse(embeddings=[], model=MODEL_NAME)

    # Encode the texts
    embeddings = model.encode(request.texts, convert_to_numpy=True)
    
    # Convert numpy array to list of lists for JSON serialization
    embeddings_list = embeddings.tolist()
    
    return EmbeddingResponse(embeddings=embeddings_list, model=MODEL_NAME)

@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(request: ImageEmbedRequest):
    """Generates embeddings for a list of image URLs."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    if not request.image_urls:
        return EmbeddingResponse(embeddings=[], model=MODEL_NAME)

    images = []
    for url in request.image_urls:
        # Download and process image
        image = get_image_from_url(url)
        images.append(image)

    # Encode the images
    # The model.encode method handles both text and image inputs for multimodal models
    embeddings = model.encode(images, convert_to_numpy=True)
    
    # Convert numpy array to list of lists for JSON serialization
    embeddings_list = embeddings.tolist()
    
    return EmbeddingResponse(embeddings=embeddings_list, model=MODEL_NAME)