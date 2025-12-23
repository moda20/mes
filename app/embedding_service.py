import os
import io
from typing import Literal, Optional, Union

import requests
from PIL import Image
from pydantic import BaseModel, Field
from app.model_service import loadModel

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME", "clip-ViT-B-32")
# TRANSFORMERS_CACHE is set via environment variable in compose.yml
# and defaults to /app/model_cache in the Dockerfile
MODEL_CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", "/app/model_cache")

# --- Initialization ---
default_model = loadModel(MODEL_NAME)

# --- Pydantic Schemas ---
class TextEmbedRequest(BaseModel):
    texts: list[str]

class ImageEmbedRequest(BaseModel):
    image_urls: list[str]

class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str


# --- OpenAI Embedding Response Schema ---
class OpenAIEmbeddingRequest(BaseModel):
    model: Optional[str] = Field(MODEL_NAME, example="text-embedding-ada-002")
    input: Union[str, list[str]]
    encoding_format: Optional[Literal["float"]] = "float"


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingItem(BaseModel):
    object: str = Field(..., example="embedding")
    embedding: list[float]
    index: int


class OpenAIEmbeddingResponse(BaseModel):
    object: str = Field(..., example="list")
    data: list[EmbeddingItem]
    model: str
    usage: Usage


# --- Utility Functions ---
def get_image_from_url(url: str) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image object."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image from {url}: {e}")
    except Exception as e:
        raise Exception(f"Failed to process image from {url}: {e}")

def embed_text(texts: list[str]) -> EmbeddingResponse:
    """Generates embeddings for a list of text strings."""
    if default_model is None:
        raise Exception("Model not loaded.")
    
    if not texts:
        return EmbeddingResponse(embeddings=[], model=MODEL_NAME)
    
    # Encode the texts
    embeddings = default_model.encode(texts, convert_to_numpy=True)
    
    # Convert numpy array to list of lists for JSON serialization
    embeddings_list = embeddings.tolist()
    
    return EmbeddingResponse(embeddings=embeddings_list, model=MODEL_NAME)

def embed_image(image_urls: list[str]) -> EmbeddingResponse:
    """Generates embeddings for a list of image URLs."""
    if default_model is None:
        raise Exception("Model not loaded.")
    
    if not image_urls:
        return EmbeddingResponse(embeddings=[], model=MODEL_NAME)
    
    images = []
    for url in image_urls:
        # Download and process image
        image = get_image_from_url(url)
        images.append(image)
    
    # Encode the images
    # The model.encode method handles both text and image inputs for multimodal models
    embeddings = default_model.encode(images, convert_to_numpy=True)
    
    # Convert numpy array to list of lists for JSON serialization
    embeddings_list = embeddings.tolist()
    
    return EmbeddingResponse(embeddings=embeddings_list, model=MODEL_NAME)



def open_ai_embed_image(image_urls: list[str], model_name: str) -> OpenAIEmbeddingResponse:
    """Generates embeddings for a list of image URLs."""
    if default_model is None and model_name is None:
        raise Exception("Model not loaded, and no model is provided.")

    if not image_urls:
        return OpenAIEmbeddingResponse(object="list", data=[], model=model_name)

    images = []
    for url in image_urls:
        # Download and process image
        image = get_image_from_url(url)
        images.append(image)

    target_model = default_model
    if model_name is not MODEL_NAME:
        target_model = loadModel(model_name)

    # Encode the images
    # The model.encode method handles both text and image inputs for multimodal models
    embeddings = target_model.encode(images, convert_to_numpy=True)

    # Convert numpy array to list of lists for JSON serialization
    embeddings_list = [
        EmbeddingItem(
            object="embedding",
            embedding=vector,
            index=i,
        )
        for i, vector in enumerate(embeddings.tolist())
    ]

    return OpenAIEmbeddingResponse(object="list", data=embeddings_list, model=model_name, usage=Usage(prompt_tokens=0, total_tokens=0))