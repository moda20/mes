import os

import torch
from sentence_transformers import SentenceTransformer
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "clip-ViT-B-32")
MODEL_CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", "/app/model_cache")
def loadModel(modelName: str):
    # --- Initialization ---
    final_model_name = modelName or DEFAULT_MODEL_NAME

    # Initialize the model globally to load it once on startup
    try:
        # 1. Check if CUDA (GPU) is available
        if torch.cuda.is_available():
            device = 'cuda'
            print("GPU is available. Using GPU.")
        else:
            device = 'cpu'
            print("GPU not available. Using CPU.")
        # The model will be downloaded to MODEL_CACHE_DIR if not present
        model = SentenceTransformer(final_model_name, cache_folder=MODEL_CACHE_DIR, device=device)
        print(f"Successfully loaded model: {final_model_name} from {MODEL_CACHE_DIR}")
        return model
    except Exception as e:
        print(f"Error loading model {final_model_name}: {e}")
        # In a real service, you might want to exit or raise an error here
        model = None
