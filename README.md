# Multimodal Embedding Service

This is a vibecodded application to serve as an HTTP API for CLIP models like from Jina or OpenAI. It uses the `sentence-transformers` library with the `clip-ViT-B-32` model to generate high-quality, multimodal embeddings for both text and images.

The service is fully containerized using Docker and Docker Compose, and it utilizes a persistent volume to cache the large model file, ensuring fast startup times after the initial download.

## 🚀 Getting Started

### Prerequisites

*   Docker
*   Docker Compose


### Environment Variables

The service uses the following environment variables:

*   `MODEL_NAME`: The name of the model to use. Defaults to `clip-ViT-B-32`.
*   `TRANSFORMERS_CACHE`: The path to the persistent volume where the model cache will be stored.

### Running the Service

1.  **Build and Run:** Execute the following command in the root directory:

    ```bash
    docker compose up --build -d
    ```

    The first run will download the `clip-ViT-B-32` model by default (approx. 600MB) and store it in the persistent `model_cache` volume.

2.  **Access the API:** The service will be available at `http://localhost:8000`. You can view the interactive documentation (Swagger UI) at `http://localhost:8000/docs`.

## 💡 API Endpoints

The service exposes three primary endpoints for generating embeddings.

### 1. Embed Text

*   **Endpoint:** `POST /embed/text`
*   **Description:** Generates embeddings for a list of text strings.
*   **Request Body (`application/json`):**
    ```json
    {
      "texts": [
        "A photo of a cat sitting on a couch.",
        "The quick brown fox jumps over the lazy dog."
      ]
    }
    ```
*   **Response Body (`application/json`):**
    ```json
    {
      "embeddings": [
        [0.123, 0.456, ...],
        [0.789, 0.101, ...]
      ],
      "model": "clip-ViT-B-32"
    }
    ```

### 2. Embed Image

*   **Endpoint:** `POST /embed/image`
*   **Description:** Downloads images from provided URLs and generates embeddings.
*   **Request Body (`application/json`):**
    ```json
    {
      "image_urls": [
        "https://example.com/image1.jpg",
        "https://example.com/image2.png"
      ]
    }
    ```
*   **Response Body (`application/json`):**
    ```json
    {
      "embeddings": [
        [0.123, 0.456, ...],
        [0.789, 0.101, ...]
      ],
      "model": "clip-ViT-B-32"
    }
    ```

### 3. OpenAI-Compatible Embeddings

*   **Endpoint:** `POST /v1/embeddings`
*   **Description:** Generates embeddings for images using an OpenAI-compatible API interface.
*   **Request Body (`application/json`):**
    ```json
    {
      "model": "clip-ViT-B-32",
      "input": [
        "https://example.com/image1.jpg",
        "https://example.com/image2.png"
      ]
    }
    ```
*   **Response Body (`application/json`):**
    ```json
    {
      "object": "list",
      "data": [
        {
          "object": "embedding",
          "embedding": [0.123, 0.456, ...],
          "index": 0
        },
        {
          "object": "embedding",
          "embedding": [0.789, 0.101, ...],
          "index": 1
        }
      ],
      "model": "clip-ViT-B-32",
      "usage": {
        "prompt_tokens": 0,
        "total_tokens": 0
      }
    }
    ```

### Health Check

*   **Endpoint:** `GET /health`
*   **Description:** Checks if the service is running and the model is loaded.

## 🤝 Contribution

We welcome contributions! Please follow these basic rules:

1.  **Fork** the repository and create your feature branch (`git checkout -b feature/AmazingFeature`).
2.  Ensure your code adheres to the existing style and conventions (Python, FastAPI).
3.  Write clear, concise commit messages.
4.  Open a **Pull Request** describing your changes.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.