
FROM python:3.10-slim AS builder

ARG TORCH_VERSION
ARG CUDA_VERSION
# Install dependencies
COPY app/requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

RUN if [ -n "$CUDA_VERSION" ]; then \
      pip install torch --no-build-isolation --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"; \
      pip install flit-core; \
    else \
      pip install torch; \
    fi

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV MODEL_NAME clip-ViT-B-32
ENV MODEL_CACHE_DIR /app/model_cache

# Create and set working directory
WORKDIR /app
# Copy application code
COPY app ./app

COPY --from=builder /install /usr/local
# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]