FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV MODEL_NAME clip-ViT-B-32
ENV MODEL_CACHE_DIR /app/model_cache

# Create and set working directory
WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/main.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]