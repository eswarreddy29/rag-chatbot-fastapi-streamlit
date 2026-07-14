# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Grant permissions for ChromaDB to write to disk in the cloud
RUN mkdir -p /app/backend/chroma_db /app/backend/documents && chmod -R 777 /app/backend

# Hugging Face Spaces run on port 7860 by default
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]