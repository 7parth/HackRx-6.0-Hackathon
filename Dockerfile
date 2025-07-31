FROM python:3.11-slim-bookworm

# Install system dependencies required for PyMuPDF (fitz)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies, including psycopg2-binary, PyMuPDF, and python-multipart
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir psycopg2-binary PyMuPDF python-multipart && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Define an upload directory environment variable matching FastAPI code
ENV UPLOAD_DIR=/app/uploaded_documents

# Create the upload directory in the container
RUN mkdir -p "$UPLOAD_DIR"

EXPOSE 8000

# Command to run the application (correct module path casing)
CMD ["uvicorn", "App.main:app", "--host", "0.0.0.0", "--port", "8000"]
