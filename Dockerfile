# SceneFinder — AI YouTube Episode & Scene Finder
# Hugging Face Spaces (Docker SDK) deployment
FROM python:3.11-slim

# ffmpeg for audio extraction, tesseract for screenshot OCR (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/dist/ ./frontend/dist/

# HF Spaces routes traffic to port 7860
ENV PORT=7860 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# Pre-download AI models at build time so first run is fast
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" \
 && python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup']))"

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
