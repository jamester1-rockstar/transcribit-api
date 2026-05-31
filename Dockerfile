# Transcribit API — explicit build (Railway uses this Dockerfile verbatim)
FROM python:3.11-slim

# System deps: ffmpeg for Whisper/mp3, build tools for any source wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install setuptools/wheel FIRST so pkg_resources exists for whisper's build step
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel

# Install Python deps (CPU-only torch to keep the image as small as possible)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Railway provides $PORT at runtime
ENV PORT=8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 620"]
