FROM python:3.12-slim

# ffmpeg is needed for audio extraction (ffprobe + ffmpeg)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .

# Render sets $PORT; default to 8000 for local runs
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
