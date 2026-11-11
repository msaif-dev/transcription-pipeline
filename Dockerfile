FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    STT_MODEL_DIR=/models STT_UPLOAD_DIR=/tmp/uploads

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image so containers start fast and need no internet at runtime.
ARG MODEL_SIZE=base
ENV STT_MODEL_SIZE=${MODEL_SIZE}
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='int8', download_root='/models')"

COPY app ./app

# Run as a non-root user that owns only what it needs to write.
RUN useradd --create-home appuser \
    && mkdir -p /tmp/uploads \
    && chown -R appuser /tmp/uploads /models
USER appuser

# The model is already in the image: never reach out to Hugging Face at runtime.
ENV HF_HUB_OFFLINE=1 STT_PRELOAD_MODEL=true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
