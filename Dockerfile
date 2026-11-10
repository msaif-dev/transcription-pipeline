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
RUN useradd --create-home appuser && mkdir -p /tmp/uploads && chown appuser /tmp/uploads
USER appuser

EXPOSE 8000
ENV STT_PRELOAD_MODEL=true
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
