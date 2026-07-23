FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    APP_NAME=knowledge-base-chat \
    DEBUG=False \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LLM_PROVIDER=ollama \
    OLLAMA_MODEL=tinyllama \
    OLLAMA_BASE_URL=http://host.docker.internal:11434/api/generate \
    VECTOR_STORE_PATH=/app/data/vector_store \
    AUDIT_DB_PATH=/app/data/audit/audit.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .

RUN pip install --upgrade pip

RUN pip install \
    torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements-prod.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/raw /app/data/vector_store /app/data/audit /app/.cache/huggingface \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
