FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev
COPY app ./app
COPY download-model.sh ./
COPY keywords.txt ./
RUN chmod +x /app/download-model.sh

EXPOSE 8003
CMD ["sh", "-c", "/app/download-model.sh && exec uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8003"]
