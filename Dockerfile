FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DOCQA_INDEX_DIR=/data/index

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home app && mkdir -p /data && chown app /data

USER app
EXPOSE 8000
CMD ["docqa", "serve", "--host", "0.0.0.0", "--port", "8000"]
