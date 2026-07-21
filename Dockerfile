FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install the package from /src so the runtime image keeps only the data files.
# Having app/ on disk twice would make imports ambiguous.
COPY pyproject.toml README.md /src/
COPY app /src/app
RUN pip install /src && rm -rf /src

WORKDIR /app
COPY migrations ./migrations
COPY eval ./eval
ENV MIGRATIONS_DIR=/app/migrations

RUN useradd --create-home --uid 10001 rag && chown -R rag /app
USER rag

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
