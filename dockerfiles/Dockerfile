FROM python:3.13-slim-trixie

# Use prebuilt uv binaries (distroless)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install build dependencies
RUN apt update && apt install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock README.md /app/
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

# Create non-root user with home and cache directories
RUN addgroup --system app \
    && adduser --system --ingroup app --home /home/app app \
    && mkdir -p /home/app/.cache/uv /models \
    && chown -R app:app /home/app /app /models

ENV HOME=/home/app
ENV UV_CACHE_DIR=$HOME/.cache/uv

USER app

# Optimized uv sync: non-editable install
RUN sh -c 'umask 000 && ulimit -n 4096 && uv sync --no-editable --frozen --no-cache'

# Set runtime environment
WORKDIR /app
ENV CLASSPOSE_MODEL_DIR=/models

RUN chmod -R 777 $HOME
RUN chmod -R 777 /app/uv.lock
RUN chmod -R 777 /app/src /app/pyproject.toml

ENTRYPOINT ["uv", "run"]
