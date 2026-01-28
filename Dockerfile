FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install build dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy application files
COPY src/ /app/src/
COPY pyproject.toml README.md ./

# Install dependencies
RUN uv pip install --system -e .

# Environment variables with defaults
ENV YUGABYTEDB_URL="dbname=yugabyte host=host.docker.internal port=5433 user=yugabyte password=yugabyte load_balance=false"

# Expose HTTP/SSE port
EXPOSE 8000

# Run the server using uv
ENTRYPOINT ["uv","--verbose", "run", "src/server.py"]
