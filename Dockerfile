FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution + virtualenv management.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

# Copy project metadata and source.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ /app/src/

# Install the package + dependencies into the system Python so the
# entry point is discoverable on PATH.
RUN uv pip install --system .

EXPOSE 8000

# Default to HTTP transport so the container is useful out of the box.
# Override via `docker run … yugabytedb-mcp <flags>`.
ENTRYPOINT ["yugabytedb-mcp"]
CMD ["--transport", "http"]
