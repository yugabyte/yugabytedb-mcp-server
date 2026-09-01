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
#
# the server refuses to start in HTTP mode when
# MCP_HOST is not loopback AND no auth provider is configured. When
# running the image as a network-reachable server, set BOTH:
#
#   -e MCP_HOST=0.0.0.0
#   -e MCP_AUTH_PROVIDER=cognito  (plus COGNITO_* env)
#
# For dev-only unauthenticated use, add `-e MCP_ALLOW_UNAUTHENTICATED=true`
# — the server will start with a prominent WARNING but no auth.
ENTRYPOINT ["yugabytedb-mcp"]
CMD ["--transport", "http"]
