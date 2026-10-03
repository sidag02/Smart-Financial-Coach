# The web app image for the Oct 6, 2026 demo (Web App UI, "Demo build"). It serves the demo
# bundle read-only, so it needs no model download, no network at start-up and no writable disk.
#
#   uv run sfc-web build-demo --data data/synthetic/default.sqlite   # -> build/demo
#   az acr build --registry <registry> --image sfc-web:<tag> .
#
# Runtime settings: SFC_DEMO_PASSWORD, SFC_SESSION_SECRET and SFC_LLM_API_KEY (secrets).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Dependencies first, so code changes don't reinstall them. No dev or train groups (FR-3)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --locked --no-default-groups

COPY configs/web ./configs/web
COPY docs/design ./docs/design
COPY build/demo ./build/demo

RUN useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["/app/.venv/bin/sfc-web", "serve", "--host", "0.0.0.0", "--port", "8000", \
     "--forwarded-allow-ips", "*"]
