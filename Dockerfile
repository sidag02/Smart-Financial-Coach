# The web app image for the Oct 6, 2026 demo (Web App UI, "Demo build"). It serves the demo
# bundle read-only, so it needs no model download and no network at start-up; the only file it
# writes are the category feedback and goal stores (below).
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
# The app runs as another user: the bundle must be readable by everyone (see sfc-web build-demo)
RUN chmod -R a+rX build/demo

RUN useradd --uid 10001 --no-create-home app
# Category feedback (FR-5, FR-6) and savings goals (FR-10) are the only things the app writes:
# SQLite files in the container's writable layer, kept until the container restarts or redeploys
# (owner, Oct 3, 2026). For durable stores, prefer Postgres (the v2 plan) to SQLite on an Azure
# Files (SMB) mount, whose file locking SQLite doesn't trust
RUN mkdir -p /var/lib/sfc && chown 10001 /var/lib/sfc
ENV SFC_FEEDBACK_DB=/var/lib/sfc/feedback.sqlite SFC_GOALS_DB=/var/lib/sfc/goals.sqlite
USER 10001
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["/app/.venv/bin/sfc-web", "serve", "--host", "0.0.0.0", "--port", "8000"]
