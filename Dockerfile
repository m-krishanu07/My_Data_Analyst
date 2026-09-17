# Matches the interpreter the test suite and CI run on. A different minor
# version would mean the image is not the thing that was verified.
FROM python:3.13-slim

# ── Runtime environment ───────────────────────────────────────
# MPLBACKEND: the sandbox worker renders charts with no display attached.
# PYTHONDONTWRITEBYTECODE: the app directory is read-only to the app user.
# PYTHONUNBUFFERED: container logs should appear as they happen, not on exit.
# SANDBOX_MEMORY_MB: Linux is the only platform where RLIMIT_AS is actually
#   enforced, and it caps *virtual* address space — pandas + sklearn +
#   matplotlib + plotly can reserve well past the 2048 MB default just by
#   importing. Raised so the worker does not die at startup on the one platform
#   where the limit is real. Lower it if you want a tighter cap and have
#   measured the floor.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HOME=/home/app \
    MPLCONFIGDIR=/home/app/.matplotlib \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    SANDBOX_MEMORY_MB=4096

WORKDIR /app

# curl is only needed by the healthcheck below. Everything else in the
# scientific stack ships manylinux wheels, so no compiler is required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, as their own layer: source edits then rebuild in seconds
# instead of re-resolving pandas, sklearn and matplotlib every time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run unprivileged. The sandbox worker executes LLM-generated code; if it ever
# escapes its restrictions, it lands as a user that owns nothing.
# Both directories must exist and be writable *before* the USER switch:
# matplotlib's font cache and Streamlit's config both write to $HOME on first
# run, and failing that is a startup crash, not a warning.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p "$MPLCONFIGDIR" "$HOME/.streamlit" \
    && chown -R app:app /app /home/app
USER app

EXPOSE 8501

# Streamlit's own liveness endpoint, so an orchestrator restarts a hung app
# rather than keeping it in rotation. Generous start period: the first request
# imports the scientific stack.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
