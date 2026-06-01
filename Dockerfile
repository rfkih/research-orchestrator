FROM python:3.12-slim

WORKDIR /app

# libgomp1: required by LightGBM (and XGBoost) training subprocess
# launched via /train/.venv when POST /ml/training-runs is called.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

RUN pip install --no-cache-dir .

EXPOSE 8082

# Liveness — no I/O, just checks the process is responding.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8082/healthz').read()" || exit 1

CMD ["python", "-m", "orchestrator"]
