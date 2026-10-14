# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : Dockerfile
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

# Connoisseur Companion: one image that runs either the web app or the MCP server.
#   docker build -t connoisseur-companion .
#   docker run -p 7860:7860 --env-file .env connoisseur-companion
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    UI_HOST=0.0.0.0

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheels.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Bake the datasets, embedding models and vector index into the image so containers start ready.
# Pass --build-arg WITH_IMAGES=false for a smaller image without the CLIP dish-photo index.
ARG WITH_IMAGES=true
RUN if [ "$WITH_IMAGES" = "true" ]; then \
        connoisseur download-data --with-images && rm -f data/raw/synthetic-recipe-images.zip && connoisseur build-index; \
    else \
        connoisseur download-data && connoisseur build-index --no-images; \
    fi

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 7860 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:' + os.environ.get('UI_PORT','7860'))" || exit 1

CMD ["connoisseur", "ui"]
