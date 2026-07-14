##################################
# Builder
##################################
FROM python:3.13-slim AS builder
ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VER=2.9.1+cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=off \
    PIP_DEFAULT_TIMEOUT=100 \
    PATH="/opt/venv/bin:${PATH}"

# Build deps with cached apt
RUN --mount=type=cache,target=/var/cache/apt,id=apt-archives-builder,sharing=locked \
    set -eux; \
    rm -f /var/cache/apt/archives/lock /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock || true; \
    apt-get update; \
    apt-get install -y --no-install-recommends git gcc g++ python3-dev build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/partial/*

RUN python -m venv /opt/venv
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache,sharing=locked \
    pip install -U pip wheel build

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache,sharing=locked \
    pip install -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /src
COPY . /src
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache,sharing=locked \
    pip install --no-deps --no-build-isolation .

# NLTK data with cache to avoid re-downloads; copy cached files into layer
ENV NLTK_DATA=/nltk_data
RUN --mount=type=cache,target=/nltk_cache,id=nltk-cache,sharing=locked bash -eux <<'SH'
mkdir -p /nltk_cache /nltk_data
python - <<'PY'
import os, nltk
cache_dir = "/nltk_cache"
for p in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']:
    nltk.download(p, download_dir=cache_dir)
PY
cp -a /nltk_cache/. /nltk_data/
SH

##################################
# Runtime
##################################
FROM python:3.13-slim AS runtime
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=off \
    PIP_ROOT_USER_ACTION=ignore \
    NLTK_DATA=/usr/local/share/nltk_data \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/code/src

RUN --mount=type=cache,target=/var/cache/apt,id=apt-archives-runtime,sharing=locked \
    set -eux; \
    rm -f /var/cache/apt/archives/lock /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock || true; \
    apt-get update; \
    apt-get install -y --no-install-recommends poppler-utils tesseract-ocr libgl1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/partial/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /nltk_data/ /usr/local/share/nltk_data/

WORKDIR /code

# Reuse builder copy of the source (already filtered by .dockerignore)
COPY --from=builder /src /code
COPY --from=builder --chmod=0755 /src/entrypoint.sh /code/entrypoint.sh

WORKDIR /code/src/paper_data_linking/web_app/
ENTRYPOINT ["/code/entrypoint.sh"]
