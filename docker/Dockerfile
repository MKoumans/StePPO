FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev \
        git curl build-essential \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

# uv installs the locked dependencies into /opt/venv; UV_PROJECT_ENVIRONMENT
# makes `uv run` / `uv sync` in /workspace reuse that environment.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_PYTHON=python3.11
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_LINK_MODE=copy
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

COPY pyproject.toml uv.lock /tmp/deps/
WORKDIR /tmp/deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# The released StePPO models (PAPER_MODELS in src/steppo/cli.py), so `steppo solve`
# loads them from the image instead of downloading them on first use.
ENV HF_HOME=/opt/huggingface
RUN python -c "from huggingface_hub import snapshot_download; [snapshot_download(repo) for repo in ('MKoumans/sd', 'MKoumans/vdp', 'MKoumans/bru')]"

RUN ln -sf /workspace/scripts/running.sh /usr/local/bin/running

ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID dev \
    && useradd -m -u $UID -g $GID -s /bin/bash dev \
    && mkdir -p /workspace \
    && chown $UID:$GID /workspace \
    && chown -R $UID:$GID $UV_PROJECT_ENVIRONMENT $HF_HOME

USER dev
WORKDIR /workspace
