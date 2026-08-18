FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.0 \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    MAX_JOBS=8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        colmap \
        ffmpeg \
        git \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128

RUN git clone --depth 1 --branch v1.4.0 --recurse-submodules --shallow-submodules \
        https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat \
    && python -m pip install --no-cache-dir --no-build-isolation /tmp/gsplat \
    && rm -rf /tmp/gsplat \
    && python -m pip install --no-cache-dir nerfstudio==1.1.5

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check

WORKDIR /workspace
