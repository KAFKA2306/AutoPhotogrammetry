FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCH_LIST=12.0
ARG NERFSTUDIO_REVISION=50e0e3c70c775e89333256213363badbf074f29d
ARG GSPLAT_REVISION=v1.4.0
ARG OPEN3D_VERSION=0.19.0
ARG UV_VERSION=0.12.8
ARG AUTOPHOTOGRAMMETRY_REVISION
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    MAX_JOBS=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AUTOPHOTOGRAMMETRY_SOURCE_REVISION=${AUTOPHOTOGRAMMETRY_REVISION}

LABEL org.opencontainers.image.source="https://github.com/KAFKA2306/AutoPhotogrammetry" \
      org.opencontainers.image.description="Pinned AutoPhotogrammetry CUDA environment for Splatfacto quality experiments" \
      org.opencontainers.image.revision="${AUTOPHOTOGRAMMETRY_REVISION}"

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

RUN git clone --depth 1 --branch "${GSPLAT_REVISION}" --recurse-submodules --shallow-submodules \
        https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat \
    && python -m pip install --no-cache-dir --no-build-isolation /tmp/gsplat \
    && rm -rf /tmp/gsplat

RUN git init /opt/nerfstudio \
    && git -C /opt/nerfstudio remote add origin https://github.com/nerfstudio-project/nerfstudio.git \
    && git -C /opt/nerfstudio fetch --depth 1 origin "${NERFSTUDIO_REVISION}" \
    && git -C /opt/nerfstudio checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/nerfstudio rev-parse HEAD)" = "${NERFSTUDIO_REVISION}" \
    && python -m pip install --no-cache-dir /opt/nerfstudio

RUN python -m pip install --no-cache-dir "open3d==${OPEN3D_VERSION}"

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"
COPY pyproject.toml uv.lock /tmp/autophotogrammetry/
RUN cd /tmp/autophotogrammetry \
    && uv export --frozen --no-dev --format requirements-txt --output-file /tmp/requirements.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check

COPY . /opt/autophotogrammetry
WORKDIR /opt/autophotogrammetry
