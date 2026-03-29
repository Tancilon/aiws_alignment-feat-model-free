FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy

ENV TZ=Etc/UTC \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    ALL_PROXY=${ALL_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy:-${HTTP_PROXY}} \
    https_proxy=${https_proxy:-${HTTPS_PROXY}} \
    all_proxy=${all_proxy:-${ALL_PROXY}} \
    no_proxy=${no_proxy:-${NO_PROXY}} \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    for attempt in 1 2 3 4 5; do \
      apt-get update -o Acquire::Retries=5 && break; \
      echo "apt-get update failed on attempt ${attempt}, retrying..."; \
      sleep 5; \
    done && \
    apt-get install -y -o Acquire::Retries=5 --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      cuda-toolkit-11-8 \
      ffmpeg \
      freeglut3-dev \
      git \
      libboost-program-options-dev \
      libboost-system-dev \
      libegl1 \
      libegl1-mesa-dev \
      libeigen3-dev \
      libgl1 \
      libgl1-mesa-dev \
      libglib2.0-0 \
      libgomp1 \
      libgles2-mesa-dev \
      libglvnd-dev \
      libsm6 \
      libspatialindex-dev \
      libx11-6 \
      libxext6 \
      libxrender1 \
      pkg-config \
      pybind11-dev \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L --retry 5 --retry-all-errors --retry-delay 5 \
      https://micro.mamba.pm/api/micromamba/linux-64/latest \
      -o /tmp/micromamba.tar.bz2 && \
    tar -xvjf /tmp/micromamba.tar.bz2 -C /usr/local/bin --strip-components=1 bin/micromamba && \
    rm -f /tmp/micromamba.tar.bz2

WORKDIR /opt/foundationpose

COPY docker/runtime-requirements.txt /tmp/runtime-requirements.txt

SHELL ["/bin/bash", "-lc"]

RUN micromamba create -y -n foundationpose python=3.9 pip && \
    micromamba clean --all --yes

ENV PATH=/opt/micromamba/envs/foundationpose/bin:/opt/micromamba/condabin:$PATH \
    LD_LIBRARY_PATH=/opt/micromamba/envs/foundationpose/lib:/opt/micromamba/envs/foundationpose/lib/python3.9/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
    PYTHONPATH=/opt/foundationpose:${PYTHONPATH:-} \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118

COPY . /opt/foundationpose

RUN python -m pip install --upgrade pip wheel setuptools==69.5.1

RUN python -m pip install \
      torch==2.0.0+cu118 \
      torchvision==0.15.1+cu118 \
      torchaudio==2.0.1+cu118 \
      --index-url https://download.pytorch.org/whl/cu118

RUN python -m pip install -r /tmp/runtime-requirements.txt

RUN python -m pip install \
      pytorch3d==0.7.3 \
      -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

RUN curl -L --retry 5 --retry-all-errors --retry-delay 5 \
      https://github.com/NVlabs/nvdiffrast/archive/refs/heads/main.zip \
      -o /tmp/nvdiffrast.zip && \
    python -c "import zipfile; zipfile.ZipFile('/tmp/nvdiffrast.zip').extractall('/tmp')" && \
    python -m pip install --no-build-isolation /tmp/nvdiffrast-main && \
    rm -rf /tmp/nvdiffrast.zip /tmp/nvdiffrast-main

RUN bash build_all_conda.sh

RUN python /opt/foundationpose/docker/import_check.py

CMD ["bash"]
