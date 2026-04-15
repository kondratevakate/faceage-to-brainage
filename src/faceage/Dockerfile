FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL description="Docker container"
ENV DEVICE="cuda"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    wget \
    python3-pip \
    python3-dev \
    python3-packaging

RUN apt-get install -y xvfb libgl1-mesa-glx libosmesa6

RUN pip install --upgrade pip
RUN pip install nibabel pyvista numpy

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
COPY ./faceage_mri/FaceAge/environment-gpu.yaml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

RUN conda init bash && \
    echo "conda activate faceage-gpu" >> ~/.bashrc

WORKDIR /workspace