FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Update package lists and install essential utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    locales \
    software-properties-common \
    build-essential \
    ca-certificates \
    wget \
    curl \
    sudo \
    git \
    vim \
    emacs \
    htop \
    rsync \
    less \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Install Python and related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python-is-python3 \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MPI and related libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmpich-dev \
    libopenmpi-dev \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages and JupyterLab
RUN pip install --upgrade pip setuptools \
    && pip install torch torchvision torchaudio #--index-url https://download.pytorch.org/whl/cu12.4/

RUN pip install jupyterlab tensorflow tf-keras datasets 

RUN pip install bitsandbytes==0.44.1
 
RUN pip install flash_attn tim timm bitsandbytes deepspeed mpi4py \
    && jupyter-lab --generate-config

RUN git clone https://github.com/EpsilonLabsInc/InternVL-Epsi.git \
    && cd InternVL-Epsi \
    && pip install -r requirements/internvl_chat.txt

# Configure SSH and root login
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/#PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && mkdir /root/.ssh \
    && mkdir /var/run/sshd

# Clean up unnecessary files
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
