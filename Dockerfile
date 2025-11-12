# Base Image
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-dev \
      git build-essential pkg-config \
      ffmpeg libgl1 libglib2.0-0 \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Python environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --upgrade pip wheel

# Setup pip mirror
RUN mkdir -p /etc && \
    printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntimeout = 120\ntrusted-host = pypi.tuna.tsinghua.edu.cn\n" > /etc/pip.conf

# Set up the application directory and copy source code into a subdirectory
# DataFlow Commit b27d6bc24cf86835fda7bc6fe1a289cb9eb63bd2
WORKDIR /app
COPY . ./DataFlow/

# Set the working directory to the project source
WORKDIR /app/DataFlow

# Install the project in editable mode with its dependencies
RUN pip install -e ".[vllm]"

# Set the container's default command to an interactive shell
CMD ["/bin/bash"]