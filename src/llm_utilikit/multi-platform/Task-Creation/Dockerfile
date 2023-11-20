ARG IMAGE_NAME
FROM ${IMAGE_NAME}:12.3.0-base-ubuntu22.04 as base

ENV NV_CUDA_LIB_VERSION 12.3.0-1

FROM base as base-amd64

ENV NV_NVTX_VERSION 12.3.52-1
ENV NV_LIBNPP_VERSION 12.2.2.32-1
ENV NV_LIBNPP_PACKAGE libnpp-12-3=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 12.1.3.153-1

ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-12-3
ENV NV_LIBCUBLAS_VERSION 12.3.2.9-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

ENV NV_LIBNCCL_PACKAGE_NAME libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION 2.19.3-1
ENV NCCL_VERSION 2.19.3-1
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.3

FROM base as base-arm64

ENV NV_NVTX_VERSION 12.3.52-1
ENV NV_LIBNPP_VERSION 12.2.2.32-1
ENV NV_LIBNPP_PACKAGE libnpp-12-3=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 12.1.3.153-1

ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-12-3
ENV NV_LIBCUBLAS_VERSION 12.3.2.9-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

ENV NV_LIBNCCL_PACKAGE_NAME libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION 2.19.3-1
ENV NCCL_VERSION 2.19.3-1
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.3

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# Install Python, pip, and Python dependencies (transformers and langchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && python3 -m pip install --upgrade pip \
    && pip install transformers langchain \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python script and demo.wav into the container
COPY transcribe_text.py /app/transcribe_text.py
COPY demo.wav /app/demo.wav

# Set working directory
WORKDIR /app

# Continue with your existing Dockerfile setup...
RUN cuda-libraries-12-3=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    cuda-nvtx-12-3=${NV_NVTX_VERSION} \
    libcusparse-12-3=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

# Add entrypoint items
COPY entrypoint.d/ /opt/nvidia/entrypoint.d/
COPY nvidia_entrypoint.sh /opt/nvidia/
ENV NVIDIA_PRODUCT_NAME="CUDA"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

# The container will wait for user input after setup
CMD ["/bin/bash"]
