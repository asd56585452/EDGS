ARG BASE_IMAGE=asdggg/gscodec-studio:latest
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install dependencies and ensure numpy is pinned to <2.0 (1.26.4)
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    pip install "numpy==1.26.4" && \
    rm -rf /tmp/requirements.txt

# Copy submodules needed for CUDA extension compilation
COPY submodules/gaussian-splatting/submodules/diff-gaussian-rasterization /tmp/diff-gaussian-rasterization
COPY submodules/gaussian-splatting/submodules/simple-knn /tmp/simple-knn
COPY submodules/RoMa /tmp/RoMa

# Compile and install CUDA extensions and RoMa into system Python site-packages
RUN if ! grep -q "<cstdint>" /tmp/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h; then \
        sed -i '1i #include <cstdint>' /tmp/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h; \
    fi && \
    pip install --no-build-isolation /tmp/diff-gaussian-rasterization && \
    pip install --no-build-isolation /tmp/simple-knn && \
    pip install --no-deps /tmp/RoMa && \
    rm -rf /tmp/diff-gaussian-rasterization /tmp/simple-knn /tmp/RoMa

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV TORCH_HOME=/workspace/cache/torch
