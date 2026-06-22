# Stage 1: Build
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y \
    git cmake build-essential libcurl4-openssl-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use 'master' for the latest Qwen3.5 support
ARG LLAMA_CPP_VERSION=master
RUN git clone https://github.com/ggml-org/llama.cpp.git . && \
    git checkout ${LLAMA_CPP_VERSION}

# Build with modern GGML_CUDA flags
# Quadro P5000 is Pascal architecture (6.1)
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O1 -DNDEBUG" \
    -DGGML_CUDA=ON \
    -DGGML_CURL=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --config Release -j $(nproc)

# Stage 2: Runtime
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# Install runtime dependencies for curl/healthchecks
RUN apt-get update && \
    apt-get install -y libcurl4 libssl3 libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure 'ubuntu' user exists and has correct permissions for the volume
RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -u 1000 ubuntu
RUN mkdir -p /models && chown -R ubuntu:ubuntu /models /app

# 1. Copy the entire bin directory from the builder
COPY --from=builder /app/build/bin /app/bin

# 2. Add the bin directory to the PATH and the Library Path
ENV PATH="/app/bin:${PATH}"
ENV LD_LIBRARY_PATH="/app/bin:${LD_LIBRARY_PATH}"

USER ubuntu
EXPOSE 8080
VOLUME /models

# Entrypoint
ENTRYPOINT []
