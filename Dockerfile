# Stage 1: Build
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y \
    cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy local llama.cpp submodule instead of cloning from GitHub
COPY ./llama.cpp /app

WORKDIR /app

# Build with modern GGML_CUDA flags
# Quadro P5000 is Pascal architecture (6.1)
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O -DNDEBUG" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DCMAKE_CUDA_ARCHITECTURES="61;75;86;89;90" \
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
VOLUME /models

# Entrypoint
ENTRYPOINT []
