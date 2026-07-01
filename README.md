# llama.cpp CUDA Docker

Multi-architecture CUDA build of [llama.cpp](https://github.com/ggerganov/llama.cpp) for running LLM inference on NVIDIA GPUs.

## GPU Coverage

The binary is compiled with fat binaries covering five CUDA architectures:

| Compute | Architecture | Example GPUs |
|---------|--------------|--------------|
| 6.1     | Pascal       | GTX 1080, GTX 1070, Quadro P5000 |
| 7.5     | Turing       | RTX 2080, RTX 2060, Quadro RTX 5000 |
| 8.6     | Ampere       | RTX 3090, RTX 3080, RTX 3060 |
| 8.9     | Ada Lovelace | RTX 4090, RTX 4080, RTX 4060 |
| 9.0     | Hopper       | H100, H200 |

This spans consumer GPUs from ~2016 to present. One container runs on any of them.

## Build Flags

- **`GGML_CUDA_FA_ALL_QUANTS=ON`** — precompiles CUDA kernels for all quantization types, avoiding just-in-time compilation overhead at runtime.
- **`GGML_CUDA_FORCE_MMQ=ON`** — forces matrix×quantized-matrix kernels (better performance on multi-GPU setups without NVLink, e.g. dual P5000).

## K/V Cache Quantization

You can independently set quantization for K and V cache via `models.ini` or CLI flags:

```ini
cache-type-k = q8_0
cache-type-v = q8_0
```

Common options: `f16` (default), `q8_0`, `q4_0`, `q4_1`. Using lower precision (e.g. `q8_0` or `q4_0`) reduces VRAM usage with minimal quality loss. K and V can be set to different types.

## Image Tags

Images are tagged with the corresponding llama.cpp release tag (e.g. `b9851`).
When the `llama.cpp` submodule is updated, bump the tag in `docker-compose.yml` to match.

```bash
cd llama.cpp && git describe --tags --abbrev=0
```

## Quick Start

```bash
# Pull the pre-built image
docker compose pull

# Run with model preset (see models.ini.sample)
docker compose up -d

# Or with custom flags
docker compose run --rm server \
  llama-server --host 0.0.0.0 --port 8000 \
    --model /models/my-model.gguf \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn --ctx-size 32768
```

## Configuration

Copy `models.ini.sample` to `models.ini` and edit it to manage model presets with per-model settings for GPU layers, tensor split, context size, KV cache quantization, speculative decoding, and sampling parameters.

```bash
cp models.ini.sample models.ini
```

## Tests

Tests in `tests/` run against a live server:

```bash
# Run all suites
python tests/run_all.py

# Run a specific suite
python tests/run_all.py --only test_reasoning

# Verbose output for failures
python tests/run_all.py --verbose
```

Unit tests for the grading infrastructure (no server needed):

```bash
python tests/test_graders.py
# or
python -m pytest tests/test_graders.py
```
