# llama.cpp CUDA Docker

[![CI](https://github.com/openmtx/docker-llamacpp-cuda/actions/workflows/docker-build.yml/badge.svg?branch=dev)](https://github.com/openmtx/docker-llamacpp-cuda/actions/workflows/docker-build.yml)
![ghcr](https://img.shields.io/badge/ghcr-openmtx%2Fllamacpp--cuda-blue)

Run [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` on an NVIDIA GPU
using a prebuilt, multi-architecture CUDA image. The default branch (`master`) is the
**run** branch — it has everything you need to serve a model; you only need to add a model
file and a small config.

> The container image itself is built on the `dev` branch. On `master` you just run the
> published image.

## Prerequisites

- An **NVIDIA GPU** (compute 6.1–9.0; see [GPU coverage](#gpu-coverage)) with current drivers.
- **Docker** + the **NVIDIA Container Toolkit** (so containers can access the GPU).
- **[`uv`](https://docs.astral.sh/uv/)**, used by the model download script:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Quick start

### 1. Clone the default branch

```bash
git clone https://github.com/openmtx/docker-llamacpp-cuda.git
cd docker-llamacpp-cuda
```

This checks out `master` (the default, run-only branch).

### 2. Download a model

`hf_download.sh` fetches GGUF files from Hugging Face into `~/.cache/llm/models/`, which
the container mounts at `/models`:

```bash
# Generic form:  ./hf_download.sh <hf-repo>[:<quant-or-filename>]
./hf_download.sh unsloth/gemma-3-1b-it-GGUF:Q4_K_XL
```

The script:

- requires `uv` (it runs `uvx hf download`),
- resolves a quantization tag (e.g. `Q4_K_XL`) to the matching filename,
- merges split `.gguf.*` parts automatically,
- supports `--force` (re-download), `--remove` (delete), and a trailing `--dry-run` (list files).

Run `./hf_download.sh --help` for the full reference.

### 3. Configure `models.ini`

```bash
cp models.ini.sample models.ini
```

Edit a preset's `model =` line to point at the file you downloaded, using the
**in-container** path `/models/<filename>`:

```ini
[my-model]
model = /models/gemma-3-1b-it-Q4_K_XL.gguf
ctx-size = 32768
flash-attn = on
cache-type-k = q8_0
cache-type-v = q8_0
n-gpu-layers = 99
load-on-startup = true
```

A preset is loaded on server start only when `load-on-startup = true`; the rest stay
registered as aliases for on-demand use. See `models.ini.sample` for all available knobs.

### 4. (Optional) set an API key

To require an API key on the endpoint, create a `.env` file (already gitignored) next to
`docker-compose.yml`:

```bash
echo "LLAMA_API_KEY=your-secret-key" > .env
```

Docker Compose reads `.env` automatically and passes `LLAMA_API_KEY` into the container.
Skip this step if you don't want key-protected access.

### 5. Run

```bash
docker compose pull        # pulls ghcr.io/openmtx/llamacpp-cuda:latest
docker compose up -d       # starts llama-server on http://localhost:8000
```

Verify it's up:

```bash
curl http://localhost:8000/health
docker compose logs -f server
```

You now have an OpenAI-compatible API served at `http://localhost:8000`.

## Configuration reference

### `models.ini`

Each preset can set GPU layers (`n-gpu-layers`), tensor split (`tensor-split`), context
size (`ctx-size`), KV cache quantization (`cache-type-k` / `cache-type-v`), speculative
decoding (`spec-*`), and sampling parameters. See `models.ini.sample` for the full set.

#### K/V cache quantization

Reduce VRAM usage by quantizing the K and/or V cache:

```ini
cache-type-k = q8_0
cache-type-v = q8_0
```

Options: `f16` (default), `q8_0`, `q4_0`, `q4_1`. K and V can differ. Lower precision
trades a little quality for substantially less memory.

## GPU coverage

The image ships CUDA fat binaries for five architectures, so one container runs on any of them:

| Compute | Architecture | Example GPUs |
|---------|--------------|--------------|
| 6.1     | Pascal       | GTX 1080, GTX 1070, Quadro P5000 |
| 7.5     | Turing       | RTX 2080, RTX 2060, Quadro RTX 5000 |
| 8.6     | Ampere       | RTX 3090, RTX 3080, RTX 3060 |
| 8.9     | Ada Lovelace | RTX 4090, RTX 4080, RTX 4060 |
| 9.0     | Hopper       | H100, H200 |

## Image details

Characteristics of the published image (built on the `dev` branch):

- **`GGML_CUDA_FA_ALL_QUANTS=ON`** — precompiles CUDA kernels for all quantization types, avoiding runtime JIT overhead.
- **`GGML_CUDA_FORCE_MMQ=ON`** — forces matrix × quantized-matrix kernels (better performance on multi-GPU setups without NVLink, e.g. dual P5000).

## Tests

Tests in `tests/` run against a live server:

```bash
python tests/run_all.py                        # all suites
python tests/run_all.py --only test_reasoning  # a specific suite
python tests/run_all.py --verbose              # verbose failure output
```

Grader unit tests need no server:

```bash
python tests/test_graders.py
```
