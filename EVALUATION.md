# Hardware Evaluation

## GPU Setup

- 2x NVIDIA Quadro P5000 (16 GB GDDR5X each, 180 W TDP)
- Pascal architecture (compute 6.1)

## Model: Qwen3.6-27B

- Architecture: 27.3B params, 5120 hidden dim, 248k vocab
- Native training ctx: 262144

## Configuration

| Setting | Value |
|---|---|
| `tensor-split` | 50,50 |
| `flash-attn` | on |
| `cache-type-k/v` | q8_0 |
| `n-gpu-layers` | 99 |
| `spec-type` | draft-mtp (2 tokens) |

## Context Size Results

| ctx-size | GPU 0 Used | GPU 1 Used | GPU 0 Free | GPU 1 Free | Status |
|---|---|---|---|---|---|
| 131072 | 11066 MiB | 13130 MiB | 5206 MiB | 3134 MiB | OK |
| 163840 | 11866 MiB | 14186 MiB | 4406 MiB | 2078 MiB | OK |
| 196608 | 12666 MiB | 15242 MiB | 3606 MiB | 1022 MiB | OK |
| 262144 (native) | — | — | — | — | OOM (estimated) |

**Max stable ctx-size: 196608 (192K)**

## Quantization & MTP

| Quant | With MTP Heads? | Decode Speed |
|---|---|---|
| `Q4_K_M` (17.1 GB) | No | ~8.8 tok/s |
| `UD-Q4_K_XL` (17.9 GB) | **Yes** | ~16.0 tok/s |

The Unsloth Dynamic (`UD-*`) quants preserve the MTP (Multi-Token Prediction) heads needed for speculative decoding. Standard `Q*_K_*` quants drop them.

## Inference Speed (UD-Q4_K_XL)

| ctx-size | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|
| 512 | 150 | 16.6 |
| 1K | 173 | 16.8 |
| 2K | 201 | 16.3 |
| 4K | 235 | 18.0 |
| 8K | 241 | 15.2 |
| 16K | 240 | 14.9 |
| 32K | 220 | 12.9 |
| 64K | 186 | 12.9 |
| 128K | 141 | 8.4 |
| 192K | — | — |

Decode speed: **~16.0 tok/s** (short prompt, MTP draft acceptance ~71%, mean len 2.42).

## Evaluation Results

### Qwen3.6-27B (Dense)

| Test | Result |
|---|---|
| Graders (unit tests) | **45/45** PASS |
| World Knowledge | **11/13** PASS |
| Tool Calling | **7/7** PASS |
| Reasoning & Math | **20/24** PASS |
| API/Path Precision | **14/14** PASS |
| Needle-in-Haystack (80K) | **1/1** PASS (366s) |
| Code Generation | **10/11** PASS |
| **Total** | **108/114 (95%)** |
| Decode Speed | ~16 tok/s |

### Qwen3.6-35B-A3B (MoE, 3B active)

| Test | Result |
|---|---|
| Graders (unit tests) | **45/45** PASS |
| World Knowledge | **11/13** PASS |
| Tool Calling | **7/7** PASS |
| Reasoning & Math | **19/24** PASS |
| API/Path Precision | **14/14** PASS |
| Needle-in-Haystack (80K) | **1/1** PASS (**116s**, 3x faster) |
| Code Generation | **11/11** PASS |
| **Total** | **109/116 (94%)** |
| Decode Speed | **~40 tok/s** |

The MoE model matches the dense model on most metrics while being **~2.5x faster** at decode and **~3x faster** on long-context needle recall. It also fixed the bugfix test that the 27B failed.

### Verdict

Both models are strong on code/API tasks. The 35B-A3B MoE is the clear winner on these P5000s — same accuracy at 2.5x the speed.
