#!/usr/bin/env python3
"""
Benchmark prefill & decode speeds across context window sizes.

Usage:
    pip install requests
    python tests/bench_speed.py [--host HOST] [--port PORT] [--model ALIAS]
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

PAD = "The quick brown fox jumps over the lazy dog. " * 50
N_WARMUP = 1
N_RUNS = 3
CTX_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768]
GEN_TOKENS = 128


def build_prompt(target_chars: int) -> str:
    reps = target_chars // len(PAD) + 1
    return (PAD * reps)[:target_chars]


def run_completion(url: str, model: str, prompt: str, ctx: int, api_key: str | None) -> dict:
    payload = {
        "prompt": prompt,
        "model": model,
        "n_predict": GEN_TOKENS,
        "cache_prompt": False,
        "n_ctx": ctx,
        "temperature": 0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.post(f"{url}/completion", json=payload, headers=headers, timeout=900)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark prefill/decode speeds")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--api-key")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    try:
        requests.get(f"{base_url}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Error: cannot reach {base_url} — {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Server: {base_url}  |  Model: {args.model}")
    print(f"Generate: {GEN_TOKENS} tok/run  |  Warmup: {N_WARMUP}  |  Runs: {N_RUNS}")
    print()
    print(f"{'ctx-size':>10} | {'prompt_len':>10} | {'prefill(t/s)':>12} | {'decode(t/s)':>12}")
    print("-" * 52)

    for ctx in CTX_SIZES:
        prompt_chars = ctx * 3
        prompt = build_prompt(prompt_chars)

        try:
            for _ in range(N_WARMUP):
                run_completion(base_url, args.model, prompt, ctx, args.api_key)
        except Exception as e:
            print(f"{ctx:>10} | {'FAILED':>10} | {str(e)[:30]:>26}")
            sys.stdout.flush()
            continue

        prefill_rates: list[float] = []
        decode_rates: list[float] = []
        for _ in range(N_RUNS):
            try:
                data = run_completion(base_url, args.model, prompt, ctx, args.api_key)
            except Exception as e:
                print(f"{ctx:>10} | {'FAILED':>10} | {str(e)[:30]:>26}")
                break
            t = data.get("timings", {})
            pn = t.get("prompt_n", 0) or 1
            pn_ms = t.get("prompt_ms", 0) or 1
            dn = t.get("predicted_n", 0) or 1
            dn_ms = t.get("predicted_ms", 0) or 1
            prefill_rates.append(pn / (pn_ms / 1000))
            decode_rates.append(dn / (dn_ms / 1000))
        else:
            avg_prefill = sum(prefill_rates) / len(prefill_rates)
            avg_decode = sum(decode_rates) / len(decode_rates)
            print(f"{ctx:>10} | {int(pn):>10} | {avg_prefill:>12.1f} | {avg_decode:>12.1f}")
            sys.stdout.flush()

    print()


if __name__ == "__main__":
    main()
