import random
import string
import sys
import time

import requests

from config import build_args, resolve
from utils import RESULTS, log_result, print_summary, reset_results, completion


random.seed(42)
PAD = "The quick brown fox jumps over the lazy dog. " * 50
CTX_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]
N_WARMUP = 1
N_RUNS = 3
GEN_TOKENS = 128
MIN_DECODE_TOK_S = 10


def estimate_decode_speed(comp_url, headers, model, timeout):
    print("\n=== Decode Speed Check ===\n")
    prompt = "Write a detailed story about a futuristic city."
    try:
        r = completion(comp_url, headers, model, prompt,
                       n_predict=256, timeout=timeout)
        speed = r["decode_tok_s"]
        print(f"  Warmup: {speed:.1f} tok/s  (gen={r['pred_n']})")
        print(f"  --> Decode speed: {speed:.1f} tok/s")
        return speed
    except Exception as e:
        print(f"  FAILED: {e}")
        return 0


def test_decode_speed(comp_url, headers, model, timeout, estimate):
    print(f"\n=== Decode Speed (no prefill) ===\n")
    prompt = "Write a detailed story about a futuristic city."
    speeds = [estimate]
    for i in range(N_RUNS):
        try:
            r = completion(comp_url, headers, model, prompt,
                           n_predict=256, timeout=timeout)
        except Exception as e:
            print(f"  FAIL: {e}")
            continue
        speeds.append(r["decode_tok_s"])
        print(f"  Run {i + 1}: {r['decode_tok_s']:.1f} tok/s  (gen={r['pred_n']})")
    if speeds:
        avg = sum(speeds) / len(speeds)
        log_result(f"decode_speed", avg > 5, f"{avg:.1f} tok/s")
        print(f"  --> Average decode: {avg:.1f} tok/s")


def test_prefill_sweep(comp_url, headers, model, timeout):
    print("\n=== Prefill Sweep ===\n")
    print(f"{'ctx-size':>10} | {'prompt_len':>10} | {'prefill(t/s)':>12} | "
          f"{'decode(t/s)':>12}")
    print("-" * 52)
    for ctx in CTX_SIZES:
        prompt_chars = ctx * 3
        reps = prompt_chars // len(PAD) + 1
        prompt = (PAD * reps)[:prompt_chars]
        try:
            for _ in range(N_WARMUP):
                completion(comp_url, headers, model, prompt,
                           n_predict=GEN_TOKENS, ctx_size=ctx, timeout=timeout)
        except Exception as e:
            print(f"{ctx:>10} | {'FAILED':>10} | {str(e)[:30]:>26}")
            sys.stdout.flush()
            continue
        prefill = []
        decode = []
        for _ in range(N_RUNS):
            try:
                r = completion(comp_url, headers, model, prompt,
                               n_predict=GEN_TOKENS, ctx_size=ctx,
                               timeout=timeout)
            except Exception as e:
                print(f"{ctx:>10} | {'FAILED':>10} | {str(e)[:30]:>26}")
                break
            prefill.append(r["prefill_tok_s"])
            decode.append(r["decode_tok_s"])
        else:
            ap = sum(prefill) / len(prefill)
            ad = sum(decode) / len(decode)
            print(f"{ctx:>10} | {r['prompt_n']:>10} | {ap:>12.1f} | {ad:>12.1f}")
            sys.stdout.flush()
    log_result("prefill_sweep", True, "completed")
    print()


def test_prefill_large_burst(chat_url, headers, model, timeout):
    print("\n=== Prefill Burst (50K context) ===\n")
    chars_needed = int(50000 * 1.55)
    haystack = "".join(random.choices(string.ascii_letters + " ", k=chars_needed))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize this: {haystack}"},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    start = time.time()
    try:
        r = requests.post(chat_url, json=payload, headers=headers,
                          timeout=timeout)
        elapsed = time.time() - start
        if r.status_code == 200:
            data = r.json()
            pt = data.get("usage", {}).get("prompt_tokens", 0)
            tps = pt / elapsed if elapsed > 0 else 0
            log_result("prefill_burst_50k", tps > 500,
                       f"{pt:,} tok in {elapsed:.1f}s = {tps:.0f} tok/s")
            print(f"  Tokens: {pt:,} | Time: {elapsed:.1f}s | "
                  f"Speed: {tps:.0f} tok/s")
        else:
            log_result("prefill_burst_50k", False, f"HTTP {r.status_code}")
    except Exception as e:
        log_result("prefill_burst_50k", False, str(e))
        print(f"  FAILED: {e}")


def main(args=None):
    ap = build_args("Speed benchmarks", add_url=False)
    if args is None:
        args = ap.parse_args()
    chat_url, comp_url, headers, model, timeout = resolve(args)
    try:
        requests.get(f"http://{args.host}:{args.port}/health",
                     timeout=5).raise_for_status()
    except Exception as e:
        print(f"Server not reachable: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Server: {args.host}:{args.port}  |  Model: {model}")
    reset_results()
    estimate = estimate_decode_speed(comp_url, headers, model, timeout)
    if estimate < MIN_DECODE_TOK_S:
        print(f"\nDecode speed too slow ({estimate:.1f} < {MIN_DECODE_TOK_S} tok/s), "
              f"skipping further benchmarks.")
        log_result("decode_speed", False, f"{estimate:.1f} tok/s (below {MIN_DECODE_TOK_S})")
        ok = print_summary()
        sys.exit(0 if ok else 1)
    test_decode_speed(comp_url, headers, model, timeout, estimate)
    test_prefill_sweep(comp_url, headers, model, timeout)
    test_prefill_large_burst(chat_url, headers, model, timeout)
    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
