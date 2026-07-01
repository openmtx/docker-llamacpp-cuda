import base64
import sys
from pathlib import Path

import requests

from config import build_args, resolve
from utils import RESULTS, chat, log_result, print_summary, reset_results


def image_data_url(path: str) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


CASES = [
    ("currency-id",
     "What currency is shown in this image? Be specific.",
     ["togrog", "mongolia", "mongolian"]),
    ("currency-value",
     "What denomination is this banknote?",
     ["100", "hundred"]),
    ("language",
     "What script is used on the front of this banknote?",
     ["mongol", "cyrillic"]),
]


def test_vision(chat_url, headers, model, timeout, verbose=False):
    img = image_data_url("tests/images/mongolia_money_100Togrog.jpg")
    print("\n=== Vision / Multimodal ===\n")
    for cid, question, keywords in CASES:
        print(f"  {cid}: ", end="", flush=True)
        try:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": img}},
                ]}
            ]
            resp, _, dt, usage = chat(chat_url, headers, model,
                                       messages, temperature=0.0,
                                       max_tokens=512, timeout=timeout)
        except Exception as e:
            print(f"ERROR: {e}")
            log_result(cid, False, str(e))
            continue
        resp_lower = resp.lower()
        ok = any(kw in resp_lower for kw in keywords)
        tag = "PASS" if ok else "FAIL"
        print(f"{tag}  ({dt:.1f}s)")
        if not ok:
            detail = f"no keyword match; got={resp[:120]!r}"
            log_result(cid, False, detail)
            if verbose:
                print(f"        Resp: {resp!r}")
        else:
            log_result(cid, True)


def main(args=None):
    ap = build_args("Vision test suite")
    a = ap.parse_args(args)
    chat_url, _, headers, model, timeout = resolve(a)
    print(f"Server: {a.host}:{a.port}  |  Model: {model}\n")
    reset_results()
    test_vision(chat_url, headers, model, timeout, verbose=a.verbose)
    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
