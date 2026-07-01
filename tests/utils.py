import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


RESULTS = {"total": 0, "passed": 0, "failures": []}


def reset_results():
    RESULTS.clear()
    RESULTS["total"] = 0
    RESULTS["passed"] = 0
    RESULTS["failures"] = []


def log_result(name: str, ok: bool, detail: str = ""):
    RESULTS["total"] += 1
    if ok:
        RESULTS["passed"] += 1
    else:
        RESULTS["failures"].append((name, detail))


def print_summary():
    p = RESULTS["passed"]
    t = RESULTS["total"]
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {p}/{t} passed  ({p / max(t, 1):.0%})")
    if RESULTS["failures"]:
        print(f"FAILURES:")
        for name, detail in RESULTS["failures"]:
            print(f"  ✗ {name}: {detail[:120]}")
    print(f"{'=' * 60}")
    return p == t


def chat(chat_url, headers, model, messages, temperature=0.0, max_tokens=1024,
         tools=None, timeout=600):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    t0 = time.time()
    r = requests.post(chat_url, json=payload, headers=headers, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]["message"]
    text = (choice.get("content") or "").strip()
    tool_calls = choice.get("tool_calls", [])
    usage = data.get("usage", {})
    return text, tool_calls, dt, usage


def completion(comp_url, headers, model, prompt, n_predict=128,
               ctx_size=65536, timeout=900):
    payload = {
        "prompt": prompt,
        "model": model,
        "n_predict": n_predict,
        "cache_prompt": False,
        "n_ctx": ctx_size,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(comp_url, json=payload, headers=headers, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    data = r.json()
    timings = data.get("timings", {})
    prompt_n = timings.get("prompt_n", 0) or 1
    prompt_ms = timings.get("prompt_ms", 0) or 1
    pred_n = timings.get("predicted_n", 0) or 1
    pred_ms = timings.get("predicted_ms", 0) or 1
    return {
        "prefill_tok_s": prompt_n / (prompt_ms / 1000),
        "decode_tok_s": pred_n / (pred_ms / 1000),
        "prompt_n": prompt_n,
        "pred_n": pred_n,
        "dt": dt,
    }


# --- Coding eval helpers ---

_RUNNER = r'''
import json, sys, traceback
_PAYLOAD = json.loads(sys.stdin.read())
_NS = {"__name__": "__eval__"}
try:
    exec(_PAYLOAD["code"], _NS)
except Exception:
    print(json.dumps({"error": traceback.format_exc()[-800:]}))
    sys.exit(0)
_f = _NS.get(_PAYLOAD["func"])
if not callable(_f):
    print(json.dumps({"error": "function " + str(_PAYLOAD["func"]) + " not defined"}))
    sys.exit(0)
_out = []
for _args in _PAYLOAD["args"]:
    try:
        _got = _f(*list(_args))
        _out.append({"ok": True, "got": _got})
    except Exception as _e:
        _out.append({"ok": False, "err": str(_e)[:200]})
print(json.dumps({"results": _out}))
'''


def extract_code(text: str) -> str:
    m = re.search(r"```[a-zA-Z0-9]*\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    if "def " in text:
        return text[text.index("def "):]
    return text


def run_code(code: str, func: str, args_list: list, timeout: int = 15) -> dict:
    payload = json.dumps({"code": code, "func": func, "args": args_list})
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_RUNNER)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path],
                              input=payload, capture_output=True, text=True,
                              timeout=timeout,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    finally:
        Path(path).unlink(missing_ok=True)
    out = proc.stdout.strip().splitlines()
    if not out:
        return {"error": (proc.stderr or "no output")[-400:]}
    try:
        return json.loads(out[-1])
    except Exception:
        return {"error": out[-1][-400:]}


def grade_coding(response: str, func: str, tests: list, mode: str = "eq"):
    code = extract_code(response)
    if "def " not in code:
        return False, "no function defined"
    args_list = [t[0] for t in tests]
    res = run_code(code, func, args_list)
    if "error" in res:
        return False, f"exec error: {res['error'][:120]}"
    results = res.get("results", [])
    detail = []
    all_ok = True
    for got_entry, (args, expected) in zip(results, tests):
        if not got_entry.get("ok"):
            all_ok = False
            detail.append(f"{args} -> ERROR: {got_entry.get('err')}")
            continue
        got = got_entry["got"]
        if mode == "set":
            try:
                ok = set(got) == set(expected)
            except TypeError:
                ok = False
        else:
            ok = got == expected
        if not ok:
            all_ok = False
            detail.append(f"{args} -> got {got!r}, want {expected!r}")
    return all_ok, "; ".join(detail)[:160] if detail else "all tests pass"
