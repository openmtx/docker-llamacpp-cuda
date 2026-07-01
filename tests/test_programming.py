import sys
import time

import requests

from config import build_args, resolve
from utils import (RESULTS, chat, extract_code, grade_coding, log_result,
                   print_summary, reset_results)


# (id, prompt, func_name, [(args_tuple, expected)], check_mode)
CODING_CASES = [
    ("C1-lsus",
     "Write a Python function `solve(s)` that returns the length of the "
     "longest substring of `s` containing no repeating characters. "
     "Only output the code in a ```python block.",
     "solve",
      [(("abcabcbb",), 3), (("bbbbb",), 1), (("pwwkew",), 3),
       (("",), 0), (("abcdef",), 6), (("aab",), 2), (("dvdf",), 3)],
     "eq"),
    ("C2-parens",
     "Write a Python function `solve(s)` that returns True if the string `s` "
     "contains properly matched and nested brackets '()[]{}', else False. "
     "Only output the code in a ```python block.",
     "solve",
      [(("()",), True), (("()[]{}",), True), (("(]",), False),
       (("([)]",), False), (("{[]}",), True), (("",), True), (("(",), False)],
     "eq"),
    ("C3-twosum",
     "Write a Python function `solve(nums, target)` that returns a list of "
     "the two indices [i, j] such that nums[i]+nums[j]==target. Exactly one "
     "solution exists. Only output the code in a ```python block.",
     "solve",
     [(([2, 7, 11, 15], 9), {0, 1}), (([3, 2, 4], 6), {1, 2}),
      (([3, 3], 6), {0, 1}), (([-1, -2, -3, -4, -5], -8), {2, 4})],
     "set"),
    ("C4-mincoins",
     "Write a Python function `solve(coins, amount)` that returns the "
     "minimum number of coins needed to make `amount`, or -1 if impossible. "
     "Only output the code in a ```python block.",
     "solve",
      [(([1, 2, 5], 11), 3), (([2], 3), -1), (([1], 0), 0),
       (([1, 2, 5], 100), 20), (([186, 419, 83, 408], 6249), 20)],
     "eq"),
    ("C5-wordbreak",
     "Write a Python function `solve(s, word_dict)` that returns True if `s` "
     "can be segmented into a space-separated sequence of words all present "
     "in `word_dict` (a list of strings). Only output the code in a "
     "```python block.",
     "solve",
      [(("leetcode", ["leet", "code"]), True),
       (("applepenapple", ["apple", "pen"]), True),
       (("catsandog", ["cats", "dog", "sand", "and", "cat"]), False),
       (("aaaaaaa", ["aaaa", "aaa"]), True)],
     "eq"),
]


def test_coding_eval(chat_url, headers, model, timeout, verbose=False):
    print("\n=== Code Generation & Testing ===\n")
    for cid, prompt, func, tests, mode in CODING_CASES:
        print(f"  {cid}: ", end="", flush=True)
        try:
            resp, _, _, _ = chat(chat_url, headers, model,
                                 [{"role": "user", "content": prompt}],
                                 temperature=0.0, max_tokens=1536,
                                 timeout=timeout)
        except Exception as e:
            print(f"REQUEST ERROR: {e}")
            log_result(f"coding_{cid}", False, str(e))
            continue
        ok, detail = grade_coding(resp, func, tests, mode)
        tag = "PASS" if ok else "FAIL"
        print(f"{tag}  {detail[:80]}")
        log_result(f"coding_{cid}", ok, detail)


CODEGEN_TASKS = [
    ("Write a Python function that implements binary search. "
     "Include type hints and a docstring.",
     ["def ", "mid"]),
    ("Write a JavaScript debounce function that takes a function and delay.",
     ["settimeout"]),
    ("Write a SQL query to find top 5 customers by total order amount "
     "from customers(id,name,email) and orders(customer_id,amount).",
     ["select", "order by"]),
    ("Write a Python async function using aiohttp that fetches "
     "a list of URLs concurrently.",
     ["async", "aiohttp"]),
]


def test_code_generation(chat_url, headers, model, timeout, verbose=False):
    print("\n=== Code Generation Tasks ===\n")
    for i, (task, keywords) in enumerate(CODEGEN_TASKS):
        print(f"  task_{i+1}: ", end="", flush=True)
        try:
            resp, _, _, _ = chat(chat_url, headers, model,
                                 [{"role": "user", "content": task}],
                                 temperature=0.0, max_tokens=800,
                                 timeout=timeout)
        except Exception as e:
            print(f"ERROR: {e}")
            log_result(f"codegen_{i+1}", False, str(e))
            continue
        low = resp.lower()
        has_fence = "```" in resp
        missing = [k for k in keywords if k not in low]
        ok = has_fence and len(resp.strip()) >= 50 and not missing
        if not has_fence:
            detail = "no code fence"
        elif missing:
            detail = f"missing {missing}"
        else:
            detail = f"{len(resp)} chars"
        tag = "PASS" if ok else "FAIL"
        print(f"{tag}: {detail}")
        log_result(f"codegen_{i+1}", ok, detail)
        time.sleep(1)


def test_bug_fixing(chat_url, headers, model, timeout, verbose=False):
    print("\n=== Bug Fixing ===\n")
    bugs = [
        ("lists",
         "Find the bug: def get_squares(nums):\n"
         "    squares = []\n"
         "    for n in nums:\n"
         "        squares = squares.append(n ** 2)\n"
         "    return squares\n",
         lambda code, resp: (
             "= squares.append" not in code
             and ("squares.append" in code or "squares += " in code
                  or ".extend" in code),
             "still reassigns list.append" if "= squares.append" in code
             else "no corrected append")),
        ("async",
         "Find the bug: async function fetchData() {\n"
         "  const response = await fetch('/api/data');\n"
         "  return response.json();\n"
         "}\n"
         "async function main() {\n"
         "  const data = fetchData();\n"
         "  console.log(data.name);\n"
         "}\n",
         lambda code, resp: (
             "await fetchdata" in code.lower(),
             "missing 'await' on fetchData()")),
    ]
    for cid, prompt, check in bugs:
        print(f"  bug_{cid}: ", end="", flush=True)
        try:
            resp, _, _, _ = chat(chat_url, headers, model,
                                 [{"role": "user",
                                   "content": f"{prompt}\nWhat's wrong and "
                                              "how to fix it?"}],
                                 temperature=0.0, max_tokens=800,
                                 timeout=timeout)
        except Exception as e:
            print(f"ERROR: {e}")
            log_result(f"bugfix_{cid}", False, str(e))
            continue
        code = extract_code(resp)
        ok, detail = check(code, resp)
        tag = "PASS" if ok else "FAIL"
        print(f"{tag} ({len(resp)} chars) {detail}")
        log_result(f"bugfix_{cid}", ok, f"{len(resp)} chars; {detail}")
        time.sleep(1)


def main(args=None):
    ap = build_args("Programming ability tests")
    if args is None:
        args = ap.parse_args()
    chat_url, _, headers, model, timeout = resolve(args)
    try:
        requests.get(f"http://{args.host}:{args.port}/health",
                     timeout=5).raise_for_status()
    except Exception as e:
        print(f"Server not reachable: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Server: {args.host}:{args.port}  |  Model: {model}")
    verbose = getattr(args, 'verbose', False)
    reset_results()
    test_coding_eval(chat_url, headers, model, timeout, verbose)
    test_code_generation(chat_url, headers, model, timeout, verbose)
    test_bug_fixing(chat_url, headers, model, timeout, verbose)
    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
