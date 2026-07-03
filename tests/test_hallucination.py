import os
import re
import sys

import requests

from config import build_args, resolve
from utils import RESULTS, chat, log_result, print_summary, reset_results


WORK_DIR = "/home/user/projects/sst_s3_rwo/"

_WS_BLOCK = (
    "```python\n"
    "def initialize_pipeline(config):\n"
    "    setup_logger(config.log_level)" + " " * 4 + "\n"
    "    load_plugins()" + " " * 2 + "\n"
    "    return True\n"
    "```"
)
_LOG_TAIL = "log_level)" + " " * 4
_PLUG_TAIL = "plugins()" + " " * 2


def _check_cd_path(resp_low: str, expected_subpath: str) -> bool:
    m = re.search(r'cd\s+(.+)', resp_low)
    if not m:
        return False
    path = m.group(1).strip().rstrip("/")
    if not path.startswith("/"):
        path = os.path.normpath(WORK_DIR + path)
    return expected_subpath in path


HALLUCINATION_CASES = [
    # --- Path recall ---
    ("PATH1-repo-root",
     "You are working on a project at /home/user/projects/sst_s3_rwo/. "
     "Run a grep command to find where DirtySet is defined. "
     "Output only the exact command, no explanation.",
     lambda low: "sst_s3_rwo" in low and "sst_s3_rgo" not in low,
     "correct path"),
    ("PATH2-subdir",
     "The project root is /home/user/projects/sst_s3_rwo/. "
     "There is a file at internal/driver/cache.go relative to the root. "
     "What is the full absolute path to this file? "
     "Output only the path.",
     lambda low: "sst_s3_rwo/internal/driver/cache" in low
                 and "sst_s3_rgo" not in low,
     "correct path"),
    ("PATH3-config-path",
     "The project is at /home/user/projects/sst_s3_rwo/. "
     "List the contents of the configs/ directory. "
     "Output only the exact ls command.",
     lambda low: "sst_s3_rwo/configs" in low and "sst_s3_rgo" not in low,
     "correct path"),
    ("PATH4-src-dir",
     "Working directory: /home/user/projects/sst_s3_rwo/. "
     "Change directory to the pkg/storage/ subdirectory. "
     "Output only the exact cd command.",
     lambda low: _check_cd_path(low, "sst_s3_rwo/pkg/storage"),
     "correct path"),
    ("PATH5-locked-root",
     "Your active project root workspace is strictly locked to "
     "'/srv/www/production/app/api/auth/v3'. There are no other directories "
     "on this system. Never reference default folders like '/src', "
     "'/workspace', or './utils' unless they are explicitly present in the "
     "path above. An uncaught exception was thrown in token verification. "
     "Which file under your active root workspace needs to be read to locate "
     "the route handler? Output only the file path.",
     lambda low: "/srv/www/production" in low
                 and not any(p in low for p in
                             ["/src", "/workspace", "./utils", "/app/src"]),
     "stayed within locked root"),

    # --- Identifier recall ---
    ("ID1-method-name",
     "In a Go project, there is a function defined in internal/driver/cache.go: "
     "func (c *Cache) DirtySet(key string) bool { ... }. "
     "Write a call to this function on a Cache instance named 'cache', passing "
     "the string 'test-key'. Output only the Go code line.",
     lambda low: "dirtyset" in low
                 and not any(w in low for w in ["dirtyrange", "dirtylist"]),
     "correct identifier"),
    ("ID2-function-name",
     "The project has these helper functions:\n"
     "  func MinUInt64(a, b uint64) uint64\n"
     "  func MinInt64(a, b int64) int64\n"
     "  func MaxUInt64(a, b uint64) uint64\n"
     "Call the function that returns the smaller of two uint64 values, "
     "with arguments x and y. Output only the call.",
     lambda low: bool(re.search(r'minuint64\b', low))
                 and not bool(re.search(r'\bminint64\b|\bmin\b[^(]', low)),
     "correct identifier"),
    ("ID3-type-name",
     "The project defines:\n"
     "  type GetBlobInput struct {\n"
     "      Start uint64\n"
     "      Length uint64\n"
     "  }\n"
     "Create a GetBlobInput literal with Start=0 and Length=1024. "
     "Output only the Go code.",
     lambda low: "getblobinput" in low
                 and re.search(r'\bstart\s*:', low) is not None
                 and re.search(r'\blength\s*:', low) is not None
                 and "int64" not in low and "int32" not in low,
     "correct identifier"),

    # --- API signature recall ---
    ("API1-interface",
     "The project has: type ReadSeeker interface { io.Reader; io.Seeker }. "
     "Write a function signature that takes a ReadSeeker parameter named 'rs'. "
     "Output only the function signature line.",
     lambda low: "readseeker" in low
                 and "seeker" not in low.replace("readseeker", ""),
     "correct API usage"),
    ("API2-struct-field",
     "Given this struct:\n"
     "  type MBuf struct {\n"
     "      buffers [][]byte\n"
     "      wp      int\n"
     "  }\n"
     "Append data to the current buffer at index wp. "
     "Output only the Go code line.",
     lambda low: "buffers[wp]" in low or "buffers[m.wp]" in low
                 or ("buffers" in low and "wp" in low),
     "correct API usage"),

    # --- Edit tool compliance ---
    ("EDIT1-read-before-edit",
     "You need to edit file /path/to/file.go. "
     "What should you do IMMEDIATELY before calling the edit tool? "
     "Answer in one short sentence.",
     lambda low: any(w in low for w in ["read", "cat", "view", "open"]),
     "correct practice"),
    ("EDIT2-exact-match",
     "The edit tool's oldText parameter must match the file contents exactly. "
     "True or False? Output only True or False.",
     lambda low: low.strip(" \n.,!?") == "true",
     "correct answer"),
    ("EDIT3-whitespace-mirror",
     "You are a file editing tool. To update files, you must mirror the "
     "target block characters and indentation spaces EXACTLY as they appear, "
     "or the system tool will fail. Replicate the following code block exactly "
     "inside a 'SEARCH' block, preserving the trailing whitespace at the end "
     "of lines 2 and 3:\n" + _WS_BLOCK,
     lambda low: _LOG_TAIL in low and _PLUG_TAIL in low,
     "exact whitespace mirror"),
]

PRECISION_CASES = [
    ("T1-similar-suffix",
     "Pick the correct option from the list: [rwo, rgo, rwo, rvo]. "
     "Choose the one that matches 'rwo'. Output only the word.",
     lambda low: low.strip(" \n.,!?\"'`") in {"rwo"}),
    ("T2-similar-prefix",
     "Pick: [DirtySet, DirtyRanges, DirtyList, DirtyMap]. "
     "Choose the one called DirtySet. Output only the word.",
     lambda low: low.strip(" \n.,!?\"'`") in {"dirtyset"}),
    ("T3-uint-vs-int",
     "Pick: [MinUInt64, MinInt64, Min, MaxUInt64]. "
     "Choose the one called MinUInt64. Output only the word.",
     lambda low: low.strip(" \n.,!?\"'`") in {"minuint64", "minuint64)"}),
]


def run_test(cid, prompt, grader, ok_msg, chat_url, headers, model, timeout):
    print(f"  {cid}: ", end="", flush=True)
    try:
        resp, _, dt, _ = chat(chat_url, headers, model,
                              [{"role": "user", "content": prompt}],
                              temperature=0.0, max_tokens=100,
                              timeout=timeout)
    except Exception as e:
        print(f"ERROR: {e}")
        log_result(cid, False, str(e))
        return
    low = resp.lower()
    ok = grader(low)
    detail = f"resp={resp[:80]!r}" if not ok else ok_msg
    tag = "PASS" if ok else "FAIL"
    print(f"{tag}  {detail}")
    log_result(cid, ok, detail)


def test_hallucination_paths(chat_url, headers, model, timeout):
    print("\n=== Path Hallucination ===\n")
    for cid, prompt, grader, ok_msg in HALLUCINATION_CASES:
        if not cid.startswith("PATH"):
            continue
        run_test(cid, prompt, grader, ok_msg, chat_url, headers, model, timeout)


def test_hallucination_identifiers(chat_url, headers, model, timeout):
    print("\n=== Identifier Hallucination ===\n")
    for cid, prompt, grader, ok_msg in HALLUCINATION_CASES:
        if not cid.startswith("ID"):
            continue
        run_test(cid, prompt, grader, ok_msg, chat_url, headers, model, timeout)


def test_hallucination_apis(chat_url, headers, model, timeout):
    print("\n=== API/Type Hallucination ===\n")
    for cid, prompt, grader, ok_msg in HALLUCINATION_CASES:
        if not cid.startswith("API") and not cid.startswith("EDIT"):
            continue
        run_test(cid, prompt, grader, ok_msg, chat_url, headers, model, timeout)


def test_precision(chat_url, headers, model, timeout):
    print("\n=== Token-Level Precision ===\n")
    for cid, prompt, grader in PRECISION_CASES:
        print(f"  {cid}: ", end="", flush=True)
        try:
            resp, _, dt, _ = chat(chat_url, headers, model,
                                  [{"role": "user", "content": prompt}],
                                  temperature=0.0, max_tokens=50,
                                  timeout=timeout)
        except Exception as e:
            print(f"ERROR: {e}")
            log_result(cid, False, str(e))
            continue
        low = resp.lower().strip(" \n.,!?\"'`")
        ok = grader(low)
        detail = f"got={resp[:60]!r}" if not ok else "correct pick"
        tag = "PASS" if ok else "FAIL"
        print(f"{tag}  {detail}")
        log_result(cid, ok, detail)


def main(args=None):
    ap = build_args("Hallucination & precision tests")
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
    reset_results()
    test_hallucination_paths(chat_url, headers, model, timeout)
    test_hallucination_identifiers(chat_url, headers, model, timeout)
    test_hallucination_apis(chat_url, headers, model, timeout)
    test_precision(chat_url, headers, model, timeout)
    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
