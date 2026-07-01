import importlib
import sys
import time
from argparse import Namespace
from pathlib import Path


SUITES = [
    "test_speed",
    "test_needle",
    "test_programming",
    "test_tools",
    "test_reasoning",
    "test_world_knowledge",
    "test_hallucination",
    "test_vision",
]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run all test suites")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--verbose", action="store_true",
                    help="print full responses for failures")
    ap.add_argument("--only", default=None,
                    help="run only this suite (e.g. test_speed)")
    ap.add_argument("--list", action="store_true",
                    help="list available suites")
    args = ap.parse_args()

    suites = [args.only] if args.only else SUITES
    if args.list:
        print("Available suites:")
        for s in SUITES:
            print(f"  {s}")
        sys.exit(0)

    here = Path(__file__).parent
    sys.path.insert(0, str(here))

    namespace = Namespace(
        model=args.model,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    results = []
    for s in suites:
        if s not in SUITES:
            print(f"Unknown suite: {s}")
            continue
        print(f"\n{'#' * 60}")
        print(f"#  {s}")
        print(f"{'#' * 60}")
        t0 = time.time()
        mod = importlib.import_module(s)
        try:
            mod.main(args=namespace)
            ok = True
        except SystemExit as e:
            ok = e.code == 0
        except Exception as e:
            print(f"\n  CRASH: {e}")
            ok = False
        elapsed = time.time() - t0
        results.append((s, ok, elapsed))

    print(f"\n{'=' * 60}")
    print("OVERALL")
    print(f"{'=' * 60}")
    all_ok = True
    for s, ok, elapsed in results:
        all_ok = all_ok and ok
        print(f"  {s:25s} {'PASS' if ok else 'FAIL'}  ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
