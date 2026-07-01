import argparse
import os
import sys
from pathlib import Path


def env_path(start: Path) -> str | None:
    for p in (start / ".env", start.parent / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith("LLAMA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return os.environ.get("LLAMA_API_KEY")


def build_args(description: str = "Test suite", add_url: bool = True):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--verbose", action="store_true",
                    help="print full responses for failures")
    if add_url:
        ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    return ap


def resolve(args):
    key = args.api_key or env_path(Path(__file__).resolve().parent)
    chat_url = getattr(args, "url", None) or f"http://{args.host}:{args.port}/v1/chat/completions"
    comp_url = f"http://{args.host}:{args.port}/completion"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return chat_url, comp_url, headers, args.model, args.timeout
