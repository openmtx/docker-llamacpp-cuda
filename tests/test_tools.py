import json
import sys

import requests

from config import build_args, resolve
from utils import RESULTS, chat, log_result, print_summary, reset_results


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string",
                                        "description": "City name"}},
            "required": ["location"],
        },
    },
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get current time for a location",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string",
                                        "description": "Timezone"}},
            "required": ["timezone"],
        },
    },
}

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a calculation",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string",
                                          "description": "Math expression"}},
            "required": ["expression"],
        },
    },
}

FLIGHT_TOOL = {
    "type": "function",
    "function": {
        "name": "book_flight",
        "description": "Book a flight",
        "parameters": {
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Origin city"},
                "to": {"type": "string", "description": "Destination city"},
                "date": {"type": "string",
                         "description": "Departure date (YYYY-MM-DD)"},
                "passengers": {"type": "integer",
                               "description": "Number of passengers"},
            },
            "required": ["from", "to", "date"],
        },
    },
}


def test_basic_tool(chat_url, headers, model, timeout, tools, prompt,
                    expected_name):
    resp, calls, dt, usage = chat(chat_url, headers, model,
                                  [{"role": "user", "content": prompt}],
                                  tools=tools, temperature=0.0,
                                  max_tokens=8192, timeout=timeout)
    if calls and calls[0].get("function", {}).get("name") == expected_name:
        args = calls[0].get("function", {}).get("arguments", "")
        return True, f"called {expected_name}: {str(args)[:80]}"
    return False, f"expected {expected_name}, got calls={calls[:2]}"


def test_basic(chat_url, headers, model, timeout):
    print("\n=== Basic Tool Call ===\n")
    tools = [WEATHER_TOOL]
    ok, det = test_basic_tool(chat_url, headers, model, timeout, tools,
                              "What's the weather in Tokyo?", "get_weather")
    log_result("basic_tool", ok, det)
    print(f"  {'PASS' if ok else 'FAIL'}: {det}")


def test_tool_chain(chat_url, headers, model, timeout):
    print("\n=== Tool Result Continuation ===\n")
    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant",
         "tool_calls": [{"id": "call_123", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": '{"location": "Tokyo"}'}}]},
        {"role": "tool", "tool_call_id": "call_123",
         "content": '{"temperature": 22, "condition": "sunny"}'},
    ]
    try:
        resp, calls, dt, _ = chat(chat_url, headers, model, messages,
                                   max_tokens=8192, timeout=timeout)
    except Exception as e:
        log_result("tool_chain", False, str(e))
        print(f"  FAIL: {e}")
        return
    ok = len(resp.strip()) > 0
    log_result("tool_chain", ok, resp[:80])
    print(f"  {'PASS' if ok else 'FAIL'}: {resp[:80]!r}")


def test_multiple_tools(chat_url, headers, model, timeout):
    print("\n=== Multiple Tool Selection ===\n")
    tools = [WEATHER_TOOL, TIME_TOOL, CALC_TOOL]
    cases = [
        ("What's the weather in Paris?", "get_weather"),
        ("What time is it in New York?", "get_time"),
        ("What's 123 * 456?", "calculate"),
    ]
    for prompt, expected in cases:
        try:
            ok, det = test_basic_tool(chat_url, headers, model, timeout,
                                      tools, prompt, expected)
        except Exception as e:
            ok, det = False, str(e)
        log_result(f"multi_{expected}", ok, det)
        print(f"  {'PASS' if ok else 'FAIL'}: {det}")


def test_parallel(chat_url, headers, model, timeout):
    print("\n=== Parallel Tool Calls ===\n")
    tools = [WEATHER_TOOL, TIME_TOOL]
    prompt = "Compare weather and time in Tokyo, London, and New York."
    try:
        resp, calls, dt, _ = chat(chat_url, headers, model,
                                  [{"role": "user", "content": prompt}],
                                  tools=tools, temperature=0.0,
                                  max_tokens=8192, timeout=timeout)
    except Exception as e:
        log_result("parallel_tools", False, str(e))
        print(f"  FAIL: {e}")
        return
    names = [c.get("function", {}).get("name") for c in calls]
    distinct = {n for n in names if n}
    ok = len(calls) >= 2 and len(distinct) >= 2
    log_result("parallel_tools", ok, f"{len(calls)} calls: {names}")
    print(f"  {'PASS' if ok else 'FAIL'}: {len(calls)} calls ({names})")


def test_complex_args(chat_url, headers, model, timeout):
    print("\n=== Complex Argument Parsing ===\n")
    prompt = "Book a flight from San Francisco to Tokyo on 2024-06-15 for 2 passengers."
    try:
        resp, calls, dt, _ = chat(
            chat_url, headers, model,
            [{"role": "user", "content": prompt}],
            tools=[FLIGHT_TOOL], temperature=0.0, max_tokens=8192,
            timeout=timeout)
    except Exception as e:
        log_result("complex_args", False, str(e))
        print(f"  FAIL: {e}")
        return
    if not calls:
        log_result("complex_args", False, "no tool call")
        print(f"  FAIL: no tool call (content: {resp[:100]!r})")
        return
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except Exception as e:
        log_result("complex_args", False, f"bad JSON: {e}")
        print(f"  FAIL: {e}")
        return
    required = {"from", "to", "date"}
    missing = required - set(args.keys())
    ok = not missing
    log_result("complex_args", ok, f"args={args}")
    print(f"  {'PASS' if ok else 'FAIL'}: args={args} "
          f"{'(missing: ' + ', '.join(missing) + ')' if missing else ''}")


def main(args=None):
    ap = build_args("Tool calling tests")
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
    test_basic(chat_url, headers, model, timeout)
    test_tool_chain(chat_url, headers, model, timeout)
    test_multiple_tools(chat_url, headers, model, timeout)
    test_parallel(chat_url, headers, model, timeout)
    test_complex_args(chat_url, headers, model, timeout)
    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
