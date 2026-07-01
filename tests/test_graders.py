#!/usr/bin/env python3
"""
Unit tests for grading functions and helpers.

Tests logic that doesn't need a running server:
  - grade_reasoning   (accept/reject precedence)
  - extract_code      (markdown / bare code extraction)
  - run_code          (subprocess exec of model-emitted code)
  - grade_coding      (eq + set modes, missing-fn, exec error)
  - detect_refusal    (phrase matching)
  - _count_sentences  (code-block stripping, sentence-end counting)
  - _try_json         (fenced / embedded / invalid JSON)
  - INSTRUCTION_CASES lambdas (a representative sample)

Run with:
    python -m pytest tests/test_graders.py
or standalone:
    python tests/test_graders.py
"""

from __future__ import annotations

import sys

from utils import extract_code, grade_coding, run_code
from test_programming import CODING_CASES
from test_reasoning import (INSTRUCTION_CASES, REASONING_CASES,
                            _count_sentences, _try_json, detect_refusal,
                            grade_reasoning)


def _rcase(cid: str):
    for c in REASONING_CASES:
        if c[0] == cid:
            return c
    raise KeyError(cid)


def _ccase(name):
    for c in CODING_CASES:
        if c[0] == name:
            return c
    raise KeyError(name)


def _icase(name):
    for c in INSTRUCTION_CASES:
        if c[0] == name:
            return c
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# grade_reasoning
# --------------------------------------------------------------------------- #

def test_reasoning_accept_match():
    case = _rcase("R1-batball")
    ok, _ = grade_reasoning("The ball costs $0.05.", case[2], case[3])
    assert ok

def test_reasoning_reject_when_no_accept():
    case = _rcase("R1-batball")
    ok, detail = grade_reasoning("The ball is 0.10.", case[2], case[3])
    assert not ok
    assert "0.10" in detail

def test_reasoning_accept_overrides_co_present_reject():
    case = _rcase("R1-batball")
    ok, _ = grade_reasoning("It's not 0.10, the ball costs 0.05.", case[2], case[3])
    assert ok

def test_reasoning_no_match_fails():
    case = _rcase("R1-batball")
    ok, _ = grade_reasoning("I have no idea.", case[2], case[3])
    assert not ok

def test_reasoning_substring_collision_with_coined_digits():
    case = _rcase("R3-12coins")
    ok, _ = grade_reasoning("Weighing #1 splits them, then #2, then #3: done.",
                            case[2], case[3])
    assert ok

def test_reasoning_reject_fires_when_only_wrong_digits():
    case = _rcase("R3-12coins")
    ok, _ = grade_reasoning("Try 1 and 2 weighings first.", case[2], case[3])
    assert not ok

def test_reasoning_case_insensitive():
    case = _rcase("R5-knights")
    ok, _ = grade_reasoning("A=Knave, B=Knight obviously.", case[2], case[3])
    assert ok

def test_reasoning_documented_substring_brittleness():
    case = _rcase("R6-tax")
    ok, _ = grade_reasoning("The total is 16.80 dollars.", case[2], case[3])
    assert ok


# --------------------------------------------------------------------------- #
# extract_code
# --------------------------------------------------------------------------- #

def test_extract_code_python_fence():
    src = extract_code("Here:\n```python\ndef solve(s): return 1\n```\nDone.")
    assert "def solve(s): return 1" in src

def test_extract_code_bare_fence():
    src = extract_code("```\ndef solve(s): return 1\n```")
    assert "def solve(s): return 1" in src

def test_extract_code_no_fence_with_def():
    text = "Sure.\ndef solve(s):\n    return 1\n"
    assert extract_code(text).startswith("def solve")

def test_extract_code_no_def_returns_as_is():
    text = "just some prose with no code"
    assert extract_code(text) == text

def test_extract_code_first_fence_wins():
    text = "```python\ndef f(): return 1\n```\n```\ndef g(): return 2\n```"
    assert "def f" in extract_code(text)
    assert "def g" not in extract_code(text)


# --------------------------------------------------------------------------- #
# run_code (subprocess)
# --------------------------------------------------------------------------- #

def test_run_code_correct_eq():
    code = "def solve(s): return len(s)\n"
    res = run_code(code, "solve", [("hi",), ("hello",)])
    assert "error" not in res
    assert res["results"] == [
        {"ok": True, "got": 2},
        {"ok": True, "got": 5},
    ]

def test_run_code_wrong_value():
    code = "def solve(s): return 0\n"
    res = run_code(code, "solve", [("hi",)])
    assert res["results"][0] == {"ok": True, "got": 0}

def test_run_code_missing_function():
    code = "def other(s): return s\n"
    res = run_code(code, "solve", [])
    assert "error" in res
    assert "not defined" in res["error"]

def test_run_code_exec_error():
    code = "def solve(s)\n    return s\n"
    res = run_code(code, "solve", [(("hi",), 0)])
    assert "error" in res

def test_run_code_runtime_exception_per_case():
    code = "def solve(s):\n    if s == 'x': raise ValueError('nope')\n    return s\n"
    res = run_code(code, "solve", [("x",), ("y",)])
    assert res["results"][0]["ok"] is False
    assert "nope" in res["results"][0]["err"]
    assert res["results"][1] == {"ok": True, "got": "y"}

def test_run_code_does_not_collide_on_runner_identifiers():
    code = (
        "def _ARGS(): return 'sentinel'\n"
        "def solve(_FUNC=None):\n"
        "    return ('args', _ARGS(), _FUNC)\n"
    )
    res = run_code(code, "solve", [(None,)])
    assert "error" not in res
    assert res["results"][0]["ok"] is True
    assert res["results"][0]["got"] == ["args", "sentinel", None]

def test_run_code_timeout():
    code = "def solve(s):\n    while True:\n        pass\n"
    res = run_code(code, "solve", [("",)], timeout=2)
    assert res == {"error": "timeout"}


# --------------------------------------------------------------------------- #
# grade_coding (end-to-end on synthetic responses)
# --------------------------------------------------------------------------- #

def test_grade_coding_all_pass():
    case = _ccase("C1-lsus")
    cid, prompt, func, tests, mode = case
    resp = ("```python\n"
            "def solve(s):\n"
            "    seen={}\n    start=0\n    best=0\n"
            "    for i,c in enumerate(s):\n"
            "        if c in seen and seen[c]>=start:\n"
            "            start=seen[c]+1\n"
            "        seen[c]=i\n"
            "        best=max(best,i-start+1)\n"
            "    return best\n```")
    ok, detail = grade_coding(resp, func, tests, mode)
    assert ok, detail

def test_grade_coding_missing_function():
    case = _ccase("C1-lsus")
    cid, prompt, func, tests, mode = case
    ok, detail = grade_coding("no code here, just words", func, tests, mode)
    assert not ok
    assert "no function" in detail or "not defined" in detail

def test_grade_coding_set_mode():
    case = _ccase("C3-twosum")
    cid, prompt, func, tests, mode = case
    resp = ("```python\ndef solve(nums, target):\n"
            "    seen={}\n"
            "    for i,n in enumerate(nums):\n"
            "        if target-n in seen:\n"
            "            return [seen[target-n],i]\n"
            "        seen[n]=i\n```")
    ok, detail = grade_coding(resp, func, tests, mode)
    assert ok, detail

def test_grade_coding_wrong_value_records_detail():
    case = _ccase("C1-lsus")
    cid, prompt, func, tests, mode = case
    resp = "```python\ndef solve(s):\n    return 999\n```"
    ok, detail = grade_coding(resp, func, tests, mode)
    assert not ok
    assert "999" in detail


# --------------------------------------------------------------------------- #
# detect_refusal
# --------------------------------------------------------------------------- #

def test_refusal_empty():
    assert detect_refusal("") is False

def test_refusal_classic_prefix():
    assert detect_refusal("I can't help with that.") is True

def test_refusal_apologetic():
    assert detect_refusal("I'm sorry, but I can't do that.") is True

def test_refusal_compliant_response():
    assert detect_refusal("Sure, here's how SQL injection works...") is False


# --------------------------------------------------------------------------- #
# _count_sentences
# --------------------------------------------------------------------------- #

def test_count_sentences_basic():
    assert _count_sentences("Hello world. Hi there!") == 2

def test_count_sentences_strips_code_blocks():
    txt = "See this. ```py\nx=1\n``` Done."
    assert _count_sentences(txt) == 2

def test_count_sentences_no_ends():
    assert _count_sentences("just running text") == 0

def test_count_sentences_multi_punct():
    assert _count_sentences("Wait... really?! Yes.") == 3


# --------------------------------------------------------------------------- #
# _try_json
# --------------------------------------------------------------------------- #

def test_try_json_plain():
    assert _try_json('{"name": "a", "age": 1}') == {"name": "a", "age": 1}

def test_try_json_fenced():
    assert _try_json("```json\n{\"x\": 1}\n```") == {"x": 1}

def test_try_json_embedded():
    assert _try_json('prefix {"x": 1} suffix') == {"x": 1}

def test_try_json_invalid():
    assert _try_json("not json at all") is None


# --------------------------------------------------------------------------- #
# INSTRUCTION_CASES lambdas (representative)
# --------------------------------------------------------------------------- #

def test_instruction_three_sentences_pass():
    grader = _icase("I1-3sentences")[2]
    ok, _ = grader("The ocean is vast. It is deep. Waves crash.")
    assert ok

def test_instruction_three_sentences_fail():
    grader = _icase("I1-3sentences")[2]
    ok, _ = grader("Only one sentence here.")
    assert not ok

def test_instruction_yesno_pass():
    grader = _icase("I4-yesno")[2]
    ok, _ = grader("YES")
    assert ok

def test_instruction_yesno_fail():
    grader = _icase("I4-yesno")[2]
    ok, _ = grader("Yes, the symbol is Au.")
    assert not ok

def test_instruction_lipogram_pass():
    grader = _icase("I5-lipogram")[2]
    ok, detail = grader("A warm orb hangs aloft, casting rays of light.")
    assert ok, detail

def test_instruction_lipogram_fail():
    grader = _icase("I5-lipogram")[2]
    ok, _ = grader("The sun is yellow.")
    assert not ok

def test_instruction_wordcount_exact():
    grader = _icase("I8-wordcount")[2]
    ok, _ = grader("one two three four five six seven eight nine ten "
                   "eleven twelve thirteen fourteen fifteen sixteen "
                   "seventeen eighteen nineteen twenty")
    assert ok

def test_instruction_negation_clean():
    grader = _icase("I7-negation")[2]
    ok, _ = grader("A bicycle has a frame and two handlebars.")
    assert ok

def test_instruction_negation_forbidden_word():
    grader = _icase("I7-negation")[2]
    ok, _ = grader("A bicycle has a chain and two pedals.")
    assert not ok


# --------------------------------------------------------------------------- #
# Standalone runner (no pytest required)
# --------------------------------------------------------------------------- #

def _all_test_fns():
    return [(n, getattr(sys.modules[__name__], n))
            for n in dir(sys.modules[__name__])
            if n.startswith("test_") and callable(getattr(sys.modules[__name__], n))]


if __name__ == "__main__":
    failures = 0
    for name, fn in _all_test_fns():
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(_all_test_fns()) - failures}/{len(_all_test_fns())} passed")
    sys.exit(1 if failures else 0)
