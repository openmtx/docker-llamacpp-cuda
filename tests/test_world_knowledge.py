import sys

import requests

from config import build_args, resolve
from utils import RESULTS, chat, log_result, print_summary, reset_results


QUESTIONS = [
    # Obscure history
    ("H1-cymon",
     "What was the name of the final flagship of the Welsh pirate "
     "Bartholomew Roberts (Black Bart) — the ship he commanded when he was "
     "killed in battle in 1722? Answer with just the ship name.",
     ["royal fortune", "ranger"],
     ["queen anne"]),
    ("H2-muskets",
     "In the American Civil War, what was the primary infantry firearm of "
     "the Union Army? Just the model name.",
     ["springfield model 1861", "springfield 1861",
      "model 1861", "springfield rifle"]),
    ("H3-napoleon",
     "What was Napoleon Bonaparte's horse's name? Just the name.",
     ["marengo", "vizir", "le cheval"]),
    ("H4-pharaoh",
     "Which female pharaoh was known for dressing in men's clothing and "
     "wearing a false beard? Just the name.",
     ["hatshepsut"]),

    # Obscure science
    ("S1-element",
     "What is the only naturally occurring substance harder than diamond? "
     "Just the name.",
     ["lonsdaleite", "wurtzite boron nitride",
      "boron nitride"]),
    ("S2-moth",
     "What is the name of the insect species that was once the most abundant "
     "in North America but is now virtually extinct? Just the common or "
     "scientific species name.",
     ["rocky mountain locust",
      "melanoplus spretus"]),
    ("S3-volcano",
     "What is the name of the volcano in Indonesia whose 1815 eruption "
     "caused the 'Year Without a Summer'? Just the name.",
     ["mount tambora", "tambora"]),

    # Obscure geography
    ("G1-capital",
     "What is the capital of Burkina Faso? Just the city name.",
     ["ouagadougou"]),
    ("G2-river",
     "Which European river flows through the most countries (10)? "
     "Just the name.",
     ["danube"]),
    ("G3-country",
     "Which is the only country whose name begins with 'Q'? "
     "Just the country name.",
     ["qatar"]),

    # Obscure pop culture / arts
    ("P1-painting",
     "What was the painter Caravaggio's real first name? "
     "Just the name.",
     ["michelangelo merisi", "michelangelo",
     "merisi"]),
    ("P2-programming",
     "What programming language was created by Yukihiro Matsumoto? "
     "Just the name.",
     ["ruby"]),
    ("P3-game",
     "What was the first home console video game to use a battery-backed "
     "save system, letting players keep their progress? Just the game name.",
     ["the legend of zelda",
      "zelda"]),
]


def grade_knowledge(response, accept, reject):
    low = response.lower().strip(".!,\"'")
    for r in reject or []:
        if r.lower() in low:
            return False, f"matched reject {r!r}"
    for a in accept:
        if a.lower() in low:
            return True, f"matched {a!r}"
    return False, f"no match; got={response[:80]!r}"


def test_world_knowledge(chat_url, headers, model, timeout, verbose=False):
    print("\n=== World Knowledge ===\n")
    for qid, prompt, accept, *rest in QUESTIONS:
        reject = rest[0] if rest else []
        print(f"  {qid}: ", end="", flush=True)
        try:
            resp, _, dt, usage = chat(chat_url, headers, model,
                                      [{"role": "user", "content": prompt}],
                                      temperature=0.0, max_tokens=8192,
                                      timeout=timeout)
        except Exception as e:
            print(f"ERROR: {e}")
            log_result(qid, False, str(e))
            continue
        ok, detail = grade_knowledge(resp, accept, reject)
        tag = "PASS" if ok else "FAIL"
        print(f"{tag}  {detail}")
        log_result(qid, ok, detail)
        if not ok:
            if verbose:
                print(f"        Response: {resp!r}")
            else:
                print(f"        Response: {resp[:100]!r}")


def main(args=None):
    ap = build_args("World knowledge tests")
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
    test_world_knowledge(chat_url, headers, model, timeout, verbose)
    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
