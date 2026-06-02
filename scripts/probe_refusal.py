#!/usr/bin/env python3
"""Diagnostic probe: does prompt wording rescue the depth-0 collapse?

NOT part of the suite. A one-shot experiment to answer a single question
that the full long-context runs left open: every model scores 0% whenever
the needle is NOT at the end of the context, and they do it by *refusing*
("the document does not contain..."). Is that a weak/ignored instruction,
or a genuine retrieval limit?

Strategy: hold the model and the (short, 4k) context fixed — so the
instruction can't be "lost" in a long prompt — and vary only the prompt
wording across three variants, at depths 0 / 50 / 100:

  baseline  - the current v0.4.2 prompt (anti-refusal line at the top)
  tail      - anti-refusal line + question moved to the END (recency)
  forceful  - a maximally directive, refusal-forbidding instruction

If a variant lifts depth-0/50 at 4k, the collapse is fixable wording.
If nothing helps even at 4k with a forceful prompt, it's a real limit
(or a needle/scoring problem), and no prompt tweak will save it.

Pure stdlib + Ollama HTTP, so it runs without installing the package.

Usage:
  scripts/probe_refusal.py
  scripts/probe_refusal.py --model qwen3.6:35b
  scripts/probe_refusal.py --tokens 4096 --json /tmp/refusal.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_MODEL = "nemotron3:33b"
CHARS_PER_TOKEN = 6.8
ANSWER_BUDGET = 256
SAFETY_MARGIN = 512

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "data" / "long_context" / "long_context_retrieval_v1.json"
DEFAULT_HAYSTACK = REPO_ROOT / "data" / "long_context" / "haystacks" / "darwin_origin_of_species.txt"

# Two stable needles re-used across every depth/variant so the only thing
# that changes is the prompt wording.
NEEDLE_IDS = ["code_7B_MIRA", "location_battery_plant"]

_DIGIT_SEP_RE = re.compile(r"(?<=\d)[ ,\u00a0\u202f'](?=\d)")


# --------------------------------------------------------------------- #
# Console                                                               #
# --------------------------------------------------------------------- #

def hr(title: str) -> None:
    print(f"\n\033[1;36m=== {title} ===\033[0m")


def ok(msg: str) -> None:
    print(f"  \033[1;32m✓ {msg}\033[0m")


def bad(msg: str) -> None:
    print(f"  \033[1;31m✗ {msg}\033[0m")


# --------------------------------------------------------------------- #
# Scoring (mirrors score_niah in long_context.py)                       #
# --------------------------------------------------------------------- #

def score(response: str, expected: str) -> bool:
    def norm(s: str) -> str:
        return _DIGIT_SEP_RE.sub("", " ".join(s.lower().split()))

    return norm(expected) in norm(response)


# --------------------------------------------------------------------- #
# Prompt assembly                                                       #
# --------------------------------------------------------------------- #

def slice_haystack(raw: str, target_chars: int) -> str:
    if len(raw) >= target_chars:
        return raw[:target_chars]
    reps = (target_chars // max(1, len(raw))) + 1
    return (raw * reps)[:target_chars]


def insert_needle(hay: str, needle: str, depth_pct: int) -> str:
    if depth_pct <= 0:
        return f"{needle} {hay}"
    if depth_pct >= 100:
        return f"{hay} {needle}"
    cut = int(len(hay) * depth_pct / 100)
    space = hay.rfind(" ", 0, cut)
    if space <= 0:
        space = cut
    return f"{hay[:space]} {needle} {hay[space:].lstrip()}"


def prompt_baseline(doc: str, q: str, nonce: str) -> str:
    return (
        f"[session {nonce}] You are given a long document. A specific fact "
        "needed to answer the question has been inserted somewhere inside it. "
        "Read carefully, find that fact, and answer using only information "
        "stated in the document. The answer is present in the document, so do "
        "not reply that it is missing.\n\n"
        f"=== DOCUMENT START ===\n{doc}\n=== DOCUMENT END ===\n\n"
        f"Question: {q}\nAnswer:"
    )


def prompt_tail(doc: str, q: str, nonce: str) -> str:
    # Anti-refusal instruction + question moved AFTER the document, where a
    # recency-biased model is most likely to actually read it.
    return (
        f"[session {nonce}] Read the following document carefully.\n\n"
        f"=== DOCUMENT START ===\n{doc}\n=== DOCUMENT END ===\n\n"
        "A specific fact that answers the question below was inserted into the "
        "document above; it is definitely present somewhere in the text. Find "
        "that fact and answer using only that information. Do not reply that it "
        "is missing or not found.\n"
        f"Question: {q}\nAnswer:"
    )


def prompt_forceful(doc: str, q: str, nonce: str) -> str:
    return (
        f"[session {nonce}] You are given a long document containing one "
        "planted fact. The document DEFINITELY contains the exact answer to the "
        "question. Your only job is to locate that fact and quote it verbatim. "
        "You must NEVER answer that the information is absent, not found, or not "
        "in the text — it is there, so search until you find it. Read carefully.\n\n"
        f"=== DOCUMENT START ===\n{doc}\n=== DOCUMENT END ===\n\n"
        f"Question: {q}\nAnswer with only the exact value from the document:"
    )


VARIANTS = {
    "baseline": prompt_baseline,
    "tail": prompt_tail,
    "forceful": prompt_forceful,
}


# --------------------------------------------------------------------- #
# Ollama                                                                #
# --------------------------------------------------------------------- #

def generate(endpoint: str, model: str, prompt: str, num_ctx: int, timeout: float) -> str:
    options = {"temperature": 0.0, "seed": 42, "num_predict": ANSWER_BUDGET, "num_ctx": num_ctx}
    payload = {"model": model, "prompt": prompt, "stream": False, "think": False, "options": options}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response") or "").strip()


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Probe whether prompt wording fixes the depth-0 refusal collapse")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--haystack", default=str(DEFAULT_HAYSTACK))
    ap.add_argument("--tokens", type=int, default=4096, help="context length to probe (default 4096)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    try:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"could not read fixture {FIXTURE}: {exc}", file=sys.stderr)
        return 2
    by_id = {n["id"]: n for n in fixture["needles"]}
    needles = [by_id[i] for i in NEEDLE_IDS if i in by_id]

    hay_path = Path(args.haystack)
    if not hay_path.exists():
        print(f"haystack not found: {hay_path}\nrun scripts/fetch_haystacks.sh first", file=sys.stderr)
        return 2
    raw = hay_path.read_text(encoding="utf-8", errors="replace")

    target_tokens = max(256, args.tokens - ANSWER_BUDGET - SAFETY_MARGIN)
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    base_doc = slice_haystack(raw, target_chars)

    hr(f"Refusal probe — {args.model} @ {args.tokens} tokens")
    print(f"  haystack: {hay_path.name} ({len(raw)} chars, sliced to {len(base_doc)})")
    print(f"  needles : {', '.join(NEEDLE_IDS)}")

    # grid[variant][depth] = (passes, total)
    grid: dict[str, dict[int, list[int]]] = {v: {0: [0, 0], 50: [0, 0], 100: [0, 0]} for v in VARIANTS}
    rows: list[dict[str, Any]] = []

    for variant, builder in VARIANTS.items():
        hr(f"variant: {variant}")
        for depth in (0, 50, 100):
            for needle in needles:
                doc = insert_needle(base_doc, needle["text"], depth)
                nonce = f"{variant}-{depth}-{needle['id']}-{int(time.time()*1000)%100000}"
                prompt = builder(doc, needle["question"], nonce)
                try:
                    out = generate(args.endpoint, args.model, prompt, args.tokens, args.timeout)
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    bad(f"d{depth:<3} {needle['id']:<22} ERROR: {exc}")
                    rows.append({"variant": variant, "depth": depth, "needle": needle["id"], "error": str(exc)})
                    continue
                passed = score(out, needle["expected_substring"])
                grid[variant][depth][0] += int(passed)
                grid[variant][depth][1] += 1
                mark = "✓" if passed else "✗"
                preview = out.replace("\n", " ")[:64]
                print(f"  d{depth:<3} {needle['id']:<22} {mark} exp={needle['expected_substring']!r:18} -> {preview!r}")
                rows.append(
                    {
                        "variant": variant,
                        "depth": depth,
                        "needle": needle["id"],
                        "expected": needle["expected_substring"],
                        "passed": passed,
                        "output": out[:300],
                    }
                )

    hr("Summary — pass rate by variant × depth")
    print(f"  {'variant':<10}  d0     d50    d100")
    for variant in VARIANTS:
        cells = []
        for depth in (0, 50, 100):
            p, t = grid[variant][depth]
            cells.append(f"{(p/t*100 if t else 0):>4.0f}%")
        print(f"  {variant:<10}  {cells[0]}  {cells[1]}  {cells[2]}")

    d0_any = any(grid[v][0][0] for v in VARIANTS)
    hr("Verdict")
    if d0_any:
        ok("at least one variant retrieved a depth-0 needle at this length — "
           "wording matters; the collapse is (partly) a refusal/prompt issue")
    else:
        bad("no variant retrieved ANY depth-0 needle even at this short length "
            "with a forceful prompt — this is a genuine retrieval limit, not wording")

    if args.json:
        Path(args.json).write_text(json.dumps({"model": args.model, "tokens": args.tokens, "grid": grid, "rows": rows}, indent=2))
        ok(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
