#!/usr/bin/env python3
"""Diagnostic probe for the long_context_retrieval suite.

This is NOT part of the suite — it is a one-shot reconnaissance tool you
run against a *real* Ollama on the Spark before the long-context runner
(layer 2) is written. It answers the questions that fakes cannot:

  A. Connectivity + which models are loaded, and each model's claimed
     context length (via /api/show).
  B. Tokenization ground truth: is `prompt_eval_count` reliable, and
     what is the chars-per-token ratio per model? (drives how we
     truncate a haystack to an exact token target).
  C. Prefill-duration honesty + the cache-hit trap: does Ollama report
     a non-zero prompt_eval_duration, and does it drop to ~0 when the
     same long prompt is re-sent?
  D. num_ctx behaviour + long-context load / OOM / runtime: does the
     prompt actually load at the requested length (only when we set
     options.num_ctx!), how long does prefill take, and what does an
     OOM look like?
  E. GPU telemetry availability: what does nvidia-smi expose right now
     (the in-repo telemetry collector is still a stub).

Nothing is written to the repo. Results print to the console and,
optionally, to a JSON file via --json.

Usage:
  scripts/probe_long_context.py                       # auto-detect models
  scripts/probe_long_context.py --models qwen3.6:35b
  scripts/probe_long_context.py --haystack data/long_context/haystacks/melville_moby_dick.txt
  scripts/probe_long_context.py --max-context 65536 --json /tmp/probe.json

Only the standard library is used, so this runs anywhere the Spark can
reach the Ollama endpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_ENDPOINT = "http://localhost:11434"

# A deterministic filler sentence used when no --haystack is supplied.
_FILLER_SENTENCE = (
    "The quarterly logistics review noted that throughput across the "
    "northern corridor remained steady while ambient conditions varied. "
)


# --------------------------------------------------------------------- #
# Console helpers                                                        #
# --------------------------------------------------------------------- #

def hr(title: str) -> None:
    print(f"\n\033[1;36m=== {title} ===\033[0m")


def info(msg: str) -> None:
    print(f"  {msg}")


def warn(msg: str) -> None:
    print(f"  \033[1;33m! {msg}\033[0m")


def ok(msg: str) -> None:
    print(f"  \033[1;32m✓ {msg}\033[0m")


def bad(msg: str) -> None:
    print(f"  \033[1;31m✗ {msg}\033[0m")


# --------------------------------------------------------------------- #
# HTTP                                                                   #
# --------------------------------------------------------------------- #

def _post(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = endpoint.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(endpoint: str, path: str, timeout: float) -> dict[str, Any]:
    url = endpoint.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def classify_error(exc: Exception) -> str:
    """Turn an exception into a short, stable label for the report."""
    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        return f"HTTP {exc.code}: {detail.strip()}"
    if isinstance(exc, urllib.error.URLError):
        return f"URLError: {exc.reason}"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return f"{type(exc).__name__}: {exc}"


def generate(
    endpoint: str,
    model: str,
    prompt: str,
    *,
    num_ctx: int | None,
    num_predict: int,
    timeout: float,
) -> dict[str, Any]:
    """Single non-streaming generate call. Returns parsed Ollama JSON."""
    options: dict[str, Any] = {"temperature": 0.0, "seed": 42, "num_predict": num_predict}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    payload = {"model": model, "prompt": prompt, "stream": False, "think": False, "options": options}
    return _post(endpoint, "/api/generate", payload, timeout=timeout)


# --------------------------------------------------------------------- #
# Filler                                                                 #
# --------------------------------------------------------------------- #

def make_filler(target_chars: int, haystack: str | None) -> str:
    if haystack:
        if len(haystack) >= target_chars:
            return haystack[:target_chars]
        reps = (target_chars // max(1, len(haystack))) + 1
        return (haystack * reps)[:target_chars]
    reps = (target_chars // len(_FILLER_SENTENCE)) + 1
    return (_FILLER_SENTENCE * reps)[:target_chars]


# --------------------------------------------------------------------- #
# Probes                                                                 #
# --------------------------------------------------------------------- #

def probe_models(endpoint: str, timeout: float) -> list[dict[str, Any]]:
    hr("Probe A — connectivity & models")
    try:
        tags = _get(endpoint, "/api/tags", timeout=timeout)
    except Exception as exc:
        bad(f"cannot reach Ollama at {endpoint}: {classify_error(exc)}")
        return []
    models = tags.get("models") or []
    if not models:
        warn("Ollama reachable but no models are pulled")
        return []
    out: list[dict[str, Any]] = []
    for m in models:
        name = m.get("name") or m.get("model") or ""
        ctx = None
        try:
            show = _post(endpoint, "/api/show", {"model": name}, timeout=timeout)
            for k, v in (show.get("model_info") or {}).items():
                if k.endswith(".context_length"):
                    ctx = int(v)
                    break
        except Exception:
            pass
        out.append({"name": name, "context_length": ctx})
        ok(f"{name}  (claimed context: {ctx if ctx else 'unknown'})")
    return out


def probe_tokenization(
    endpoint: str, model: str, haystack: str | None, timeout: float
) -> dict[str, Any]:
    hr(f"Probe B — tokenization ground truth [{model}]")
    samples = []
    for char_len in (2000, 8000, 32000):
        prompt = make_filler(char_len, haystack)
        try:
            data = generate(
                endpoint, model, prompt, num_ctx=32768, num_predict=1, timeout=timeout
            )
            count = int(data.get("prompt_eval_count") or 0)
            ratio = (char_len / count) if count else 0.0
            samples.append({"chars": char_len, "prompt_eval_count": count, "chars_per_token": ratio})
            info(f"{char_len:>6} chars -> prompt_eval_count={count:<6} ({ratio:.2f} chars/token)")
        except Exception as exc:
            bad(f"{char_len} chars failed: {classify_error(exc)}")
    ratios = [s["chars_per_token"] for s in samples if s["chars_per_token"]]
    avg = sum(ratios) / len(ratios) if ratios else 0.0
    if avg:
        ok(f"mean chars/token ≈ {avg:.2f}  (use to size haystack truncation)")
    else:
        warn("could not derive a chars/token ratio")
    return {"samples": samples, "mean_chars_per_token": avg}


def probe_prefill_cache(
    endpoint: str, model: str, haystack: str | None, ratio: float, timeout: float
) -> dict[str, Any]:
    hr(f"Probe C — prefill duration & cache-hit trap [{model}]")
    target_tokens = 4000
    chars = int(target_tokens * (ratio or 4.0))
    prompt = make_filler(chars, haystack)
    runs = []
    for i in (1, 2):
        try:
            data = generate(
                endpoint, model, prompt, num_ctx=8192, num_predict=4, timeout=timeout
            )
            pe_count = int(data.get("prompt_eval_count") or 0)
            pe_ns = int(data.get("prompt_eval_duration") or 0)
            pe_s = pe_ns / 1e9
            runs.append({"attempt": i, "prompt_eval_count": pe_count, "prefill_s": pe_s})
            info(f"attempt {i}: prompt_eval_count={pe_count}, prefill={pe_s:.3f}s")
        except Exception as exc:
            bad(f"attempt {i} failed: {classify_error(exc)}")
            runs.append({"attempt": i, "error": classify_error(exc)})
    if len(runs) == 2 and "prefill_s" in runs[0] and "prefill_s" in runs[1]:
        first, second = runs[0]["prefill_s"], runs[1]["prefill_s"]
        if first > 0 and second < first * 0.2:
            warn(
                f"cache-hit detected: prefill dropped {first:.3f}s -> {second:.3f}s. "
                "Layer 2 must vary prompts or disable caching to measure prefill honestly."
            )
        elif first > 0:
            ok("prefill duration looks stable across repeats (no obvious cache zeroing)")
    return {"runs": runs}


def probe_long_context(
    endpoint: str,
    model: str,
    claimed_ctx: int | None,
    haystack: str | None,
    ratio: float,
    max_context: int,
    timeout: float,
) -> dict[str, Any]:
    hr(f"Probe D — num_ctx, long-context load / OOM / runtime [{model}]")
    ratio = ratio or 4.0
    results = []

    # First: probe whether Ollama's *default* context window truncates a
    # long prompt when options.num_ctx is NOT set. Test high enough to
    # actually exceed the default (modern Ollama auto-sizes, so a 16k
    # prompt may pass uncapped — the cap shows up further out).
    demo_tokens = min(max_context, 65536)
    demo_prompt = make_filler(int(demo_tokens * ratio), haystack)
    try:
        no_ctx = generate(endpoint, model, demo_prompt, num_ctx=None, num_predict=2, timeout=timeout)
        loaded = int(no_ctx.get("prompt_eval_count") or 0)
        capped = loaded < demo_tokens * 0.9
        results.append(
            {"length": demo_tokens, "mode": "no_num_ctx", "prompt_eval_count": loaded, "capped": capped}
        )
        if capped:
            warn(
                f"WITHOUT options.num_ctx a ~{demo_tokens}-token prompt loaded only "
                f"{loaded} tokens — Ollama's default window truncated it. "
                "Layer 2 MUST set options.num_ctx per request."
            )
        else:
            ok(
                f"default window covered ~{demo_tokens} tokens (loaded {loaded}); "
                "still set num_ctx explicitly in layer 2 to be safe across Ollama versions"
            )
    except Exception as exc:
        info(f"no-num_ctx demo errored: {classify_error(exc)}")

    lengths = [n for n in (4096, 16384, 65536, 131072) if n <= max_context]
    for length in lengths:
        if claimed_ctx and length > claimed_ctx:
            info(f"{length}: skipped (exceeds claimed context {claimed_ctx})")
            results.append({"length": length, "status": "skipped_unsupported"})
            continue
        # Size slightly under target so the needle/question fit too.
        chars = int(length * ratio * 0.92)
        prompt = make_filler(chars, haystack)
        started = time.perf_counter()
        try:
            data = generate(
                endpoint, model, prompt, num_ctx=length, num_predict=8, timeout=timeout
            )
            wall = time.perf_counter() - started
            pe_count = int(data.get("prompt_eval_count") or 0)
            pe_s = int(data.get("prompt_eval_duration") or 0) / 1e9
            tps = (pe_count / pe_s) if pe_s else 0.0
            ok(
                f"{length}: loaded {pe_count} tok, prefill {pe_s:.1f}s "
                f"({tps:.0f} tok/s), wall {wall:.1f}s"
            )
            results.append(
                {
                    "length": length,
                    "status": "ok",
                    "prompt_eval_count": pe_count,
                    "prefill_s": pe_s,
                    "prefill_tps": tps,
                    "wall_s": wall,
                }
            )
        except Exception as exc:
            wall = time.perf_counter() - started
            label = classify_error(exc)
            bad(f"{length}: {label}  (after {wall:.1f}s)")
            results.append({"length": length, "status": "error", "error": label, "wall_s": wall})
    return {"results": results}


def probe_telemetry() -> dict[str, Any]:
    hr("Probe E — GPU telemetry availability")
    info("note: the in-repo telemetry collector is a stub; real memory/temp must come from here")
    query = "memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        bad("nvidia-smi not found on PATH")
        return {"nvidia_smi": None}
    except Exception as exc:
        bad(f"nvidia-smi failed: {exc}")
        return {"nvidia_smi": None}
    if out.returncode != 0:
        bad(f"nvidia-smi exit {out.returncode}: {out.stderr.strip()[:200]}")
        return {"nvidia_smi": None}
    line = out.stdout.strip()
    ok(f"nvidia-smi OK -> [{query}] = {line}")
    return {"nvidia_smi": line, "query": query}


# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Long-context reconnaissance probe for Ollama on Spark")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--models", default="", help="comma-separated tags; default = auto-detect all")
    ap.add_argument("--haystack", default="", help="path to a real haystack .txt (optional)")
    ap.add_argument("--max-context", type=int, default=131072)
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout (s)")
    ap.add_argument("--json", default="", help="write full results to this JSON path")
    args = ap.parse_args()

    haystack_text: str | None = None
    if args.haystack:
        try:
            with open(args.haystack, encoding="utf-8", errors="replace") as fh:
                haystack_text = fh.read()
            ok_chars = len(haystack_text)
            print(f"using haystack {args.haystack} ({ok_chars} chars)")
        except OSError as exc:
            print(f"could not read haystack: {exc}", file=sys.stderr)
            return 2

    report: dict[str, Any] = {"endpoint": args.endpoint, "probes": {}}

    detected = probe_models(args.endpoint, args.timeout)
    report["probes"]["models"] = detected
    if not detected:
        if args.json:
            _write_json(args.json, report)
        return 1

    if args.models.strip():
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        ctx_by_name = {d["name"]: d.get("context_length") for d in detected}
        targets = [(w, ctx_by_name.get(w)) for w in wanted]
    else:
        targets = [(d["name"], d.get("context_length")) for d in detected]

    report["probes"]["per_model"] = {}
    for name, claimed_ctx in targets:
        per: dict[str, Any] = {}
        tok = probe_tokenization(args.endpoint, name, haystack_text, args.timeout)
        per["tokenization"] = tok
        ratio = tok.get("mean_chars_per_token") or 0.0
        per["prefill_cache"] = probe_prefill_cache(
            args.endpoint, name, haystack_text, ratio, args.timeout
        )
        per["long_context"] = probe_long_context(
            args.endpoint, name, claimed_ctx, haystack_text, ratio, args.max_context, args.timeout
        )
        report["probes"]["per_model"][name] = per

    report["probes"]["telemetry"] = probe_telemetry()

    hr("Summary — decisions this informs")
    info("B -> truncation strategy (chars/token ratio; is prompt_eval_count trustworthy?)")
    info("C -> whether prompts must vary per cell to avoid cache-zeroed prefill")
    info("D -> required num_ctx; real OOM signature; whether 131k fits & timeout headroom")
    info("E -> where peak memory / temperature actually come from (telemetry is a stub today)")

    if args.json:
        _write_json(args.json, report)
        ok(f"wrote {args.json}")
    return 0


def _write_json(path: str, report: dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except OSError as exc:
        print(f"could not write JSON: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
