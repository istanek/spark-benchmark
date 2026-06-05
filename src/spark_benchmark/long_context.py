"""Long-context retrieval suite (single-needle NIAH).

This module is the v0.4.0 implementation of `long_context_retrieval`,
following the design in `docs/long-context-spec.md`. It is platform-
agnostic; the v1 configuration targets Spark only.

Layering note: this first slice ships the **fixture schema, loader, and
the pure deterministic plumbing** (needle/haystack selection, substring
scoring). The actual run loop — per-model tokenization, haystack
truncation, needle insertion, backend calls, three-state cell logic —
lands in the follow-up slice. Everything here is side-effect-free and
unit-testable without a backend.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, model_validator

from spark_benchmark.models import (
    BackendConfig,
    GenerationResult,
    ModelConfig,
    SamplingConfig,
)
from spark_benchmark.results_bundle import write_json, write_result


class Needle(BaseModel):
    """A single fact hidden in the haystack, plus how to query/score it."""

    id: str
    category: str
    text: str
    question: str
    expected_substring: str

    @model_validator(mode="after")
    def _validate(self) -> "Needle":
        if not self.id.strip():
            raise ValueError("needle id must not be empty")
        if not self.expected_substring.strip():
            raise ValueError(f"needle {self.id!r} has empty expected_substring")
        if self.expected_substring not in self.text:
            raise ValueError(
                f"needle {self.id!r}: expected_substring is not contained in its own text "
                "(the scorer would never be able to pass)"
            )
        return self


class HaystackSpec(BaseModel):
    """Provenance + on-disk location for one filler corpus.

    The bytes themselves are fetched on demand (see
    ``scripts/fetch_haystacks.sh``) and are git-ignored; only this
    metadata ships in the fixture so the repo stays lean.
    """

    source_url: str
    license: str
    text_file: str
    sha256: str | None = None


class TestMatrix(BaseModel):
    context_lengths_tokens: list[int]
    depth_percentages: list[int]
    needles_per_cell: int
    haystacks: list[str]

    @model_validator(mode="after")
    def _validate(self) -> "TestMatrix":
        for field_name in ("context_lengths_tokens", "depth_percentages", "haystacks"):
            if not getattr(self, field_name):
                raise ValueError(f"test_matrix.{field_name} must not be empty")
        if self.needles_per_cell < 1:
            raise ValueError("test_matrix.needles_per_cell must be >= 1")
        for depth in self.depth_percentages:
            if not 0 <= depth <= 100:
                raise ValueError(f"depth_percentages must be in [0, 100], got {depth}")
        return self


class LongContextFixture(BaseModel):
    name: str
    category: str = "reliability"
    version: str = "0.4.0"
    description: str = ""
    notes: list[str] = Field(default_factory=list)
    haystacks: dict[str, HaystackSpec]
    needles: list[Needle]
    test_matrix: TestMatrix
    # Named alternative grids (e.g. "fast") selectable at run time. The
    # top-level test_matrix is the default ("full") profile.
    profiles: dict[str, TestMatrix] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "LongContextFixture":
        if not self.haystacks:
            raise ValueError("fixture must define at least one haystack")
        # Validate the default matrix and every named profile against the
        # same invariants (enough needles, haystacks defined).
        for label, matrix in [("test_matrix", self.test_matrix), *self.profiles.items()]:
            if len(self.needles) < matrix.needles_per_cell:
                raise ValueError(
                    f"fixture has {len(self.needles)} needles but {label}.needles_per_cell="
                    f"{matrix.needles_per_cell}; a cell could not be filled without repeats"
                )
            missing = [h for h in matrix.haystacks if h not in self.haystacks]
            if missing:
                raise ValueError(f"{label} references undefined haystacks: {missing}")
        return self


def load_long_context_fixture(path: Path | str) -> LongContextFixture:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LongContextFixture.model_validate(payload)


# Suite name suffix → profile. The full grid is the default; a "_fast"
# suffix selects the lighter, range-covering preview grid.
FULL_PROFILE = "full"
FAST_PROFILE = "fast"


def profile_for_suite_name(suite_name: str) -> str:
    """Map a dispatched suite name to a fixture profile.

    ``long_context_retrieval`` / ``..._v1`` → full grid;
    ``long_context_retrieval_fast`` → the fast preview grid.
    """
    return FAST_PROFILE if suite_name.endswith("_fast") else FULL_PROFILE


def resolve_profile_matrix(fixture: LongContextFixture, profile: str | None) -> TestMatrix:
    """Return the TestMatrix for ``profile`` (default/``full`` → top-level)."""
    if not profile or profile == FULL_PROFILE:
        return fixture.test_matrix
    matrix = fixture.profiles.get(profile)
    if matrix is None:
        known = ", ".join(sorted([FULL_PROFILE, *fixture.profiles])) or FULL_PROFILE
        raise ValueError(f"unknown long-context profile {profile!r}; known: {known}")
    return matrix


def _stable_hash(*parts: Any) -> int:
    """Process-stable hash (Python's builtin ``hash`` is salted per run).

    Used to make needle/haystack selection reproducible across runs and
    machines: same inputs always yield the same task plan.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def select_needle_index(length: int, depth: int, repetition: int, n_needles: int) -> int:
    """Deterministically pick which needle a (length, depth, rep) cell uses."""
    if n_needles < 1:
        raise ValueError("n_needles must be >= 1")
    return _stable_hash(length, depth, repetition) % n_needles


def select_haystack(length: int, depth: int, haystacks: list[str]) -> str:
    """Deterministically rotate which haystack a (length, depth) cell uses."""
    if not haystacks:
        raise ValueError("haystacks must not be empty")
    return haystacks[_stable_hash(length, depth) % len(haystacks)]


# Thousands separators (comma, space, NBSP, narrow NBSP, apostrophe) that
# sit *between two digits*. Stripping these makes "1,840" == "1 840" ==
# "1840" so a correct number isn't failed on formatting, while leaving
# non-numeric punctuation (e.g. the comma in "November 14, 2023", which is
# followed by a space) untouched.
_DIGIT_SEPARATOR_RE = re.compile(r"(?<=\d)[ ,\u00a0\u202f'](?=\d)")


def score_niah(response: str, expected: str) -> tuple[bool, dict[str, Any]]:
    """Case-insensitive substring match with whitespace + number normalisation.

    Deterministic, no LLM judge (see docs/long-context-spec.md). Whitespace
    is collapsed and thousands separators inside numbers are removed so a
    formatting-only difference ("1840" vs "1,840") still counts as a match.
    Returns ``(passed, details)``.
    """

    def norm(s: str) -> str:
        collapsed = " ".join(s.lower().split())
        return _DIGIT_SEPARATOR_RE.sub("", collapsed)

    passed = norm(expected) in norm(response)
    return passed, {
        "matched": passed,
        "response_length": len(response),
        "expected": expected,
    }


# --------------------------------------------------------------------- #
# Prompt assembly                                                        #
# --------------------------------------------------------------------- #

# Probe-derived default (real models measured 6.7-7.3 chars/token). We
# fill slightly conservatively so the prompt + answer fit inside num_ctx;
# the *reported* context length is always the backend's actual
# prompt_eval_count, never this estimate.
DEFAULT_CHARS_PER_TOKEN = 6.8

# A long 131k prefill can take ~200 s on gemma-class models; the default
# 300 s request timeout is too tight. The runner bumps the backend to at
# least this when it can.
LONG_CONTEXT_MIN_TIMEOUT_S = 600.0

# Tokens reserved for the answer + a safety margin so prompt + answer fit
# inside num_ctx == the cell's nominal length.
_ANSWER_TOKEN_BUDGET = 256
_CONTEXT_SAFETY_MARGIN = 512

# Threshold below which a context length is considered "failed" for the
# first-failure-length summary (config-tunable later).
DEFAULT_FAILURE_THRESHOLD = 0.5


def estimate_chars_for_tokens(
    target_tokens: int, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
) -> int:
    return max(1, int(target_tokens * chars_per_token))


# Calibration probe: a small sample whose token count we read back from the
# backend (prompt_eval_count) to learn the model's *actual* chars/token, rather
# than trusting the fixed DEFAULT_CHARS_PER_TOKEN estimate.
_CALIBRATION_SAMPLE_CHARS = 24000
_CALIBRATION_PROBE_NUM_CTX = 32768
# Conservative lower bound on chars/token used only to keep the probe itself
# safely inside num_ctx (real prose is ~3.8-7.3); never used for cell sizing.
_CALIBRATION_MIN_RATIO = 3.0


def calibrate_chars_per_token(
    backend: Any,
    sampling: SamplingConfig,
    haystack_texts: dict[str, str],
    haystack_names: list[str],
    *,
    max_ctx: int = _CALIBRATION_PROBE_NUM_CTX,
    default_ratio: float = DEFAULT_CHARS_PER_TOKEN,
) -> float:
    """Measure the loaded model's real chars/token via ``prompt_eval_count``.

    docs/long-context-spec.md is explicit that ``prompt_eval_count`` is the
    oracle and the fixed ratio must be corrected per model — but the run loop
    historically sized every prompt with the static ``DEFAULT_CHARS_PER_TOKEN``
    (6.8). On dense public-domain prose the real ratio is ~3.8-4.5, so a
    "131072-token" cell built ~1.6-1.9x too many characters; the backend then
    truncated the prompt (Ollama silently drops the front, see its
    ``truncating input prompt`` warning) or rejected it (vLLM HTTP 400). Either
    way the planted needle disappeared for every depth except 100%, collapsing
    long-context pass rates well below 50% on models that actually retrieve
    fine once the prompt fits.

    This sends one tiny needle-free probe per haystack and returns the
    **smallest** (densest-corpus) ratio, so sizing with it never overshoots
    ``num_ctx`` on any haystack. Falls back to ``default_ratio`` if the backend
    cannot be probed. Cheap: one ~few-thousand-token prefill per haystack.
    """
    probe_ctx = min(_CALIBRATION_PROBE_NUM_CTX, max_ctx) if max_ctx else _CALIBRATION_PROBE_NUM_CTX
    # Keep the probe comfortably inside the window so its own token count is not
    # itself truncated (which would corrupt the measurement).
    sample_chars = min(_CALIBRATION_SAMPLE_CHARS, int((probe_ctx - 256) * _CALIBRATION_MIN_RATIO))
    sample_chars = max(2000, sample_chars)
    probe_sampling = sampling.model_copy(update={"num_ctx": probe_ctx, "max_tokens": 1})

    ratios: list[float] = []
    for name in haystack_names:
        text = (haystack_texts.get(name) or "")[:sample_chars]
        if not text:
            continue
        try:
            result = backend.generate(text, probe_sampling)
        except Exception:  # noqa: BLE001 — calibration must never crash the suite
            continue
        tokens = getattr(result.metrics, "prefill_tokens", 0) or 0
        if tokens > 0:
            ratios.append(len(text) / tokens)
    return min(ratios) if ratios else default_ratio


def slice_haystack(raw_text: str, target_chars: int) -> str:
    """Take the first ``target_chars`` of the haystack, tiling if short."""
    if not raw_text:
        raise ValueError("haystack text is empty")
    if len(raw_text) >= target_chars:
        return raw_text[:target_chars]
    reps = (target_chars // len(raw_text)) + 1
    return (raw_text * reps)[:target_chars]


def insert_needle(haystack: str, needle_text: str, depth_pct: int) -> str:
    """Insert ``needle_text`` into ``haystack`` at ``depth_pct`` (0-100%).

    Splits on the nearest whitespace boundary so words aren't cut.
    """
    if depth_pct <= 0:
        return f"{needle_text} {haystack}"
    if depth_pct >= 100:
        return f"{haystack} {needle_text}"
    cut = int(len(haystack) * depth_pct / 100)
    space = haystack.rfind(" ", 0, cut)
    if space <= 0:
        space = cut
    return f"{haystack[:space]} {needle_text} {haystack[space:].lstrip()}"


def cell_nonce(length: int, depth: int, repetition: int) -> str:
    """Deterministic-but-unique tag per cell.

    Unique across repetitions (defeats Ollama's prefill cache, which the
    probe showed zeroes prefill time on identical re-sends) yet stable
    across runs (reproducible task plan).
    """
    return f"{length}-{depth}-{repetition}-{_stable_hash(length, depth, repetition) % 100000:05d}"


def build_cell_prompt(haystack_with_needle: str, question: str, nonce: str) -> str:
    # The "fact is present" framing is deliberate: without it, models treat
    # the planted needle as out-of-place in the public-domain filler and
    # refuse ("not answerable"), which collapsed every non-recency cell to
    # zero across all models. This is standard needle-in-a-haystack framing.
    return (
        f"[session {nonce}] You are given a long document. A specific fact "
        "needed to answer the question has been inserted somewhere inside it. "
        "Read carefully, find that fact, and answer using only information "
        "stated in the document. The answer is present in the document, so do "
        "not reply that it is missing.\n\n"
        f"=== DOCUMENT START ===\n{haystack_with_needle}\n=== DOCUMENT END ===\n\n"
        f"Question: {question}\nAnswer:"
    )


def load_haystack_texts(fixture: LongContextFixture, repo_root: Path | str) -> dict[str, str]:
    """Read the (git-ignored, fetched) haystack texts referenced by the fixture.

    Raises FileNotFoundError with a fix-it hint if a text is missing.
    """
    root = Path(repo_root)
    texts: dict[str, str] = {}
    for name, spec in fixture.haystacks.items():
        path = root / spec.text_file
        if not path.exists():
            raise FileNotFoundError(
                f"haystack {name!r} not found at {path}. "
                "Run scripts/fetch_haystacks.sh to download the public-domain texts."
            )
        texts[name] = path.read_text(encoding="utf-8", errors="replace")
    return texts


# --------------------------------------------------------------------- #
# Runner                                                                 #
# --------------------------------------------------------------------- #

def run_long_context_suite(
    *,
    run_dir: Path,
    fixture: LongContextFixture,
    haystack_texts: dict[str, str],
    backend: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    sampling: SamplingConfig,
    progress_callback: Callable[[str], None] | None = None,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    matrix: TestMatrix | None = None,
) -> dict[str, Any]:
    """Run single-needle NIAH across the fixture grid for each model.

    Each (length, depth, repetition) cell yields one of three states:
    ``pass``/``fail`` (ran and scored), ``skipped_unsupported`` (length
    exceeds the model's claimed context), or ``error`` (backend raised,
    e.g. OOM — captured, never fatal). Pass ``matrix`` to override the
    fixture's default grid (used by the "fast" profile).
    """
    matrix = matrix or fixture.test_matrix

    # Long prefills need headroom over the default request timeout.
    if hasattr(backend, "timeout_s"):
        try:
            backend.timeout_s = max(float(backend.timeout_s), LONG_CONTEXT_MIN_TIMEOUT_S)
        except (TypeError, ValueError):
            pass

    run_rows: list[dict[str, Any]] = []
    cells_per_model = (
        len(matrix.context_lengths_tokens)
        * len(matrix.depth_percentages)
        * matrix.needles_per_cell
    )

    for model_config in model_configs:
        if progress_callback:
            progress_callback(f"  loading {model_config.name} for long-context probe")
        backend.load_model(model_config)
        # Size prompts against THIS model's real tokenizer (prompt_eval_count
        # oracle), not the static estimate — otherwise long cells overshoot
        # num_ctx and get truncated/rejected, sinking the pass rate. See
        # calibrate_chars_per_token.
        model_chars_per_token = calibrate_chars_per_token(
            backend,
            sampling,
            haystack_texts,
            matrix.haystacks,
            max_ctx=model_config.context_length,
            default_ratio=chars_per_token,
        )
        if progress_callback:
            progress_callback(
                f"  {model_config.name}: calibrated ~{model_chars_per_token:.2f} chars/token "
                f"(was static {chars_per_token:.2f})"
            )
        task_idx = 0
        for length in matrix.context_lengths_tokens:
            supported = length <= model_config.context_length
            prepared: dict[str, str] = {}
            if supported:
                target_prompt_tokens = max(
                    256, length - _ANSWER_TOKEN_BUDGET - _CONTEXT_SAFETY_MARGIN
                )
                target_chars = estimate_chars_for_tokens(target_prompt_tokens, model_chars_per_token)
                for hname in matrix.haystacks:
                    prepared[hname] = slice_haystack(haystack_texts[hname], target_chars)
            for depth in matrix.depth_percentages:
                hname = select_haystack(length, depth, matrix.haystacks)
                for rep in range(matrix.needles_per_cell):
                    task_idx += 1
                    needle = fixture.needles[
                        select_needle_index(length, depth, rep, len(fixture.needles))
                    ]
                    base_row = {
                        "suite": fixture.name,
                        "suite_version": fixture.version,
                        "model": model_config.name,
                        "model_tag": model_config.artifact_path or model_config.revision,
                        "task_id": f"{model_config.name}::len{length}::d{depth}::r{rep}",
                        "context_length": length,
                        "depth_pct": depth,
                        "repetition": rep,
                        "haystack": hname,
                        "needle_id": needle.id,
                        "needle_category": needle.category,
                        "question": needle.question,
                        "expected_substring": needle.expected_substring,
                    }
                    if not supported:
                        row = {
                            **base_row,
                            "status": "skipped_unsupported",
                            "passed": False,
                            "reason": f"claimed context {model_config.context_length} < {length}",
                        }
                        write_result(run_dir, row)
                        run_rows.append(row)
                        continue
                    if progress_callback:
                        progress_callback(
                            f"  {model_config.name} → len {length} depth {depth}% "
                            f"rep {rep + 1}/{matrix.needles_per_cell} ({task_idx}/{cells_per_model})"
                        )
                    nonce = cell_nonce(length, depth, rep)
                    hay_needle = insert_needle(prepared[hname], needle.text, depth)
                    prompt = build_cell_prompt(hay_needle, needle.question, nonce)
                    cell_sampling = sampling.model_copy(
                        update={
                            "num_ctx": length,
                            "max_tokens": min(sampling.max_tokens or _ANSWER_TOKEN_BUDGET, _ANSWER_TOKEN_BUDGET),
                        }
                    )
                    try:
                        generation: GenerationResult = backend.generate(prompt, cell_sampling)
                    except Exception as exc:  # backend OOM / HTTP error / timeout
                        row = {
                            **base_row,
                            "status": "error",
                            "passed": False,
                            "reason": f"{type(exc).__name__}: {exc}"[:300],
                        }
                        write_result(run_dir, row)
                        run_rows.append(row)
                        continue
                    passed, details = score_niah(generation.output, needle.expected_substring)
                    metrics = generation.metrics
                    prefill_tps = (
                        metrics.prefill_tokens / metrics.prefill_time_s
                        if metrics.prefill_time_s
                        else 0.0
                    )
                    memory = None
                    snap = getattr(backend, "memory_snapshot", None)
                    if callable(snap):
                        try:
                            memory = snap()
                        except Exception:
                            memory = None
                    row = {
                        **base_row,
                        "status": "pass" if passed else "fail",
                        "passed": passed,
                        "context_tokens_loaded": metrics.prefill_tokens,
                        "prefill_time_s": metrics.prefill_time_s,
                        "prefill_tokens_per_sec": round(prefill_tps, 2),
                        "memory": memory,
                        "output_preview": generation.output[:280],
                    }
                    write_result(run_dir, row)
                    run_rows.append(row)
        if progress_callback:
            progress_callback(f"  unloading {model_config.name}")
        backend.unload()

    summary = build_long_context_summary(run_rows, fixture, backend_config, matrix=matrix)
    write_json(run_dir / "summary.json", summary)
    write_long_context_summary_markdown(run_dir, summary)
    return summary


# --------------------------------------------------------------------- #
# Summary                                                                #
# --------------------------------------------------------------------- #

def _first_failure_length(cells: list[dict[str, Any]], threshold: float) -> int | None:
    by_len: dict[int, list[float]] = {}
    for c in cells:
        if c.get("pass_rate") is None:
            continue
        by_len.setdefault(c["context_length"], []).append(c["pass_rate"])
    for length in sorted(by_len):
        rates = by_len[length]
        if sum(rates) / len(rates) < threshold:
            return length
    return None


def build_long_context_summary(
    run_rows: list[dict[str, Any]],
    fixture: LongContextFixture,
    backend: BackendConfig,
    *,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    matrix: TestMatrix | None = None,
) -> dict[str, Any]:
    matrix = matrix or fixture.test_matrix
    per_model: dict[str, dict[str, Any]] = {}

    for row in run_rows:
        name = row["model"]
        bucket = per_model.setdefault(
            name,
            {
                "model": name,
                "_cells": {},
                "_categories": {},
                "total": 0,
                "passes": 0,
                "skipped": 0,
                "errors": 0,
            },
        )
        bucket["total"] += 1
        cell_key = f"{row['context_length']}|{row['depth_pct']}"
        cell = bucket["_cells"].setdefault(
            cell_key,
            {
                "context_length": row["context_length"],
                "depth_pct": row["depth_pct"],
                "passes": 0,
                "n": 0,
                "skipped": 0,
                "errors": 0,
                "_tps": [],
                "_vram": [],
            },
        )
        status = row.get("status")
        if status == "skipped_unsupported":
            bucket["skipped"] += 1
            cell["skipped"] += 1
            continue
        # Track per-needle-category retrieval for every scored attempt
        # (pass/fail/error), since needle *type* — e.g. alphanumeric codes
        # vs. plain names — drives pass rate as much as position does.
        catb = bucket["_categories"].setdefault(
            row.get("needle_category") or "unknown",
            {"category": row.get("needle_category") or "unknown", "passes": 0, "n": 0},
        )
        catb["n"] += 1
        if status == "error":
            bucket["errors"] += 1
            cell["errors"] += 1
            cell["n"] += 1
            continue
        cell["n"] += 1
        if row.get("passed"):
            bucket["passes"] += 1
            cell["passes"] += 1
            catb["passes"] += 1
        if row.get("prefill_tokens_per_sec"):
            cell["_tps"].append(row["prefill_tokens_per_sec"])
        mem = row.get("memory") or {}
        if mem.get("size_vram_mb"):
            cell["_vram"].append(mem["size_vram_mb"])

    models_out = []
    for bucket in per_model.values():
        cells = []
        for c in bucket["_cells"].values():
            c["pass_rate"] = round(c["passes"] / c["n"], 4) if c["n"] else None
            c["avg_prefill_tps"] = round(sum(c["_tps"]) / len(c["_tps"]), 1) if c["_tps"] else None
            c["peak_vram_mb"] = max(c["_vram"]) if c["_vram"] else None
            del c["_tps"]
            del c["_vram"]
            cells.append(c)
        cells.sort(key=lambda c: (c["context_length"], c["depth_pct"]))
        categories = []
        for cat in bucket["_categories"].values():
            cat["pass_rate"] = round(cat["passes"] / cat["n"], 4) if cat["n"] else None
            categories.append(cat)
        categories.sort(key=lambda c: c["category"])
        scored = bucket["total"] - bucket["skipped"]
        models_out.append(
            {
                "model": bucket["model"],
                "total": bucket["total"],
                "passes": bucket["passes"],
                "skipped": bucket["skipped"],
                "errors": bucket["errors"],
                "pass_rate": round(bucket["passes"] / scored, 4) if scored else None,
                "first_failure_length": _first_failure_length(cells, failure_threshold),
                "cells": cells,
                "categories": categories,
            }
        )

    return {
        "suite": fixture.name,
        "suite_version": fixture.version,
        "backend": backend.name.value,
        "grid": {
            "context_lengths": matrix.context_lengths_tokens,
            "depths": matrix.depth_percentages,
            "needles_per_cell": matrix.needles_per_cell,
        },
        "total_rows": len(run_rows),
        "models": models_out,
    }


def write_long_context_summary_markdown(run_dir: Path, summary: dict[str, Any]) -> Path:
    lines = [
        f"# {summary['suite']} summary",
        "",
        f"- backend: {summary['backend']}",
        f"- grid: {summary['grid']['context_lengths']} × depths "
        f"{summary['grid']['depths']} × {summary['grid']['needles_per_cell']} needles/cell",
        f"- total rows: {summary['total_rows']}",
        "",
    ]
    depths = summary["grid"]["depths"]
    for model in summary["models"]:
        lines.append(f"## {model['model']}")
        lines.append("")
        ffl = model["first_failure_length"]
        lines.append(
            f"- overall pass rate: "
            f"{model['pass_rate']:.1%}" if model["pass_rate"] is not None else "- overall pass rate: n/a"
        )
        lines.append(f"- first-failure length: {ffl if ffl else 'none (held up across grid)'}")
        lines.append(f"- skipped (unsupported): {model['skipped']}, errors: {model['errors']}")
        lines.append("")
        header = "| context | " + " | ".join(f"{d}%" for d in depths) + " |"
        sep = "| ---: |" + " ---: |" * len(depths)
        lines.append(header)
        lines.append(sep)
        by_len: dict[int, dict[int, Any]] = {}
        for c in model["cells"]:
            by_len.setdefault(c["context_length"], {})[c["depth_pct"]] = c
        for length in sorted(by_len):
            row_cells = []
            for d in depths:
                c = by_len[length].get(d)
                if not c or c["pass_rate"] is None:
                    row_cells.append("N/A")
                else:
                    row_cells.append(f"{c['pass_rate']:.0%}")
            lines.append(f"| {length} | " + " | ".join(row_cells) + " |")
        lines.append("")
    path = run_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
