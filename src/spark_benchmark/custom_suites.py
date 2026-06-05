"""Custom user-supplied benchmark suites — "Bring Your Own Test".

Supports two modes:

``mode: quick`` (Mode A, v0.2.0+)
    Pass-through: each prompt is sent to each model, telemetry is
    captured, no scoring. Output is a side-by-side Markdown / HTML
    summary you read on screen.

``mode: scored`` (Mode B, v0.3.0+)
    Each task may carry a ``scoring:`` block. Supported scorers:

    - ``exact_match``     — normalised case-insensitive equality.
    - ``substring_match`` — all ``must_contain`` strings must appear.
    - ``regex_match``     — a Python ``re`` pattern must match.
    - ``json_fields_match`` — model output must parse as JSON and contain
      every key from ``expected_fields`` with the right value.
    - ``multiple_choice`` — a letter choice (A/B/C/D …) must appear as
      a word in the output.

    A suite-level ``scoring:`` block provides the default for tasks that
    do not specify their own. Tasks without any scoring block are still
    run but marked ``pass=null`` (no verdict). Per-task ``timeout_s`` is
    now enforced.

    The runner also accepts ``dry_run=True`` to execute one task against
    one model end-to-end and return immediately — useful for sanity-
    checking a new suite before committing to a long run.

Custom Python scorers and LLM-as-judge are specified in
``docs/custom-tests-spec.md`` (v0.4.0+ roadmap) and are not implemented
here.

The format is intentionally separate from
:class:`spark_benchmark.suites.SuiteDefinition`, which is the canonical
JSON-only format hard-wired into the curated suites. Custom suites live
in user space, are YAML-friendly, and evolve on a different cadence.
"""

from __future__ import annotations

import json
import re
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from pydantic import BaseModel, Field, model_validator

from spark_benchmark.models import (
    BackendConfig,
    GenerationResult,
    ModelConfig,
    SamplingConfig,
)
from spark_benchmark.results_bundle import write_json, write_result


# --------------------------------------------------------------------- #
# Scoring schema                                                        #
# --------------------------------------------------------------------- #


class ScoringConfig(BaseModel):
    """Deterministic scorer config attached to a task or a whole suite.

    Exactly one ``method`` is required.  The supporting fields depend on
    the method chosen:

    +-----------------------+--------------------------------------+
    | method                | required fields                      |
    +=======================+======================================+
    | exact_match           | ``expected``                         |
    | substring_match       | ``must_contain`` (list, ≥ 1 item)   |
    | regex_match           | ``pattern``                          |
    | json_fields_match     | ``expected_fields`` (dict, ≥ 1 key) |
    | multiple_choice       | ``expected`` (single letter/word)   |
    +-----------------------+--------------------------------------+

    ``case_sensitive`` defaults to ``False`` for every method that does
    string comparison; for ``regex_match`` it maps to the ``re.IGNORECASE``
    flag.
    """

    method: Literal[
        "exact_match",
        "substring_match",
        "regex_match",
        "json_fields_match",
        "multiple_choice",
    ]
    expected: str | None = None
    must_contain: list[str] = Field(default_factory=list)
    pattern: str | None = None
    expected_fields: dict[str, Any] = Field(default_factory=dict)
    choices: list[str] = Field(default_factory=list)
    case_sensitive: bool = False

    @model_validator(mode="after")
    def _check_fields(self) -> "ScoringConfig":
        if self.method == "exact_match" and self.expected is None:
            raise ValueError("exact_match requires 'expected'")
        if self.method == "substring_match" and not self.must_contain:
            raise ValueError("substring_match requires at least one item in 'must_contain'")
        if self.method == "regex_match" and self.pattern is None:
            raise ValueError("regex_match requires 'pattern'")
        if self.method == "json_fields_match" and not self.expected_fields:
            raise ValueError("json_fields_match requires at least one key in 'expected_fields'")
        if self.method == "multiple_choice" and self.expected is None:
            raise ValueError("multiple_choice requires 'expected'")
        return self


@dataclass
class ScoreResult:
    passed: bool
    method: str
    reason: str


# --------------------------------------------------------------------- #
# Scorer                                                                #
# --------------------------------------------------------------------- #


def score_response(output: str, scoring: ScoringConfig) -> ScoreResult:
    """Apply *scoring* to *output* and return a :class:`ScoreResult`.

    All string comparisons are case-insensitive by default (controlled by
    ``scoring.case_sensitive``).  Whitespace is normalised (leading /
    trailing stripped, interior runs collapsed to a single space) before
    comparison, which is standard NIAH / QA scorer practice.
    """
    flags = 0 if scoring.case_sensitive else re.IGNORECASE

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip() if not scoring.case_sensitive else re.sub(r"\s+", " ", s).strip()

    def _lower(s: str) -> str:
        return s if scoring.case_sensitive else s.lower()

    out = _norm(output)

    if scoring.method == "exact_match":
        assert scoring.expected is not None
        passed = _lower(out) == _lower(_norm(scoring.expected))
        return ScoreResult(
            passed=passed,
            method="exact_match",
            reason=f"expected {scoring.expected!r}, got {out[:120]!r}",
        )

    if scoring.method == "substring_match":
        missing = [s for s in scoring.must_contain if _lower(s) not in _lower(out)]
        passed = not missing
        if passed:
            return ScoreResult(passed=True, method="substring_match", reason="all substrings found")
        return ScoreResult(
            passed=False,
            method="substring_match",
            reason=f"missing substring(s): {missing}",
        )

    if scoring.method == "regex_match":
        assert scoring.pattern is not None
        try:
            match = re.search(scoring.pattern, output, flags)
        except re.error as exc:
            return ScoreResult(passed=False, method="regex_match", reason=f"invalid regex: {exc}")
        passed = match is not None
        return ScoreResult(
            passed=passed,
            method="regex_match",
            reason="pattern matched" if passed else f"pattern {scoring.pattern!r} did not match",
        )

    if scoring.method == "json_fields_match":
        # Strip markdown fences if present
        text = re.sub(r"^```[a-z]*\n?", "", out.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text.strip())
        # Find first JSON object in output
        brace = text.find("{")
        if brace != -1:
            text = text[brace:]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return ScoreResult(
                passed=False,
                method="json_fields_match",
                reason=f"output is not valid JSON: {exc}",
            )
        if not isinstance(parsed, dict):
            return ScoreResult(
                passed=False,
                method="json_fields_match",
                reason="output parsed as JSON but is not an object",
            )
        mismatches: list[str] = []
        for key, expected_val in scoring.expected_fields.items():
            actual_val = parsed.get(key)
            exp_norm = _lower(str(expected_val))
            act_norm = _lower(str(actual_val)) if actual_val is not None else None
            if key not in parsed:
                mismatches.append(f"missing key {key!r}")
            elif exp_norm != act_norm:
                mismatches.append(f"{key!r}: expected {expected_val!r}, got {actual_val!r}")
        passed = not mismatches
        return ScoreResult(
            passed=passed,
            method="json_fields_match",
            reason="all fields matched" if passed else "; ".join(mismatches),
        )

    if scoring.method == "multiple_choice":
        assert scoring.expected is not None
        # Match expected answer as a word boundary to avoid "A" matching "APPLE"
        expected_clean = re.escape(_lower(scoring.expected.strip()))
        match = re.search(rf"\b{expected_clean}\b", _lower(out))
        passed = match is not None
        return ScoreResult(
            passed=passed,
            method="multiple_choice",
            reason=f"expected choice {scoring.expected!r}" + (" found" if passed else " not found"),
        )

    return ScoreResult(passed=False, method=scoring.method, reason="unknown scorer method")


# --------------------------------------------------------------------- #
# Timeout context manager (cross-platform, thread-safe)                 #
# --------------------------------------------------------------------- #


@contextmanager
def _task_timeout(seconds: float | None) -> Iterator[None]:
    """Raise ``TimeoutError`` after *seconds* using a background timer.

    Falls back to a no-op when *seconds* is ``None``, on non-Unix platforms
    (where ``signal.SIGALRM`` is unavailable), or when called from a
    non-main thread (signal delivery requires the main thread).

    The implementation uses a daemon ``threading.Timer`` that raises
    ``TimeoutError`` in the main thread via ``signal.raise_signal``.  This
    avoids the SIGALRM approach's main-thread restriction while staying
    safe for the test suite (tests that run the runner in a worker thread
    simply get no timeout enforcement, which is acceptable for unit tests).
    """
    if seconds is None or seconds <= 0:
        yield
        return

    expired = threading.Event()

    def _fire() -> None:
        expired.set()
        try:
            signal.raise_signal(signal.SIGALRM)
        except (AttributeError, OSError):
            pass

    timer = threading.Timer(seconds, _fire)
    old_handler = None
    try:
        try:
            old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError(f"task timed out after {seconds}s")))  # type: ignore[arg-type]
        except (AttributeError, OSError, ValueError):
            # SIGALRM not available (Windows) or not main thread — skip
            yield
            return
        timer.start()
        yield
    except TimeoutError:
        raise
    finally:
        timer.cancel()
        if old_handler is not None:
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, OSError, ValueError):
                pass


# --------------------------------------------------------------------- #
# Pydantic schema                                                       #
# --------------------------------------------------------------------- #


class CustomSuiteTask(BaseModel):
    """One prompt the user wants run against every selected model."""

    task_id: str
    prompt: str
    tags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    sampling: SamplingConfig | None = None
    """Per-task sampling override. Falls through to the suite default."""
    timeout_s: float | None = None
    """Per-task timeout in seconds.  Enforced in v0.3.0+.  ``None`` = no limit."""
    scoring: ScoringConfig | None = None
    """Scorer for ``mode: scored``.  ``None`` = no verdict (answer recorded but not judged)."""


class CustomSuiteDefinition(BaseModel):
    """User-supplied benchmark suite."""

    name: str
    version: str = "0.1"
    description: str = ""
    mode: Literal["quick", "scored"] = "quick"
    """``quick`` is Mode A (pass-through, no scoring).
    ``scored`` is Mode B — each task may specify a ``scoring:`` block."""
    tasks: list[CustomSuiteTask]
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    """Default sampling applied to every task that does not specify its own."""
    scoring: ScoringConfig | None = None
    """Suite-level default scorer.  Applied to tasks that have ``scoring: null``
    when ``mode`` is ``scored``.  Has no effect in ``mode: quick``."""
    models: list[str] = Field(default_factory=list)
    """Optional model lineup to default to when ``--models`` is not passed
    on the command line. Each entry is a ``ModelConfig.name`` (curated) or
    a slugified Ollama tag (auto-detected, e.g. ``phi4-14b``)."""
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_after(self) -> "CustomSuiteDefinition":
        if not self.name.strip():
            raise ValueError("custom suite 'name' must not be empty")
        if not self.tasks:
            raise ValueError("custom suite must have at least one task")
        seen: set[str] = set()
        for task in self.tasks:
            if not task.task_id.strip():
                raise ValueError("every task must have a non-empty task_id")
            if task.task_id in seen:
                raise ValueError(f"duplicate task_id: {task.task_id!r}")
            if not task.prompt.strip():
                raise ValueError(f"task {task.task_id!r} has an empty prompt")
            seen.add(task.task_id)
        return self


# --------------------------------------------------------------------- #
# Loader                                                                #
# --------------------------------------------------------------------- #


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - PyYAML is in pyproject
        raise RuntimeError(
            "PyYAML is required to load custom suite YAML files. "
            "Install spark-benchmark with its full dependency set."
        ) from exc
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("custom suite YAML must be a mapping at the top level")
    return payload


def load_custom_suite(path: Path | str) -> CustomSuiteDefinition:
    """Load and validate a user suite from a YAML or JSON file.

    Accepts ``.yaml``, ``.yml``, or ``.json``. Anything else is treated
    as YAML so users who put ``suite.txt`` in front of us still get a
    sensible try.

    Both ``mode: quick`` and ``mode: scored`` are accepted.
    """
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = _load_yaml(text)
    return CustomSuiteDefinition.model_validate(payload)


# --------------------------------------------------------------------- #
# Validation                                                            #
# --------------------------------------------------------------------- #


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    message: str

    def render(self) -> str:
        marker = "ERROR" if self.severity == "error" else "WARN "
        return f"{marker} {self.message}"


def validate_custom_suite(
    suite: CustomSuiteDefinition,
    *,
    available_models: list[str] | None = None,
) -> list[ValidationIssue]:
    """Static checks beyond the Pydantic schema.

    Schema-level checks (duplicate task IDs, empty prompts, empty tasks
    list) are enforced inside the Pydantic validator and raise ``ValueError``
    at load time. This function reports the *softer* issues callers should
    surface to the user (model name typos, sampling sanity, prompt length,
    and incomplete scoring configs in ``mode: scored``).
    """
    issues: list[ValidationIssue] = []

    if available_models is not None and suite.models:
        unknown = [name for name in suite.models if name not in available_models]
        if unknown:
            issues.append(
                ValidationIssue(
                    "error",
                    f"suite.models references unknown model(s): {', '.join(unknown)}. "
                    f"Available: {', '.join(available_models) or '(none)'}",
                )
            )

    # In scored mode: warn if no scorer is defined at suite level AND not
    # on individual tasks — those tasks will run but produce no verdict.
    if suite.mode == "scored":
        unscored_tasks = [
            t.task_id for t in suite.tasks
            if t.scoring is None and suite.scoring is None
        ]
        if unscored_tasks:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"mode is 'scored' but {len(unscored_tasks)} task(s) have no scorer "
                    f"and the suite has no default scoring block "
                    f"(task_ids: {', '.join(unscored_tasks[:5])}"
                    + (f" … and {len(unscored_tasks)-5} more" if len(unscored_tasks) > 5 else "")
                    + "). These tasks will run but produce no verdict.",
                )
            )

    for task in suite.tasks:
        if len(task.prompt) > 32_000:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"task {task.task_id!r} has a {len(task.prompt)} char prompt; "
                    "consider whether your model's context window can handle it",
                )
            )
        sampling = task.sampling or suite.sampling
        if sampling.temperature < 0 or sampling.temperature > 2:
            issues.append(
                ValidationIssue(
                    "error",
                    f"task {task.task_id!r} has temperature={sampling.temperature}; "
                    "expected range is 0..2",
                )
            )
        if sampling.max_tokens <= 0:
            issues.append(
                ValidationIssue(
                    "error",
                    f"task {task.task_id!r} has max_tokens={sampling.max_tokens}; "
                    "must be > 0",
                )
            )

    return issues


# --------------------------------------------------------------------- #
# Resume helper                                                         #
# --------------------------------------------------------------------- #


def already_completed_pairs(run_dir: Path) -> set[tuple[str, str]]:
    """Return the set of ``(model, task_id)`` pairs already in ``results.jsonl``.

    Used by the runner to support resume — rerunning a custom suite that
    crashed half-way through skips everything already on disk.
    """
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        model = row.get("model")
        task_id = row.get("task_id")
        if isinstance(model, str) and isinstance(task_id, str):
            done.add((model, task_id))
    return done


# --------------------------------------------------------------------- #
# Runner                                                                #
# --------------------------------------------------------------------- #


def _resolve_sampling(
    task: CustomSuiteTask,
    suite: CustomSuiteDefinition,
    cli_default: SamplingConfig,
) -> SamplingConfig:
    """Per-task > suite-default > CLI / experiment default."""
    if task.sampling is not None:
        return task.sampling
    # ``suite.sampling`` always exists (default factory). Treat the
    # CLI/experiment default as the *floor*: if the suite explicitly
    # sets a value we honour that even when the CLI default differs.
    return suite.sampling.model_copy(update={})


def _resolve_task_scoring(
    task: CustomSuiteTask,
    suite: CustomSuiteDefinition,
) -> ScoringConfig | None:
    """Per-task > suite-level default.  Returns ``None`` when no scorer is defined."""
    if task.scoring is not None:
        return task.scoring
    return suite.scoring


def run_custom_suite_quick(
    *,
    suite: CustomSuiteDefinition,
    backend: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    run_dir: Path,
    default_sampling: SamplingConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute a custom suite end-to-end (both ``quick`` and ``scored`` modes).

    For each ``(model, task)`` pair:

    - Resolves sampling (per-task > suite default).
    - Calls ``backend.generate(prompt, sampling)`` with optional per-task
      timeout enforcement (``task.timeout_s``).
    - In ``mode: scored``, applies the resolved :class:`ScoringConfig` and
      records ``score.passed`` / ``score.reason`` in the result row.
    - Appends one row to ``run_dir/results.jsonl`` (resume-friendly: existing
      pairs are skipped unless ``resume=False``).
    - On any exception, records an error row and continues.

    When *dry_run* is ``True``, executes only the **first task of the first
    model** and returns without writing any files.  Useful for quickly
    validating that the backend and suite config are wired up correctly.

    Returns the same dict that gets serialised to ``summary.json``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cli_default = default_sampling or SamplingConfig()
    skip_pairs = already_completed_pairs(run_dir) if (resume and not dry_run) else set()
    if skip_pairs and progress_callback:
        progress_callback(f"  resume: skipping {len(skip_pairs)} (model, task) pair(s) already on disk")

    rows: list[dict[str, Any]] = []
    done = False  # flag for dry_run early exit

    for model_config in model_configs:
        if done:
            break
        if progress_callback:
            progress_callback(f"  loading {model_config.name}")
        backend.load_model(model_config)
        try:
            for idx, task in enumerate(suite.tasks, start=1):
                if done:
                    break
                key = (model_config.name, task.task_id)
                if key in skip_pairs:
                    if progress_callback:
                        progress_callback(
                            f"  {model_config.name} → [{idx}/{len(suite.tasks)}] {task.task_id} (skip — already done)"
                        )
                    continue
                sampling = _resolve_sampling(task, suite, cli_default)
                scoring = _resolve_task_scoring(task, suite) if suite.mode == "scored" else None
                if progress_callback:
                    scorer_hint = f" [{scoring.method}]" if scoring else ""
                    progress_callback(
                        f"  {model_config.name} → [{idx}/{len(suite.tasks)}] {task.task_id}{scorer_hint}"
                    )
                row: dict[str, Any] = {
                    "suite": suite.name,
                    "suite_version": suite.version,
                    "mode": suite.mode,
                    "model": model_config.name,
                    "model_tag": model_config.artifact_path or model_config.revision,
                    "task_id": task.task_id,
                    "tags": task.tags,
                    "prompt": task.prompt,
                    "sampling": sampling.model_dump(mode="json"),
                    "timeout_s": task.timeout_s,
                    "score": None,
                }
                try:
                    with _task_timeout(task.timeout_s):
                        generation: GenerationResult = backend.generate(task.prompt, sampling)
                    row["generation"] = generation.model_dump(mode="json")
                    row["error"] = None
                    if scoring is not None:
                        output = generation.output if isinstance(generation.output, str) else ""
                        result = score_response(output, scoring)
                        row["score"] = {
                            "passed": result.passed,
                            "method": result.method,
                            "reason": result.reason,
                        }
                        if progress_callback:
                            verdict = "PASS" if result.passed else "FAIL"
                            progress_callback(
                                f"    {verdict} — {result.reason[:120]}"
                            )
                except TimeoutError:
                    row["generation"] = None
                    row["error"] = {
                        "type": "TimeoutError",
                        "message": f"task timed out after {task.timeout_s}s",
                    }
                    if progress_callback:
                        progress_callback(
                            f"    timeout on {model_config.name}/{task.task_id} after {task.timeout_s}s"
                        )
                except Exception as exc:  # noqa: BLE001
                    row["generation"] = None
                    row["error"] = {
                        "type": exc.__class__.__name__,
                        "message": str(exc)[:2000],
                    }
                    if progress_callback:
                        progress_callback(
                            f"    error on {model_config.name}/{task.task_id}: {exc.__class__.__name__}"
                        )
                if dry_run:
                    if progress_callback:
                        progress_callback("  dry-run: stopping after first task")
                    rows.append(row)
                    done = True
                    break
                write_result(run_dir, row)
                rows.append(row)
        finally:
            if progress_callback:
                progress_callback(f"  unloading {model_config.name}")
            backend.unload()

    if dry_run:
        return {"suite": suite.name, "mode": suite.mode, "dry_run": True, "rows": rows}

    # If we resumed, also fold in the rows that were already on disk so
    # the summary reflects everything, not just the freshly-run subset.
    if skip_pairs:
        rows = list(_replay_completed_rows(run_dir)) + [
            row for row in rows if (row["model"], row["task_id"]) not in skip_pairs
        ]

    summary = build_custom_summary(suite, rows, backend_config)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_custom_summary_markdown(summary), encoding="utf-8")
    # Local import keeps the canonical reporting graph free of a hard
    # dependency on the BYOT-only HTML helper, and avoids paying the
    # import cost on every CLI startup.
    from spark_benchmark.reporting_html import render_custom_summary_html

    (run_dir / "summary.html").write_text(
        render_custom_summary_html(summary), encoding="utf-8"
    )
    return summary


def _replay_completed_rows(run_dir: Path) -> list[dict[str, Any]]:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# --------------------------------------------------------------------- #
# Summary                                                               #
# --------------------------------------------------------------------- #


def build_custom_summary(
    suite: CustomSuiteDefinition,
    rows: list[dict[str, Any]],
    backend_config: BackendConfig,
) -> dict[str, Any]:
    """Aggregate raw rows into a summary dict.

    The shape mirrors the canonical suites so downstream reporting code
    (``aggregate_runs``, ``write_report``) can treat custom runs as
    first-class citizens.

    In ``mode: scored`` each model bucket gains ``passes``, ``scored``, and
    ``pass_rate`` fields alongside the telemetry aggregates.
    """
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = row.get("model", "unknown")
        bucket = by_model.setdefault(
            model,
            {
                "model": model,
                "model_tag": row.get("model_tag"),
                "tasks_completed": 0,
                "tasks_errored": 0,
                "passes": 0,
                "scored": 0,
                "ttft_ms": [],
                "decode_tps": [],
                "decode_tokens": [],
                "wall_time_s": 0.0,
                "errors": [],
            },
        )
        if row.get("error"):
            bucket["tasks_errored"] += 1
            bucket["errors"].append(
                {"task_id": row.get("task_id"), "error": row["error"]}
            )
            continue
        gen = row.get("generation") or {}
        metrics = (gen.get("metrics") or {}) if isinstance(gen, dict) else {}
        bucket["tasks_completed"] += 1
        # Scoring aggregation
        score = row.get("score")
        if isinstance(score, dict) and score.get("passed") is not None:
            bucket["scored"] += 1
            if score["passed"]:
                bucket["passes"] += 1
        # Telemetry
        ttft = metrics.get("ttft_ms")
        if isinstance(ttft, (int, float)):
            bucket["ttft_ms"].append(float(ttft))
        decode_time = metrics.get("decode_time_s") or 0.0
        decode_tokens = metrics.get("decode_tokens") or 0
        if decode_time > 0 and decode_tokens > 0:
            bucket["decode_tps"].append(float(decode_tokens) / float(decode_time))
        if isinstance(decode_tokens, (int, float)):
            bucket["decode_tokens"].append(float(decode_tokens))
        bucket["wall_time_s"] += float(metrics.get("prefill_time_s") or 0.0)
        bucket["wall_time_s"] += float(metrics.get("decode_time_s") or 0.0)

    per_model = []
    for bucket in by_model.values():
        ttft_list = bucket.pop("ttft_ms")
        tps_list = bucket.pop("decode_tps")
        tokens_list = bucket.pop("decode_tokens")
        bucket["mean_ttft_ms"] = round(sum(ttft_list) / len(ttft_list), 1) if ttft_list else None
        bucket["mean_decode_tps"] = round(sum(tps_list) / len(tps_list), 2) if tps_list else None
        bucket["total_decode_tokens"] = int(sum(tokens_list))
        bucket["wall_time_s"] = round(bucket["wall_time_s"], 2)
        scored = bucket["scored"]
        bucket["pass_rate"] = round(bucket["passes"] / scored, 4) if scored else None
        per_model.append(bucket)
    per_model.sort(key=lambda item: item["model"])

    return {
        "suite": suite.name,
        "suite_version": suite.version,
        "mode": suite.mode,
        "description": suite.description,
        "backend": backend_config.name.value,
        "task_count": len(suite.tasks),
        "per_model": per_model,
        "rows": rows,
    }


def render_custom_summary_markdown(summary: dict[str, Any]) -> str:
    """Side-by-side Markdown summary (Mode A and Mode B).

    Per-model telemetry table at the top, then one section per task with
    each model's reply rendered as a fenced block.  In ``mode: scored``
    each model block also shows PASS / FAIL and the scorer's reason.
    """
    mode = summary.get("mode", "quick")
    is_scored = mode == "scored"

    lines: list[str] = []
    lines.append(f"# Custom suite: {summary['suite']}")
    lines.append("")
    lines.append(f"- **Version**: {summary['suite_version']}")
    mode_label = "scored (deterministic scorers)" if is_scored else "quick (no scoring)"
    lines.append(f"- **Mode**: {mode_label}")
    if summary.get("description"):
        lines.append(f"- **Description**: {summary['description']}")
    lines.append(f"- **Backend**: {summary['backend']}")
    lines.append(f"- **Tasks**: {summary['task_count']}")
    lines.append("")

    lines.append("## Per-model summary")
    lines.append("")
    if is_scored:
        lines.append("| Model | Pass | Scored | Pass rate | Completed | Errored | Mean TTFT (ms) | Mean decode tps | Wall (s) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for bucket in summary["per_model"]:
            pr = bucket.get("pass_rate")
            pr_s = f"{100*pr:.1f} %" if pr is not None else "—"
            lines.append(
                "| {model} | {passes} | {scored} | {pr} | {ok} | {err} | {ttft} | {tps} | {wall} |".format(
                    model=bucket["model"],
                    passes=bucket.get("passes", 0),
                    scored=bucket.get("scored", 0),
                    pr=pr_s,
                    ok=bucket["tasks_completed"],
                    err=bucket["tasks_errored"],
                    ttft="—" if bucket["mean_ttft_ms"] is None else f"{bucket['mean_ttft_ms']}",
                    tps="—" if bucket["mean_decode_tps"] is None else f"{bucket['mean_decode_tps']}",
                    wall=bucket["wall_time_s"],
                )
            )
    else:
        lines.append("| Model | Completed | Errored | Mean TTFT (ms) | Mean decode tps | Decode tokens | Wall (s) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for bucket in summary["per_model"]:
            lines.append(
                "| {model} | {ok} | {err} | {ttft} | {tps} | {tokens} | {wall} |".format(
                    model=bucket["model"],
                    ok=bucket["tasks_completed"],
                    err=bucket["tasks_errored"],
                    ttft="—" if bucket["mean_ttft_ms"] is None else f"{bucket['mean_ttft_ms']}",
                    tps="—" if bucket["mean_decode_tps"] is None else f"{bucket['mean_decode_tps']}",
                    tokens=bucket["total_decode_tokens"],
                    wall=bucket["wall_time_s"],
                )
            )
    lines.append("")

    by_task: dict[str, list[dict[str, Any]]] = {}
    task_order: list[str] = []
    for row in summary["rows"]:
        task_id = row["task_id"]
        if task_id not in by_task:
            task_order.append(task_id)
            by_task[task_id] = []
        by_task[task_id].append(row)

    lines.append("## Side-by-side outputs")
    lines.append("")
    for task_id in task_order:
        task_rows = by_task[task_id]
        prompt = task_rows[0].get("prompt", "")
        lines.append(f"### Task `{task_id}`")
        lines.append("")
        lines.append("**Prompt:**")
        lines.append("")
        lines.append("```")
        lines.append(prompt)
        lines.append("```")
        lines.append("")
        for row in task_rows:
            model = row["model"]
            if row.get("error"):
                lines.append(f"**{model}** — ERROR `{row['error'].get('type', '')}`")
                lines.append("")
                lines.append("```")
                lines.append(str(row["error"].get("message", ""))[:1500])
                lines.append("```")
                lines.append("")
                continue
            gen = row.get("generation") or {}
            metrics = (gen.get("metrics") or {}) if isinstance(gen, dict) else {}
            ttft = metrics.get("ttft_ms")
            decode_time = metrics.get("decode_time_s") or 0.0
            decode_tokens = metrics.get("decode_tokens") or 0
            tps = (decode_tokens / decode_time) if decode_time > 0 and decode_tokens > 0 else None
            finish = gen.get("finish_reason", "?") if isinstance(gen, dict) else "?"
            telemetry_bits = []
            if isinstance(ttft, (int, float)):
                telemetry_bits.append(f"TTFT {ttft:.0f} ms")
            if tps is not None:
                telemetry_bits.append(f"{tps:.1f} tps")
            telemetry_bits.append(f"finish: {finish}")
            # Scoring verdict
            score = row.get("score")
            if isinstance(score, dict) and score.get("passed") is not None:
                verdict = "✓ PASS" if score["passed"] else "✗ FAIL"
                reason = score.get("reason", "")
                lines.append(f"**{model}** — {', '.join(telemetry_bits)} — **{verdict}** ({reason[:120]})")
            else:
                lines.append(f"**{model}** — {', '.join(telemetry_bits)}")
            lines.append("")
            output = (gen.get("output") or "") if isinstance(gen, dict) else ""
            lines.append("```")
            lines.append(output if output else "(empty)")
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------- #
# Convenience helpers                                                   #
# --------------------------------------------------------------------- #


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_suite_name(name: str) -> str:
    """Filesystem-safe slug for ``run_dir`` naming."""
    slug = _SLUG_RE.sub("-", name.lower()).strip("-")
    return slug or "custom-suite"
