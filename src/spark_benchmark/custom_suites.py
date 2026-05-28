"""Custom user-supplied benchmark suites — "Bring Your Own Test" / Mode A.

This module is the ``v0.2.0`` implementation of the BYOT (Bring Your Own
Test) extension. It deliberately covers only **Mode A** ("quick" / pass-
through, no scoring). Mode B (deterministic scorers, custom Python,
LLM-as-judge) is specified in ``docs/custom-tests-spec.md`` and ships in
``v0.3.0`` and later.

What it does today
------------------

- Load a user YAML or JSON suite file.
- Run every task against every model the user picked.
- Stream per-task progress to the caller.
- Append every row to ``results.jsonl`` (resume-friendly: rerunning the
  same ``run_dir`` skips already-completed `(model, task_id)` pairs).
- Write a side-by-side ``summary.md``, a polished standalone
  ``summary.html`` (no JS, no CDN — open it anywhere), and a
  machine-readable ``summary.json`` next to the JSONL.

What it does **not** do (yet)
-----------------------------

- No scoring of any kind. Mode B (``mode: scored``) is rejected at load
  time so users know the YAML field exists but is not honoured yet.
- No per-task timeout enforcement. ``task.timeout_s`` is accepted into
  the schema (so users won't have to migrate later) and recorded in the
  result row, but not enforced in v0.2.
- No custom Python scorers, no judges. Same reason.

The format is intentionally separate from
:class:`spark_benchmark.suites.SuiteDefinition`, which is the canonical
JSON-only format hard-wired into the curated suites. Custom suites live
in user space, are YAML-friendly, and evolve on a different cadence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from spark_benchmark.models import (
    BackendConfig,
    GenerationResult,
    ModelConfig,
    SamplingConfig,
)
from spark_benchmark.results_bundle import write_json, write_result


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
    """Per-task timeout. **Not enforced in v0.2.0** — recorded for forensics."""


class CustomSuiteDefinition(BaseModel):
    """User-supplied benchmark suite."""

    name: str
    version: str = "0.1"
    description: str = ""
    mode: Literal["quick", "scored"] = "quick"
    """``quick`` is Mode A (pass-through, no scoring). ``scored`` is parked
    for v0.3.0 — it is rejected at load time today so users get a clear
    error instead of an ignored config block."""
    tasks: list[CustomSuiteTask]
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    """Default sampling applied to every task that does not specify its own."""
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
    """
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = _load_yaml(text)
    suite = CustomSuiteDefinition.model_validate(payload)
    if suite.mode != "quick":
        raise ValueError(
            f"custom suite mode {suite.mode!r} is not implemented in v0.2.0; "
            "only 'quick' (pass-through, no scoring) is available today. "
            "See docs/custom-tests-spec.md for the v0.3.0 'scored' roadmap."
        )
    return suite


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
    surface to the user (model name typos, sampling sanity, prompt length).
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
) -> dict[str, Any]:
    """Execute a Mode A custom suite end-to-end.

    For each ``(model, task)`` pair:

    - resolves sampling (per-task > suite default),
    - calls ``backend.generate(prompt, sampling)``,
    - appends one row to ``run_dir/results.jsonl``,
    - on exception, records an ``error`` row and keeps going.

    Returns the same dict that gets serialised to ``summary.json``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cli_default = default_sampling or SamplingConfig()
    skip_pairs = already_completed_pairs(run_dir) if resume else set()
    if skip_pairs and progress_callback:
        progress_callback(f"  resume: skipping {len(skip_pairs)} (model, task) pair(s) already on disk")

    rows: list[dict[str, Any]] = []
    for model_config in model_configs:
        if progress_callback:
            progress_callback(f"  loading {model_config.name}")
        backend.load_model(model_config)
        try:
            for idx, task in enumerate(suite.tasks, start=1):
                key = (model_config.name, task.task_id)
                if key in skip_pairs:
                    if progress_callback:
                        progress_callback(
                            f"  {model_config.name} → [{idx}/{len(suite.tasks)}] {task.task_id} (skip — already done)"
                        )
                    continue
                sampling = _resolve_sampling(task, suite, cli_default)
                if progress_callback:
                    progress_callback(
                        f"  {model_config.name} → [{idx}/{len(suite.tasks)}] {task.task_id}"
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
                }
                try:
                    generation: GenerationResult = backend.generate(task.prompt, sampling)
                    row["generation"] = generation.model_dump(mode="json")
                    row["error"] = None
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
                write_result(run_dir, row)
                rows.append(row)
        finally:
            if progress_callback:
                progress_callback(f"  unloading {model_config.name}")
            backend.unload()

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
    """Side-by-side Markdown summary for Mode A.

    The intent is reading on screen and committing to a wiki: per-model
    telemetry table at the top, then one section per task with each
    model's reply rendered as a fenced block underneath.
    """
    lines: list[str] = []
    lines.append(f"# Custom suite: {summary['suite']}")
    lines.append("")
    lines.append(f"- **Version**: {summary['suite_version']}")
    lines.append(f"- **Mode**: {summary['mode']} (no scoring)")
    if summary.get("description"):
        lines.append(f"- **Description**: {summary['description']}")
    lines.append(f"- **Backend**: {summary['backend']}")
    lines.append(f"- **Tasks**: {summary['task_count']}")
    lines.append("")

    lines.append("## Per-model telemetry")
    lines.append("")
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
        rows = by_task[task_id]
        prompt = rows[0].get("prompt", "")
        lines.append(f"### Task `{task_id}`")
        lines.append("")
        lines.append("**Prompt:**")
        lines.append("")
        lines.append("```")
        lines.append(prompt)
        lines.append("```")
        lines.append("")
        for row in rows:
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
