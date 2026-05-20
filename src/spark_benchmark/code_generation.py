"""Code generation suite: sandboxed execution + pass@k scoring.

The suite runs canonical code benchmarks (HumanEval, MBPP) against a backend.
Generated code is executed in a child subprocess with `resource.setrlimit`
guards and a wall-clock timeout. This module is platform-agnostic; the v1
configuration targets Spark only (see docs/extensions-spec.md).
"""

from __future__ import annotations

import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from spark_benchmark.models import (
    BackendConfig,
    GenerationResult,
    ModelConfig,
    SamplingConfig,
)
from spark_benchmark.results_bundle import write_json, write_result
from spark_benchmark.suites import SuiteDefinition, SuiteTask, load_suite_definition


DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MEMORY_LIMIT_MB = 1024
DEFAULT_TOLERANCE_PP = 3.0

# Python startup needs a fair amount of address space; setting RLIMIT_AS too
# low will SIGKILL normal imports. 1 GB is the lowest safe default.
MIN_SAFE_MEM_LIMIT_MB = 256


@dataclass
class SandboxResult:
    passed: bool
    status: str  # "passed", "failed", "timeout", "oom", "compile_error", "runtime_error"
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None


@dataclass
class SampleOutcome:
    sample_index: int
    seed: int
    extracted_code: str
    sandbox: SandboxResult


@dataclass
class TaskOutcome:
    task_id: str
    benchmark: str
    entry_point: str
    samples: list[SampleOutcome]
    pass_at_1: float
    pass_at_k_value: float | None  # None when only one sample was drawn
    pass_at_k_k: int | None


@dataclass
class ReferenceComparison:
    model: str
    benchmark: str
    metric: str
    observed_pct: float
    expected_pct: float | None
    delta_pp: float | None
    tolerance_pp: float
    enforce: bool
    within_tolerance: bool | None
    note: str = ""


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from the original HumanEval paper.

    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if n < k:
        raise ValueError("n must be >= k")
    if c < 0 or c > n:
        raise ValueError("c must be in [0, n]")
    if n - c < k:
        return 1.0
    # Numerically stable: 1 - prod_{i=0..k-1} (n - c - i) / (n - i)
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return 1.0 - product


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def extract_code(output: str, prompt: str, entry_point: str) -> str:
    """Best-effort extraction of executable code from a model output.

    Strategy:
      1. If the output contains a fenced python block, return its full
         contents — imports and decorators that precede ``def`` must be
         preserved or the sandbox blows up on ``NameError``.
      2. If there is no fence but ``def <entry_point>`` is present, slice
         from that line while keeping any preceding ``import`` / ``from``
         lines so type aliases like ``List`` stay defined.
      3. Otherwise treat the output as a raw continuation and prepend the
         original prompt.
    """
    candidate = output

    fence_match = _CODE_FENCE_RE.search(candidate)
    if fence_match:
        return fence_match.group(1)

    def_marker = f"def {entry_point}"
    def_index = candidate.find(def_marker)
    if def_index != -1:
        before = candidate[:def_index]
        prelude_lines: list[str] = []
        for line in before.splitlines()[::-1]:
            stripped = line.strip()
            if stripped == "" or stripped.startswith(("import ", "from ", "@", "#")):
                prelude_lines.append(line)
                continue
            break
        prelude = "\n".join(reversed(prelude_lines)).rstrip()
        body = candidate[def_index:]
        return f"{prelude}\n{body}" if prelude else body

    # Treat as a continuation of the canonical prompt.
    return prompt + candidate


def _build_program(extracted_code: str, tests: str, prompt: str = "") -> str:
    """Stitch extracted code and canonical tests into a runnable module.

    HumanEval prompts include the imports the function needs (``from typing
    import List``, ...). Chat models often regenerate the ``def`` signature
    without re-emitting those imports. Re-inject any imports from the prompt
    that are missing from the extracted code so the sandbox doesn't trip on
    ``NameError: List``.
    """
    extracted_import_lines = {
        line.strip()
        for line in extracted_code.splitlines()
        if line.lstrip().startswith(("import ", "from "))
    }
    missing: list[str] = []
    seen: set[str] = set()
    for line in (prompt or "").splitlines():
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        if stripped in extracted_import_lines or stripped in seen:
            continue
        missing.append(stripped)
        seen.add(stripped)
    prelude = ("\n".join(missing) + "\n\n") if missing else ""
    return f"{prelude}{extracted_code}\n\n# --- begin canonical tests ---\n{tests}\n"


def _build_preexec(mem_limit_mb: int, cpu_seconds: int) -> Any:
    mem_bytes = max(mem_limit_mb, MIN_SAFE_MEM_LIMIT_MB) * 1024 * 1024

    def _apply_limits() -> None:
        # Address space cap. Catches runaway allocations; allow normal Python startup.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        # CPU time cap as a backstop to wall-clock timeout.
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):
            pass
        # No writing huge files.
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        except (ValueError, OSError):
            pass
        # Start a fresh process group so timeout cleanup hits descendants too.
        try:
            os.setsid()
        except OSError:
            pass

    return _apply_limits


def sandbox_run(
    program: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> SandboxResult:
    """Run `program` in a child Python subprocess with resource limits.

    The child inherits an empty PYTHONPATH and a scratch CWD so accidental
    repo imports cannot influence the test. Network access is not blocked
    here; production deployments should layer Docker or firejail on top.
    """
    with tempfile.TemporaryDirectory(prefix="spark-bench-codegen-") as scratch:
        program_path = Path(scratch) / "program.py"
        program_path.write_text(program, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        cpu_seconds = max(2, int(math.ceil(timeout_s)) + 2)
        preexec = _build_preexec(memory_limit_mb, cpu_seconds)

        import time

        start = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, str(program_path)],
                capture_output=True,
                text=True,
                cwd=scratch,
                env=env,
                timeout=timeout_s,
                preexec_fn=preexec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            return SandboxResult(
                passed=False,
                status="timeout",
                duration_s=duration,
                stdout=(exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=(exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                exit_code=None,
            )

        duration = time.monotonic() - start
        if completed.returncode == 0:
            return SandboxResult(True, "passed", duration, completed.stdout, completed.stderr, 0)

        stderr = completed.stderr or ""
        status = _classify_failure(stderr, completed.returncode)
        return SandboxResult(False, status, duration, completed.stdout or "", stderr, completed.returncode)


_OOM_PATTERNS = ("MemoryError", "Cannot allocate memory")


def _classify_failure(stderr: str, exit_code: int) -> str:
    if any(pattern in stderr for pattern in _OOM_PATTERNS):
        return "oom"
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "compile_error"
    if "AssertionError" in stderr:
        return "failed"
    if exit_code in (-9, 137):
        return "oom"
    if exit_code in (-24, -25):
        return "timeout"
    return "runtime_error"


def evaluate_task(
    task: SuiteTask,
    *,
    generations: list[GenerationResult],
    sample_seeds: list[int],
    sandbox_timeout_s: float,
    sandbox_memory_mb: int,
    k_for_pass_at_k: int | None = None,
) -> TaskOutcome:
    entry_point = str(task.metadata.get("entry_point") or "")
    benchmark = str(task.metadata.get("benchmark") or "unknown")
    tests = str(task.metadata.get("tests") or "")
    if not entry_point or not tests:
        raise ValueError(f"Task {task.task_id} is missing entry_point or tests metadata")

    sample_outcomes: list[SampleOutcome] = []
    correct = 0
    for index, (generation, seed) in enumerate(zip(generations, sample_seeds)):
        extracted = extract_code(generation.output, task.prompt, entry_point)
        program = _build_program(extracted, tests, prompt=task.prompt)
        sandbox = sandbox_run(
            program,
            timeout_s=sandbox_timeout_s,
            memory_limit_mb=sandbox_memory_mb,
        )
        if sandbox.passed:
            correct += 1
        sample_outcomes.append(
            SampleOutcome(
                sample_index=index,
                seed=seed,
                extracted_code=extracted,
                sandbox=sandbox,
            )
        )

    n = len(sample_outcomes)
    pa1 = pass_at_k(n, correct, 1) if n >= 1 else 0.0
    pak_value: float | None = None
    pak_k: int | None = None
    if k_for_pass_at_k is not None and n >= k_for_pass_at_k:
        pak_value = pass_at_k(n, correct, k_for_pass_at_k)
        pak_k = k_for_pass_at_k

    return TaskOutcome(
        task_id=task.task_id,
        benchmark=benchmark,
        entry_point=entry_point,
        samples=sample_outcomes,
        pass_at_1=pa1,
        pass_at_k_value=pak_value,
        pass_at_k_k=pak_k,
    )


def load_reference_scores(path: Path | str) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw) or {}
    return parsed


def validate_reference_scores(
    *,
    per_model_per_benchmark: dict[str, dict[str, float]],
    reference: dict[str, Any],
) -> list[ReferenceComparison]:
    tolerance_pp = float(reference.get("tolerance_pp", DEFAULT_TOLERANCE_PP))
    models_section = reference.get("models", {}) or {}
    comparisons: list[ReferenceComparison] = []

    for model_name, by_benchmark in per_model_per_benchmark.items():
        for benchmark, observed_fraction in by_benchmark.items():
            observed_pct = observed_fraction * 100.0
            model_entry = models_section.get(model_name) or {}
            benchmark_entry = model_entry.get(benchmark) or {}
            expected_pct = benchmark_entry.get("pass_at_1")
            enforce = bool(benchmark_entry.get("enforce", expected_pct is not None))
            note = str(benchmark_entry.get("source", "")) if benchmark_entry else "no reference entry"

            if expected_pct is None:
                comparisons.append(
                    ReferenceComparison(
                        model=model_name,
                        benchmark=benchmark,
                        metric="pass_at_1",
                        observed_pct=observed_pct,
                        expected_pct=None,
                        delta_pp=None,
                        tolerance_pp=tolerance_pp,
                        enforce=False,
                        within_tolerance=None,
                        note=note or "no expected value",
                    )
                )
                continue

            delta_pp = observed_pct - float(expected_pct)
            within = abs(delta_pp) <= tolerance_pp
            comparisons.append(
                ReferenceComparison(
                    model=model_name,
                    benchmark=benchmark,
                    metric="pass_at_1",
                    observed_pct=observed_pct,
                    expected_pct=float(expected_pct),
                    delta_pp=delta_pp,
                    tolerance_pp=tolerance_pp,
                    enforce=enforce,
                    within_tolerance=within,
                    note=note,
                )
            )

    return comparisons


def _generations_for_task(
    *,
    backend: Any,
    task: SuiteTask,
    sampling: SamplingConfig,
    seeds: list[int],
    pass_at_1_overrides: dict[str, Any],
    pass_at_k_overrides: dict[str, Any],
) -> tuple[list[GenerationResult], list[int]]:
    """Draw all samples for one task.

    The first sample uses pass@1 sampling (typically T=0, single shot). Any
    further samples use pass@k sampling (typically T=0.8) with the supplied
    seed sequence.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")

    generations: list[GenerationResult] = []
    used_seeds: list[int] = []

    pass_at_1_sampling = sampling.model_copy(update={**pass_at_1_overrides, "seed": seeds[0]})
    generations.append(backend.generate(task.prompt, pass_at_1_sampling))
    used_seeds.append(seeds[0])

    for seed in seeds[1:]:
        pass_at_k_sampling = sampling.model_copy(update={**pass_at_k_overrides, "seed": seed})
        generations.append(backend.generate(task.prompt, pass_at_k_sampling))
        used_seeds.append(seed)

    return generations, used_seeds


def _aggregate_per_model(outcomes: list[TaskOutcome]) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        bucket = by_benchmark.setdefault(
            outcome.benchmark,
            {"benchmark": outcome.benchmark, "tasks": 0, "pass_at_1_sum": 0.0, "pass_at_k_sum": 0.0, "pass_at_k_count": 0, "pass_at_k_k": None, "failed_task_ids": []},
        )
        bucket["tasks"] += 1
        bucket["pass_at_1_sum"] += outcome.pass_at_1
        if outcome.pass_at_1 < 1.0:
            bucket["failed_task_ids"].append(outcome.task_id)
        if outcome.pass_at_k_value is not None:
            bucket["pass_at_k_sum"] += outcome.pass_at_k_value
            bucket["pass_at_k_count"] += 1
            bucket["pass_at_k_k"] = outcome.pass_at_k_k

    for bucket in by_benchmark.values():
        n = max(bucket["tasks"], 1)
        bucket["pass_at_1"] = bucket["pass_at_1_sum"] / n
        del bucket["pass_at_1_sum"]
        if bucket["pass_at_k_count"]:
            bucket["pass_at_k"] = bucket["pass_at_k_sum"] / bucket["pass_at_k_count"]
        else:
            bucket["pass_at_k"] = None
        del bucket["pass_at_k_sum"]
    return by_benchmark


def write_summary_markdown(
    run_dir: Path, summary: dict[str, Any], comparisons: list[ReferenceComparison]
) -> Path:
    lines = [
        f"# {summary['suite']} summary",
        "",
        f"- backend: {summary['backend']}",
        f"- total samples: {summary['total_samples']}",
        f"- total tasks: {summary['total_tasks']}",
        "",
        "## Pass rates",
        "",
        "| model | benchmark | tasks | pass@1 | pass@k |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model in summary["models"]:
        for bench in model["benchmarks"]:
            pak = bench.get("pass_at_k")
            pak_cell = f"{pak:.2%} (k={bench.get('pass_at_k_k')})" if pak is not None else "-"
            lines.append(
                f"| {model['model']} | {bench['benchmark']} | {bench['tasks']} | "
                f"{bench['pass_at_1']:.2%} | {pak_cell} |"
            )

    if comparisons:
        lines.extend(["", "## Reference-score validation", ""])
        lines.append("| model | benchmark | observed | expected | delta_pp | within ±tol | source |")
        lines.append("| --- | --- | ---: | ---: | ---: | :---: | --- |")
        for cmp in comparisons:
            observed = f"{cmp.observed_pct:.2f}"
            expected = f"{cmp.expected_pct:.2f}" if cmp.expected_pct is not None else "-"
            delta = f"{cmp.delta_pp:+.2f}" if cmp.delta_pp is not None else "-"
            within = "n/a" if cmp.within_tolerance is None else ("yes" if cmp.within_tolerance else "NO")
            lines.append(
                f"| {cmp.model} | {cmp.benchmark} | {observed} | {expected} | {delta} | {within} | {cmp.note} |"
            )

    path = run_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_code_generation_suite(
    *,
    run_dir: Path,
    suite: SuiteDefinition,
    backend: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    sampling: SamplingConfig,
    num_samples_per_task: int = 1,
    pass_at_k_value: int | None = None,
    sandbox_timeout_s: float = DEFAULT_TIMEOUT_S,
    sandbox_memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    pass_at_1_overrides: dict[str, Any] | None = None,
    pass_at_k_overrides: dict[str, Any] | None = None,
    seeds: list[int] | None = None,
    reference_scores_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the code generation suite against all models.

    Args:
      num_samples_per_task: total samples per task. pass@1 uses 1; pass@10
        uses 10. Defaults to 1 (v1 default).
      pass_at_k_value: the `k` for pass@k aggregation. If None and
        num_samples_per_task > 1, pass@k is reported with k=num_samples.
      pass_at_1_overrides: sampling overrides for the first sample
        (default: temperature=0, top_p=1.0).
      pass_at_k_overrides: sampling overrides for subsequent samples
        (default: temperature=0.8, top_p=0.95).
      seeds: list of seeds with length >= num_samples_per_task. Defaults to
        [42, 43, ...].
    """
    if num_samples_per_task < 1:
        raise ValueError("num_samples_per_task must be >= 1")

    pass_at_1_overrides = pass_at_1_overrides or {"temperature": 0.0, "top_p": 1.0}
    pass_at_k_overrides = pass_at_k_overrides or {"temperature": 0.8, "top_p": 0.95}
    seeds = seeds or [42 + i for i in range(num_samples_per_task)]
    if len(seeds) < num_samples_per_task:
        raise ValueError("seeds must have at least num_samples_per_task entries")

    effective_k = pass_at_k_value if pass_at_k_value is not None else (
        num_samples_per_task if num_samples_per_task > 1 else None
    )

    per_model_outcomes: dict[str, list[TaskOutcome]] = {}
    total_samples = 0

    total_tasks = len(suite.tasks)
    for model_config in model_configs:
        if progress_callback:
            progress_callback(f"  loading {model_config.name} for code generation")
        backend.load_model(model_config)
        outcomes: list[TaskOutcome] = []
        for idx, task in enumerate(suite.tasks, start=1):
            if progress_callback:
                progress_callback(
                    f"  {model_config.name} → coding task {idx}/{total_tasks} ({task.task_id})"
                )
            generations, used_seeds = _generations_for_task(
                backend=backend,
                task=task,
                sampling=sampling,
                seeds=seeds[:num_samples_per_task],
                pass_at_1_overrides=pass_at_1_overrides,
                pass_at_k_overrides=pass_at_k_overrides,
            )
            total_samples += len(generations)
            outcome = evaluate_task(
                task,
                generations=generations,
                sample_seeds=used_seeds,
                sandbox_timeout_s=sandbox_timeout_s,
                sandbox_memory_mb=sandbox_memory_mb,
                k_for_pass_at_k=effective_k,
            )
            outcomes.append(outcome)

            row = {
                "suite": suite.name,
                "suite_version": suite.version,
                "model": model_config.name,
                "model_tag": model_config.artifact_path or model_config.revision,
                "task_id": task.task_id,
                "tags": task.tags,
                "prompt": task.prompt,
                "context": task.context,
                "reference": task.reference,
                "samples": [
                    {
                        "sample_index": sample.sample_index,
                        "seed": sample.seed,
                        "extracted_code": sample.extracted_code,
                        "generation": gen.model_dump(mode="json"),
                        "sandbox": {
                            "passed": sample.sandbox.passed,
                            "status": sample.sandbox.status,
                            "duration_s": sample.sandbox.duration_s,
                            "exit_code": sample.sandbox.exit_code,
                            "stderr": sample.sandbox.stderr[-2000:],
                        },
                    }
                    for sample, gen in zip(outcome.samples, generations)
                ],
                "evaluation": {
                    "benchmark": outcome.benchmark,
                    "pass_at_1": outcome.pass_at_1,
                    "pass_at_k": outcome.pass_at_k_value,
                    "pass_at_k_k": outcome.pass_at_k_k,
                    "passed": outcome.pass_at_1 >= 1.0,
                    "score": 1 if outcome.pass_at_1 >= 1.0 else 0,
                    "reason": "all_samples_passed" if outcome.pass_at_1 >= 1.0 else "at_least_one_sample_failed",
                    "matched_reference_tokens": [],
                },
            }
            write_result(run_dir, row)
        if progress_callback:
            passed = sum(1 for o in outcomes if o.pass_at_1 >= 1.0)
            progress_callback(
                f"  {model_config.name} finished coding suite — {passed}/{total_tasks} passed"
            )
            progress_callback(f"  unloading {model_config.name} from Ollama")
        backend.unload()
        per_model_outcomes[model_config.name] = outcomes

    model_summaries: list[dict[str, Any]] = []
    pass1_by_model_benchmark: dict[str, dict[str, float]] = {}
    for model_name, outcomes in per_model_outcomes.items():
        per_benchmark = _aggregate_per_model(outcomes)
        for bench_name, bucket in per_benchmark.items():
            pass1_by_model_benchmark.setdefault(model_name, {})[bench_name] = bucket["pass_at_1"]
        model_summaries.append(
            {
                "model": model_name,
                "benchmarks": list(per_benchmark.values()),
            }
        )

    comparisons: list[ReferenceComparison] = []
    if reference_scores_path is not None and reference_scores_path.exists():
        reference = load_reference_scores(reference_scores_path)
        comparisons = validate_reference_scores(
            per_model_per_benchmark=pass1_by_model_benchmark,
            reference=reference,
        )

    summary = {
        "suite": suite.name,
        "suite_version": suite.version,
        "backend": backend_config.name.value,
        "total_samples": total_samples,
        "total_tasks": sum(len(o) for o in per_model_outcomes.values()),
        "models": model_summaries,
        "reference_warnings": [
            {
                "model": cmp.model,
                "benchmark": cmp.benchmark,
                "observed_pct": cmp.observed_pct,
                "expected_pct": cmp.expected_pct,
                "delta_pp": cmp.delta_pp,
                "tolerance_pp": cmp.tolerance_pp,
                "source": cmp.note,
            }
            for cmp in comparisons
            if cmp.enforce and cmp.within_tolerance is False
        ],
    }
    write_json(run_dir / "summary.json", summary)
    write_summary_markdown(run_dir, summary, comparisons)
    return summary


def load_code_generation_suite(repo_root: Path) -> SuiteDefinition:
    return load_suite_definition(repo_root / "data" / "code" / "code_generation_v1.json")


def default_reference_scores_path(repo_root: Path) -> Path:
    return repo_root / "data" / "code" / "reference_scores.yaml"
