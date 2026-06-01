from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from spark_benchmark.models import BackendConfig, GenerationResult, ModelConfig, SamplingConfig
from spark_benchmark.results_bundle import write_json, write_result
from spark_benchmark.suites import SuiteDefinition, SuiteTask, load_suite_definition

ABSTAIN_PHRASES = (
    "does not mention",
    "not mentioned",
    "does not contain",
    "does not say",
    "doesn't say",
    "not provided",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "do not know",
    "don't know",
    "insufficient information",
    "not enough information",
    "unknown",
    "not answerable",
    "unable to answer from the context",
    "cannot answer from the context",
)

NEGATION_PHRASES = ("no", "not", "incorrect", "inconsistent", "false")


def fixture_path_for_suite_name(repo_root: Path, suite_name: str) -> Path:
    if suite_name in {"hallucination_grounding", "hallucination_grounding_v1"}:
        return repo_root / "data" / "reliability" / "hallucination_grounding_v1.json"
    if suite_name in {"practical_structured_output", "practical_structured_output_v1"}:
        return repo_root / "data" / "practical" / "practical_structured_output_v1.json"
    if suite_name in {"openclaw_speed", "openclaw_speed_v1"}:
        return repo_root / "data" / "performance" / "openclaw_speed_v1.json"
    if suite_name in {"code_generation", "code_generation_v1"}:
        return repo_root / "data" / "code" / "code_generation_v1.json"
    if suite_name in {"sustained_throughput", "sustained_throughput_v1"}:
        return repo_root / "data" / "performance" / "sustained_throughput_v1.json"
    if suite_name in {"long_context_retrieval", "long_context_retrieval_v1"}:
        return repo_root / "data" / "long_context" / "long_context_retrieval_v1.json"
    raise ValueError(f"unsupported suite runner: {suite_name}")


def load_reliability_suite(repo_root: Path, suite_name: str) -> SuiteDefinition:
    return load_suite_definition(fixture_path_for_suite_name(repo_root, suite_name))


def build_grounded_prompt(task: SuiteTask) -> str:
    context = task.context or ""
    return (
        "You must answer using only the supplied context. "
        "If the answer is not in the context, say that clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{task.prompt}"
    )


def build_structured_output_prompt(task: SuiteTask) -> str:
    context = task.context or ""
    return (
        "Return valid JSON only. Do not include markdown fences, prose, or explanations.\n\n"
        f"Input:\n{context}\n\n"
        f"Instruction:\n{task.prompt}"
    )


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def reference_tokens(reference: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", normalize_text(reference)) if len(token) >= 3 or token.isdigit()}


def score_hallucination_task(task: SuiteTask, output: str) -> dict[str, Any]:
    behavior = str(task.metadata.get("expected_behavior") or "").strip()
    normalized_output = normalize_text(output)
    normalized_reference = normalize_text(task.reference or "")
    ref_tokens = reference_tokens(task.reference or "")
    matched_tokens = sorted(token for token in ref_tokens if token in normalized_output)

    passed = False
    reason = "unsupported_behavior"

    if behavior == "answer_from_context":
        passed = bool(normalized_reference and normalized_reference in normalized_output) or bool(matched_tokens)
        reason = "reference_or_token_match" if passed else "missing_reference_signal"
    elif behavior == "abstain":
        matched_phrase = next((phrase for phrase in ABSTAIN_PHRASES if phrase in normalized_output), None)
        passed = matched_phrase is not None
        reason = f"abstention_phrase:{matched_phrase}" if passed else "missing_abstention_phrase"
    elif behavior == "correct_user":
        has_negation = any(phrase in normalized_output for phrase in NEGATION_PHRASES)
        has_reference = bool(normalized_reference and normalized_reference in normalized_output) or bool(matched_tokens)
        passed = has_negation and has_reference
        reason = "negation_plus_reference" if passed else "missing_negation_or_reference"

    return {
        "expected_behavior": behavior,
        "passed": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "matched_reference_tokens": matched_tokens,
    }


def extract_json_value(output: str) -> tuple[Any | None, str]:
    cleaned = output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^`{3}(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*`{3}$", "", cleaned)
        cleaned = cleaned.strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        trailing = cleaned[index + end :].strip()
        if trailing:
            return None, "trailing_text_after_json"
        return value, "ok"
    return None, "no_json_object_found"


def score_structured_output_task(task: SuiteTask, output: str) -> dict[str, Any]:
    expected_behavior = str(task.metadata.get("expected_behavior") or "").strip()
    if expected_behavior != "json_exact_match":
        return {
            "expected_behavior": expected_behavior,
            "passed": False,
            "score": 0,
            "reason": "unsupported_behavior",
            "matched_reference_tokens": [],
        }

    expected = json.loads(task.reference or "null")
    parsed, parse_reason = extract_json_value(output)
    passed = parsed == expected
    reason = "exact_json_match" if passed else parse_reason if parsed is None else "json_value_mismatch"
    return {
        "expected_behavior": expected_behavior,
        "passed": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "matched_reference_tokens": [],
    }


def build_summary(run_rows: list[dict[str, Any]], suite: SuiteDefinition, backend: BackendConfig) -> dict[str, Any]:
    per_model: dict[str, dict[str, Any]] = {}
    for row in run_rows:
        model_name = row["model"]
        bucket = per_model.setdefault(
            model_name,
            {"model": model_name, "passes": 0, "total": 0, "failed_task_ids": []},
        )
        bucket["passes"] += int(row["evaluation"]["score"])
        bucket["total"] += 1
        if not row["evaluation"]["passed"]:
            bucket["failed_task_ids"].append(row["task_id"])
    for bucket in per_model.values():
        total = bucket["total"] or 1
        bucket["pass_rate"] = round(bucket["passes"] / total, 4)
    return {
        "suite": suite.name,
        "suite_version": suite.version,
        "backend": backend.name.value,
        "total_rows": len(run_rows),
        "models": list(per_model.values()),
    }


def write_summary_markdown(run_dir: Path, summary: dict[str, Any]) -> Path:
    lines = [
        f"# {summary['suite']} summary",
        "",
        f"- backend: {summary['backend']}",
        f"- total rows: {summary['total_rows']}",
        "",
        "| model | passes | total | pass_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in summary["models"]:
        lines.append(f"| {model['model']} | {model['passes']} | {model['total']} | {model['pass_rate']:.2%} |")
        if model["failed_task_ids"]:
            failed = ", ".join(model["failed_task_ids"])
            lines.append(f"| {model['model']} failed tasks | {failed} |  |  |")
    path = run_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_hallucination_grounding_suite(
    *,
    run_dir: Path,
    suite: SuiteDefinition,
    backend: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    sampling: SamplingConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    suite_sampling = sampling.model_copy(update={"max_tokens": min(sampling.max_tokens, 64)})
    run_rows: list[dict[str, Any]] = []
    total_tasks = len(suite.tasks)
    for model_config in model_configs:
        if progress_callback:
            progress_callback(f"  loading {model_config.name} for grounding probe")
        backend.load_model(model_config)
        for idx, task in enumerate(suite.tasks, start=1):
            if progress_callback:
                progress_callback(
                    f"  {model_config.name} → grounding task {idx}/{total_tasks} ({task.task_id})"
                )
            prompt = build_grounded_prompt(task)
            generation: GenerationResult = backend.generate(prompt, suite_sampling)
            evaluation = score_hallucination_task(task, generation.output)
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
                "generation": generation.model_dump(mode="json"),
                "evaluation": evaluation,
            }
            write_result(run_dir, row)
            run_rows.append(row)
        if progress_callback:
            progress_callback(f"  unloading {model_config.name} from Ollama")
        backend.unload()

    summary = build_summary(run_rows, suite, backend_config)
    write_json(run_dir / "summary.json", summary)
    write_summary_markdown(run_dir, summary)
    return summary


def run_practical_structured_output_suite(
    *,
    run_dir: Path,
    suite: SuiteDefinition,
    backend: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    sampling: SamplingConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    suite_sampling = sampling.model_copy(update={"max_tokens": min(sampling.max_tokens, 256)})
    run_rows: list[dict[str, Any]] = []
    total_tasks = len(suite.tasks)
    for model_config in model_configs:
        if progress_callback:
            progress_callback(f"  loading {model_config.name} for structured-output probe")
        backend.load_model(model_config)
        for idx, task in enumerate(suite.tasks, start=1):
            if progress_callback:
                progress_callback(
                    f"  {model_config.name} → structured task {idx}/{total_tasks} ({task.task_id})"
                )
            prompt = build_structured_output_prompt(task)
            generation: GenerationResult = backend.generate(prompt, suite_sampling)
            evaluation = score_structured_output_task(task, generation.output)
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
                "generation": generation.model_dump(mode="json"),
                "evaluation": evaluation,
            }
            write_result(run_dir, row)
            run_rows.append(row)
        if progress_callback:
            progress_callback(f"  unloading {model_config.name} from Ollama")
        backend.unload()

    summary = build_summary(run_rows, suite, backend_config)
    write_json(run_dir / "summary.json", summary)
    write_summary_markdown(run_dir, summary)
    return summary
