from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from spark_benchmark.code_generation import (
    default_reference_scores_path,
    load_code_generation_suite,
    run_code_generation_suite,
)
from spark_benchmark.sustained_throughput import (
    load_sustained_throughput_suite,
    run_sustained_throughput_suite,
)
from spark_benchmark.long_context import (
    load_haystack_texts,
    load_long_context_fixture,
    run_long_context_suite,
)
from spark_benchmark.models import BackendConfig, GenerationResult, ModelConfig, SamplingConfig
from spark_benchmark.results_bundle import write_json, write_manifest, write_result
from spark_benchmark.reliability import (
    build_summary,
    run_hallucination_grounding_suite,
    run_practical_structured_output_suite,
)
from spark_benchmark.runtime import build_manifest
from spark_benchmark.suites import SuiteDefinition, SuiteTask, load_suite_definition


@dataclass
class BenchmarkPlan:
    request: str
    selected_models: list[str]
    selected_suites: list[str]
    rationale: list[str]


def normalize_request(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def parse_benchmark_request(request: str, available_models: list[str]) -> BenchmarkPlan:
    normalized = normalize_request(request)
    selected_models = [model for model in available_models if model.lower() in normalized]
    rationale: list[str] = []

    alias_map = {
        "qwen": "qwen-3.6",
        "gemma": "gemma-4",
        "nemotron": "nemotron-3",
    }
    for alias, model in alias_map.items():
        if alias in normalized and model in available_models and model not in selected_models:
            selected_models.append(model)

    if not selected_models:
        selected_models = list(available_models)
        rationale.append("No explicit model names found, so all experiment models were selected.")
    else:
        rationale.append("Selected only models mentioned in the request.")

    selected_suites: list[str] = []
    if any(token in normalized for token in ("rychlost", "speed", "latency", "ttft", "throughput")):
        selected_suites.append("openclaw_speed")
    if any(token in normalized for token in ("spolehliv", "reliab", "hallucin", "ground", "factual")):
        selected_suites.append("hallucination_grounding")
    if any(token in normalized for token in ("openclaw", "json", "structured", "tool", "agent", "workflow", "output")):
        if "openclaw" in normalized and "openclaw_speed" not in selected_suites:
            selected_suites.append("openclaw_speed")
        selected_suites.append("practical_structured_output")
    if any(token in normalized for token in ("code", "kod", "kód", "humaneval", "mbpp", "python", "programovani", "programování")):
        selected_suites.append("code_generation")
    if any(token in normalized for token in ("sustained", "throttle", "thermal", "throttling", "long-run", "dlouhodob", "thermalni", "termaln")):
        selected_suites.append("sustained_throughput")
    if any(token in normalized for token in ("long context", "long-context", "dlouhý kontext", "dlouhy kontext", "needle", "haystack", "niah", "retrieval", "kontextové okno", "kontextove okno")):
        selected_suites.append("long_context_retrieval")

    if not selected_suites:
        selected_suites = ["openclaw_speed", "hallucination_grounding", "practical_structured_output"]
        rationale.append("No explicit suite keywords found, so speed plus both reliability slices were selected.")
    else:
        rationale.append("Selected suites from request keywords.")

    deduped_suites = []
    for suite in selected_suites:
        if suite not in deduped_suites:
            deduped_suites.append(suite)

    return BenchmarkPlan(
        request=request,
        selected_models=selected_models,
        selected_suites=deduped_suites,
        rationale=rationale,
    )


def load_openclaw_speed_suite(repo_root: Path) -> SuiteDefinition:
    return load_suite_definition(repo_root / "data" / "performance" / "openclaw_speed_v1.json")


def run_openclaw_speed_suite(
    *,
    run_dir: Path,
    suite: SuiteDefinition,
    backend: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    sampling: SamplingConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    suite_sampling = sampling.model_copy(update={"max_tokens": min(sampling.max_tokens, 160)})
    run_rows: list[dict[str, Any]] = []
    total_tasks = len(suite.tasks)
    for model_config in model_configs:
        if progress_callback:
            progress_callback(f"  loading {model_config.name} for speed probe")
        backend.load_model(model_config)
        for idx, task in enumerate(suite.tasks, start=1):
            if progress_callback:
                progress_callback(f"  {model_config.name} → speed task {idx}/{total_tasks} ({task.task_id})")
            prompt = task.prompt if not task.context else f"{task.context}\n\n{task.prompt}"
            generation: GenerationResult = backend.generate(prompt, suite_sampling)
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
                "evaluation": {
                    "expected_behavior": "performance_probe",
                    "passed": True,
                    "score": 1,
                    "reason": "captured_metrics",
                    "matched_reference_tokens": [],
                },
            }
            write_result(run_dir, row)
            run_rows.append(row)
        if progress_callback:
            progress_callback(f"  unloading {model_config.name} from Ollama")
        backend.unload()

    summary = build_summary(run_rows, suite, backend_config)
    write_json(run_dir / "summary.json", summary)
    return summary


def run_benchmark_bundle(
    *,
    bundle_dir: Path,
    repo_root: Path,
    experiment: Any,
    platform_config: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    backend: Any,
    plan: BenchmarkPlan,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    write_json(bundle_dir / "plan.json", {"request": plan.request, "rationale": plan.rationale, "suites": plan.selected_suites, "models": plan.selected_models})

    completed: list[dict[str, Any]] = []
    for suite_index, suite_name in enumerate(plan.selected_suites, start=1):
        if progress_callback:
            progress_callback(
                f"[{suite_index}/{len(plan.selected_suites)}] Starting suite '{suite_name}' "
                f"on {len(model_configs)} model(s)"
            )
        suite_dir = bundle_dir / suite_name
        suite_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(
            experiment=experiment,
            platform_config=platform_config,
            backend_config=backend_config,
            model_names=[model.name for model in model_configs],
            results_dir=suite_dir,
        )
        write_manifest(suite_dir, manifest)

        if suite_name == "hallucination_grounding":
            suite = load_suite_definition(repo_root / "data" / "reliability" / "hallucination_grounding_v1.json")
            summary = run_hallucination_grounding_suite(
                run_dir=suite_dir,
                suite=suite,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment.sampling,
                progress_callback=progress_callback,
            )
        elif suite_name == "practical_structured_output":
            suite = load_suite_definition(repo_root / "data" / "practical" / "practical_structured_output_v1.json")
            summary = run_practical_structured_output_suite(
                run_dir=suite_dir,
                suite=suite,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment.sampling,
                progress_callback=progress_callback,
            )
        elif suite_name == "openclaw_speed":
            suite = load_openclaw_speed_suite(repo_root)
            summary = run_openclaw_speed_suite(
                run_dir=suite_dir,
                suite=suite,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment.sampling,
                progress_callback=progress_callback,
            )
        elif suite_name == "code_generation":
            suite = load_code_generation_suite(repo_root)
            summary = run_code_generation_suite(
                run_dir=suite_dir,
                suite=suite,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment.sampling,
                reference_scores_path=default_reference_scores_path(repo_root),
                progress_callback=progress_callback,
            )
        elif suite_name == "sustained_throughput":
            suite = load_sustained_throughput_suite(repo_root)
            summary = run_sustained_throughput_suite(
                run_dir=suite_dir,
                suite=suite,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment.sampling,
                progress_callback=progress_callback,
            )
        elif suite_name in {"long_context_retrieval", "long_context_retrieval_v1"}:
            fixture = load_long_context_fixture(
                repo_root / "data" / "long_context" / "long_context_retrieval_v1.json"
            )
            haystack_texts = load_haystack_texts(fixture, repo_root)
            summary = run_long_context_suite(
                run_dir=suite_dir,
                fixture=fixture,
                haystack_texts=haystack_texts,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment.sampling,
                progress_callback=progress_callback,
            )
        else:
            raise ValueError(f"Unsupported suite: {suite_name}")
        completed.append({"suite": suite_name, "summary": summary})
        if progress_callback:
            progress_callback(f"Finished suite '{suite_name}'.")

    return {"bundle_dir": str(bundle_dir), "completed": completed}
