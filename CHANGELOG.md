# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Bring-Your-Own-Test (BYOT) subsystem — Mode A.** New
  `spark_benchmark.custom_suites` module with a YAML / JSON suite
  format (`CustomSuiteDefinition`), a Pydantic-validated loader,
  resume-friendly runner that records errors per `(model, task)` pair
  without aborting, side-by-side Markdown summary, and a
  ``slugify_suite_name`` helper for run-bundle naming.
- **CLI commands `spark-bench run-custom` and `spark-bench validate-custom`.**
  `run-custom` defaults to ``--allow-auto-detected`` ON (custom suites
  exist precisely for non-curated workloads) and writes its bundles to
  ``results/custom/<slug>/<run-id>/`` with a manifest tagged
  ``kind: custom`` so reporting can keep these visually distinct from
  canonical suites. ``validate-custom`` exits non-zero on any error
  issue (duplicate task IDs, empty prompts, ``mode: scored`` not yet
  implemented, unknown model references).
- **Example custom suite template** at ``examples/custom-tests/quick/``
  with a working ``suite.yaml`` (Czech idiom translation, JSON
  extraction, Python code review) and a ``README.md`` explaining how to
  copy, edit, and run it.
- **Spec doc** ``docs/custom-tests-spec.md`` covering the v0.2.0 cut
  (``mode: quick`` only) plus the explicit roadmap for v0.3.0
  (deterministic scorers + ``dry-run``), v0.4.0 (sandboxed custom
  Python scorers + per-task timeout enforcement), v0.5.0 (local-only
  LLM-as-judge), and v0.6.0+ (sharing).
- **`tests/test_custom_suites.py`** — 16 plain-Python tests covering
  schema validation (duplicate IDs, empty prompts, empty tasks),
  YAML / JSON loaders, soft validation (long prompts, bad sampling,
  unknown model refs), end-to-end runner including resume + error
  recording, summary aggregation, Markdown rendering, and the
  ``slugify_suite_name`` helper.
- **Shared model registry** (`spark_benchmark.model_registry`) extracted
  from `shell.py`. One classification path is now used by the curses TUI,
  the wizard, the console REPL, the natural-language `benchmark` command,
  and the plain `run` command.
- **`--allow-auto-detected` flag** on `run`, `console`, `benchmark`, and
  `wizard`. When set, every chat-capable Ollama tag is offered alongside
  the curated experiment lineup, with auto-synthesized `ModelConfig`s
  carrying `notes=["auto-detected from Ollama (no YAML config)"]`. Off by
  default to preserve reproducibility for `run` and the NL routers.
- **`console --model` accepts Ollama tags directly** (`--model phi4:14b`)
  via `find_config_by_name_or_tag`, in addition to slugified
  (`phi4-14b`) and curated experiment names.
- **`tests/test_model_registry.py`** — coverage for `slugify_tag`,
  `synthesize_model_config`, `classify_detected`, the new
  `resolve_runnable_models` resolver (default + auto-detect + collision),
  and `find_config_by_name_or_tag` resolution order.

### Changed

- Curses TUI no longer owns its own classifier; it delegates to
  `model_registry.classify_detected`. The `classify_models(ctx, detected)`
  shape is preserved for backwards compatibility.
- `cli.detect_ollama_model_tags` is now a thin wrapper over the shared
  `detect_ollama_models`. The duplicate URL/JSON parsing logic is gone.

## [0.1.0] - 2026-05-20

Initial public release of the spark-benchmark scaffold.

### Added

- **Core harness**
  - YAML-driven experiment definitions validated through Pydantic v2
    (`ExperimentSpec`, `PlatformConfig`, `BackendConfig`, `ModelConfig`,
    `SamplingConfig`).
  - Typer CLI (`spark-bench`, `spark-benchmark`) with `run`, `console`,
    `benchmark`, `wizard`, `aggregate`, `report`, `dashboard`, `shell`
    subcommands.
  - Curses TUI shell with model / suite multiselect, chat mode, log
    follower, and live progress callbacks.
  - Run bundle layout: `results/benchmarks/<run-id>/<suite>/` containing
    `manifest.json`, `results.jsonl`, `summary.json`, `summary.md`.
- **Backend adapters**
  - Ollama HTTP adapter (production path for v1).
  - llama.cpp subprocess adapter (`llama-cli`).
  - Stub adapter used as fallback for `trt-llm` / `vllm`.
- **Suite runners**
  - `openclaw_speed` — TTFT and decode probe on short OpenClaw-like
    prompts (no quality scoring).
  - `hallucination_grounding` — grounded answers vs. abstention vs. false
    premise correction; heuristic scorer with abstention-phrase / negation
    / token-overlap rules.
  - `practical_structured_output` — exact-match JSON evaluation with
    fenced-block extraction and trailing-text rejection.
  - `code_generation` — HumanEval starter subset with sandboxed execution
    (`subprocess + resource.setrlimit + timeout`), `pass@k` unbiased
    estimator, and reference-score validation against
    `data/code/reference_scores.yaml`.
  - `sustained_throughput` — 5-minute decode soak per model with NVML or
    `nvidia-smi` telemetry, per-window aggregation, throttle ratio,
    energy per token.
- **Reporting**
  - Aggregator (`aggregate_runs`) joining manifests, summaries, and JSONL
    rows by suite × model.
  - Markdown / HTML / CLI summary renderers with overall ranking,
    per-suite commentary, and verdict paragraph.
- **Natural-language orchestration**
  - `parse_benchmark_request` — keyword + alias router that understands
    Czech and English (`rychlost`/`speed`, `spolehliv`/`reliab`,
    `kod`/`code`, `dlouhodob`/`sustained`, `openclaw`).
- **Fixtures (v1 starter sets)**
  - `data/reliability/hallucination_grounding_v1.json` — 9 tasks across
    `answer_from_context`, `abstain`, `correct_user`.
  - `data/practical/practical_structured_output_v1.json` — 6 JSON
    exact-match scenarios.
  - `data/performance/openclaw_speed_v1.json` — 3 short prompts for
    latency / throughput probing.
  - `data/performance/sustained_throughput_v1.json` — 3 long-form
    prompts cycled during decode soak.
  - `data/code/code_generation_v1.json` — 5 canonical HumanEval problems
    plus `data/code/reference_scores.yaml` template.
- **Configs**
  - `configs/experiments/`: `spark-ollama-baseline`,
    `spark-llamacpp-baseline`, `spark-code-generation`, `spark-sustained`,
    `spark-trt-reliability`.
  - `configs/models/`: `qwen-3.6`, `gemma-4`, `nemotron-3` plus a
    tombstone for the retired `nemotron-3-super`.
  - `configs/backends/`: `ollama`, `llamacpp`, `trt-llm`.
  - `configs/platforms/spark.yaml`.
- **Tests** — plain-python, runnable via `pytest tests/` or `python3
  tests/test_<name>.py` (each file has a `_run_all()` fallback).
- **Docs** — `README.md`, `METHODOLOGY.md`, `docs/architecture.md`,
  `docs/extensions-spec.md`, `CONTRIBUTING.md`.
- **Cursor rules** — `.cursor/rules/{project-overview,python-conventions,
  fixtures-and-configs}.mdc`.
- **GitLab CI** — `.gitlab-ci.yml` with a YAML / JSON fixture lint stage
  and a pytest stage.

### Known limitations

- Reference scores in `data/code/reference_scores.yaml` are placeholders
  with `enforce: false`; populate with model-card numbers before relying
  on the warning system.
- Long-context retrieval (NIAH) suite is specified in
  `docs/extensions-spec.md` but not yet implemented.
- TRT-LLM and vLLM backends fall through to the stub adapter.

[Unreleased]: https://gitlab.com/istanek/spark-benchmark/-/compare/v0.1.0...HEAD
[0.1.0]: https://gitlab.com/istanek/spark-benchmark/-/tags/v0.1.0
