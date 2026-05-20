# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
