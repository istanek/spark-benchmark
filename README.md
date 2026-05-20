# spark-benchmark

[![pipeline status](https://gitlab.com/istanek/spark-benchmark/badges/main/pipeline.svg)](https://gitlab.com/istanek/spark-benchmark/-/commits/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)

Reproducible local LLM benchmark harness for evaluating model behavior on NVIDIA DGX Spark.

## v1 focus

Version 1 is intentionally Spark-only. The goal is to compare model variants on the same machine.

Headline priorities for v1:

- reliable, re-runnable experiment definitions from YAML
- Spark-native backend coverage, starting with shared and native backends
- classical benchmark signals plus practical reliability and hallucination checks
- public-ready raw outputs, methodology, and reports

Initial v1 lineup:

- qwen-3.6
- gemma-4
- nemotron-3

## Quick start

Interactive console:

- `cd ~/.openclaw/workspace/spark-benchmark`
- `PYTHONPATH=src python3 -m spark_benchmark.cli console --experiment configs/experiments/spark-ollama-baseline.yaml --platform spark`

Optional model override:

- `PYTHONPATH=src python3 -m spark_benchmark.cli console --experiment configs/experiments/spark-ollama-baseline.yaml --platform spark --model gemma-4`

Natural-language benchmark orchestration:

- `PYTHONPATH=src python3 -m spark_benchmark.cli benchmark otestuj qwen gemma nemotron zamer se na rychlost spolehlivost a openclaw structured output --experiment configs/experiments/spark-ollama-baseline.yaml --platform spark`

Interactive benchmark wizard:

- `PYTHONPATH=src python3 -m spark_benchmark.cli wizard --experiment configs/experiments/spark-ollama-baseline.yaml --platform spark`
- Use arrow keys to move, `Space` to toggle a model or suite, and `Enter` to continue.

## Current scaffold includes

- repository structure
- validated YAML config loading via Pydantic
- CLI with run, aggregate, report, and dashboard commands
- backend and telemetry base interfaces
- Spark-only sample experiment, platform, backend, and model configs
- first working reliability suite runner: `spark-bench run --experiment configs/experiments/spark-ollama-baseline.yaml --platform spark --run-suite hallucination_grounding` loads `data/reliability/hallucination_grounding_v1.json`, runs every task against every configured model, writes one row per (model, task) to `results.jsonl`, and emits `summary.json` + `summary.md` with per-model pass rates using simple heuristics for `answer_from_context`, `abstain`, and `correct_user`
- code generation suite (`--run-suite code_generation`): canonical HumanEval-style problems with sandboxed execution (`subprocess` + `resource.setrlimit` + timeout) and pass@k unbiased estimator; reference-score validator at `data/code/reference_scores.yaml` emits warnings when results drift from published baselines. See `docs/extensions-spec.md` for the full long-context / sustained-throughput / code-generation extension plan
- placeholder suite structure for quality, performance, reliability, and practical task checks

## Tests overview

All tests are plain-python — runnable both as `pytest tests/` and as
`python3 tests/test_<name>.py` (every file has a `_run_all()` fallback).

| File | What it covers |
| --- | --- |
| `tests/test_config_loading.py` | YAML → Pydantic loader for experiment configs |
| `tests/test_backend_registry.py` | `build_backend(BackendConfig)` dispatches the correct adapter (`llamacpp` → `LlamaCppAdapter`, etc.) |
| `tests/test_orchestration.py` | Natural-language `parse_benchmark_request` — default selections plus Czech/English keyword + alias routing (`qwen` → `qwen-3.6`, `rychlost` → `openclaw_speed`, `spolehliv` → `hallucination_grounding`, …) |
| `tests/test_reliability.py` | Reliability fixture loading + scoring for the three `expected_behavior` flags (`answer_from_context`, `abstain`, `correct_user`); JSON exact-match scorer including trailing-text rejection; `build_summary` per-model aggregation |
| `tests/test_code_generation.py` | `pass@k` unbiased estimator (edge cases + bad input); code extraction (markdown fence, inline `def`, raw continuation); sandboxed `subprocess + setrlimit` runs (pass / assertion failure / syntax error / wall-clock timeout); every fixture's `canonical_solution` actually passes the sandbox; reference-score validation (within / outside tolerance, missing expected) |
| `tests/test_shell.py` | Curses-shell classifier — vision / embedding model detection, auto-synthesis of `ModelConfig` for unknown Ollama tags, `SUITE_REGISTRY` ↔ fixture wiring sanity |
| `tests/test_sustained_throughput.py` | `compute_windows` slices generations into wall-clock buckets; `compute_derived_metrics` reports `initial / sustained / peak tokens_per_s`, `throttle_ratio`, `time_to_throttle_s`, `avg_power_w`, `peak_temp_c`, `energy_j_per_token` |

Run them all:

```bash
PYTHONPATH=src pytest tests/                       # preferred
PYTHONPATH=src python3 tests/test_reliability.py   # one file, no pytest needed
```

## Planned v1 suite mix

- quality: conventional evals and correctness-oriented tasks
- performance: throughput, TTFT, context scaling, sustained generation
- reliability: hallucination probes, unsupported-claim handling, abstention behavior
- practical: tool-like structured outputs and real-world task outcomes

This is still an early Phase 1 implementation, not a full benchmark implementation yet.
