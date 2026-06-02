# spark-benchmark

[![CI](https://github.com/istanek/spark-benchmark/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/istanek/spark-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)

> The plain-language version of this document is `../README.txt`. This
> markdown copy lives under `docs/` so the project landing page on
> GitHub renders the plain-text README instead.

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

```bash
cd ~/.openclaw/workspace/spark-benchmark
pip install -e .

# Easiest: launch the full TUI (no flags, picks defaults)
spark-bench

# Or invoke a specific subcommand explicitly
PYTHONPATH=src python3 -m spark_benchmark.cli wizard \
  --experiment configs/experiments/spark-ollama-baseline.yaml --platform spark
```

### Ollama Cloud

Point the Ollama backend at [Ollama Cloud](https://ollama.com) with two env
vars (no config edits) and select a cloud model by tag:

```bash
export OLLAMA_HOST=https://ollama.com
export OLLAMA_API_KEY=sk-...          # https://ollama.com/settings/keys

# ad-hoc one-prompt comparison
spark-bench quick "Summarize the CAP theorem." --models gpt-oss:120b-cloud

# a built-in suite against a specific cloud model (--model is repeatable)
spark-bench run --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark --run-suite hallucination_grounding --model gpt-oss:120b-cloud
```

`OLLAMA_HOST` redirects every request; `OLLAMA_API_KEY` is sent as a Bearer
token and is read from the environment only (never persisted to configs,
manifests, or reports). `--model` accepts an explicit `-cloud` tag even when
it isn't in the experiment YAML or `/api/tags`. Valid `--run-suite` values:
`hallucination_grounding`, `practical_structured_output`, `code_generation`,
`sustained_throughput`, `long_context_retrieval` (+ `_fast`). Cloud runs
report speed/quality but no local GPU telemetry (memory/power/temperature are
unavailable remotely), and calls are billed over the network.

## Interactive CLI

The harness ships with three interactive modes plus one natural-language
batch entrypoint and one bring-your-own-test entrypoint. They all
resolve the same YAML experiment + platform + backend context, so they
share models, suites, sampling, and reporting.

### `spark-bench shell` — full curses TUI

Launches a full-screen menu (`Run / Custom / Quick / Models / Suites /
Info / Chat / Refresh / Quit`) that lets you:

- live-detect models from a running Ollama (`/api/tags`), match them
  against the experiment's `configs/models/*.yaml`, and grey out
  vision / embedding tags so they can't be benchmarked by accident;
- **also offer non-curated tags** — every chat-capable Ollama model that
  doesn't have a YAML config is auto-synthesized into a runnable
  `ModelConfig` (Ollama defaults) and labeled `auto-detected` in the
  picker. The same logic is shared with `wizard` / `console` /
  `benchmark` / `run` via `--allow-auto-detected`;
- multiselect models and suites for a canonical benchmark bundle
  (`Run`), then watch per-model / per-task progress in a scrolling
  log pane;
- run **your own (BYOT) custom suites** without typing flags
  (`Custom`, since 0.2.1) — discovers `examples/custom-tests/**/suite.yaml`
  plus any suite you've already run once via
  `results/custom/<slug>/<run-id>/manifest.json`, single-selects one,
  validates it, multi-selects models, and writes a fresh run bundle
  to `results/custom/<slug>/<run-id>/` with `source: shell` in the
  manifest;
- run **one ad-hoc prompt against every model** (`Quick`, since 0.2.2) —
  multi-selects models, drops out of curses to read a single-line
  prompt on the regular TTY, fans the prompt out via the same
  `run_custom_suite_quick` runner, then asks "save this prompt as a
  reusable custom suite?" so the next time it appears in the
  `Custom` discovery list. CLI mirror: `spark-bench quick "..."`;
- drop into the Chat panel to talk to a single picked model without
  leaving the TUI;
- inspect suite metadata (description, task count, fixture path) before
  committing to a run.

`spark-bench` with no subcommand defaults to `shell`. `Esc` / `q` cancels
overlays, `Enter` confirms, `Space` toggles selections.

### `spark-bench wizard` — multiselect picker → bundle run

A lighter alternative to the full TUI when you already know what you
want. Curses overlay with arrow-key navigation:

```bash
PYTHONPATH=src python3 -m spark_benchmark.cli wizard \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark \
  --allow-auto-detected   # optional: also offer non-curated Ollama tags
```

`↑`/`↓` move, `Space` toggles a model or suite, `Enter` confirms. After
two screens (models, then suites) the harness runs the matching bundle
end-to-end and prints a CLI summary plus the paths to `report.md` and
`report.html`. The HTML report is a single-file standalone page (no JS,
no CDN, no external assets) with a gradient hero banner, stat tiles
(models / suites / tasks / overall pass rate), color-coded ranking
cells, and per-suite dashboard cards — each suite gets charts tailored
to it: pass-rate bars and TTFT (lower-is-better, inverted colour) for
the speed probe, per-task pass/fail strips for the reliability suites,
per-benchmark stacked bars for code generation, and dual-bars +
throttle-ratio gauges + peak-temp thermometers + a tps-over-time line
chart (with optional GPU-temp overlay) for sustained throughput. With
`--allow-auto-detected` the picker shows every non-vision Ollama tag,
flagged as `auto-detected`.

### `spark-bench console` — single-model REPL

One model, free-form prompts, until you type `/exit`:

```bash
PYTHONPATH=src python3 -m spark_benchmark.cli console \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark \
  --model gemma-4

# Talk to any Ollama tag, even without a YAML config:
PYTHONPATH=src python3 -m spark_benchmark.cli console \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark \
  --allow-auto-detected --model phi4:14b
```

Useful for sanity-checking a model end-to-end (Ollama tag mapping,
sampling defaults, warm-up latency) before kicking off a longer
benchmark. `--model` accepts experiment names, raw Ollama tags, and the
slugified form interchangeably. Omitting `--model` picks the first
config in the resolved list (curated first, then auto-detected).

### `spark-bench benchmark <natural-language request>` — NL batch

Not interactive in the curses sense, but accepts a Czech / English
sentence and routes it to a `BenchmarkPlan`:

```bash
PYTHONPATH=src python3 -m spark_benchmark.cli benchmark \
  otestuj qwen gemma nemotron zamer se na rychlost spolehlivost \
  a openclaw structured output \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark
```

Recognised keywords include `rychlost`/`speed`, `spolehliv`/`reliab`,
`kod`/`code`, `dlouhodob`/`sustained`, `openclaw`, `json`, `structured`,
plus model aliases (`qwen` → `qwen-3.6`, `gemma` → `gemma-4`, `nemotron`
→ `nemotron-3`). Pass `--allow-auto-detected` to also route to any
non-curated Ollama tag by its slugified name (`phi4:14b` → `phi4-14b`).

### `spark-bench run-custom` — bring your own test

Mode A custom suites: drop a YAML file with your prompts, point the CLI
at it, and get a side-by-side comparison across whatever models you
have in Ollama.

```bash
PYTHONPATH=src python3 -m spark_benchmark.cli run-custom \
  examples/custom-tests/quick/suite.yaml \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark
```

What you get under `results/custom/<slug>/<run-id>/`:

- `manifest.json` (tagged `kind: "custom"` so reports keep custom runs
  visually distinct from canonical numbers),
- `results.jsonl` (one row per `(model, task_id)`, with per-call
  telemetry; resume-friendly — rerun against the same `--output-dir`
  and the runner skips already-done pairs),
- `summary.md` (per-model telemetry table at the top, then one section
  per task with each model's reply rendered as a fenced block),
- `summary.html` (standalone styled HTML — telemetry table, mean
  decode-tps bar chart, and one collapsible `<details>` block per task
  with each model's reply side-by-side, errored cells highlighted),
- `summary.json` (machine-readable aggregates).

`run-custom` defaults to `--allow-auto-detected` because the user
explicitly opted in to a non-canonical workload. A bare
`spark-bench validate-custom path/to/suite.yaml` checks the schema
without running anything (catches duplicate task IDs, empty prompts,
unknown model references, the not-yet-implemented `mode: scored`).
There is no scoring in v0.2.0 — `quick` mode is pass-through only. See
[`custom-tests-spec.md`](custom-tests-spec.md) for the v0.3.0+ roadmap
that adds deterministic scorers, sandboxed custom-Python scorers, and
a local LLM-as-judge.

### `spark-bench quick` — one prompt, all models, no YAML

The lightest BYOT entry point. Type one prompt on the command line,
get the same side-by-side `summary.md` / `summary.html` `run-custom`
produces, no YAML required.

```bash
PYTHONPATH=src python3 -m spark_benchmark.cli quick \
  "Vysvětli česky idiom 'házet hrách na zeď' a uveď moderní příklad." \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark
```

Internally it builds a one-task `CustomSuiteDefinition` with
`task_id="ad-hoc"` and feeds it to the same
`run_custom_suite_quick` runner — there's only one runner, one
summary format, one results layout. Adds three quick-only flags:

- `--name <slug>` overrides the default `quick-<slug-from-prompt>`
  suite name (also drives the run-bundle directory).
- `--save` (or `--save-path <dir>`) writes the prompt as a
  reusable suite YAML to `examples/custom-tests/quick-saved/<slug>/`
  by default. That folder is git-ignored — quick prompts are
  personal scratchpads, not shipped templates — but
  `shell.discover_custom_suites` picks it up so the saved prompt
  shows up in the TUI's Custom menu next time.
- `--overwrite` allows replacing an existing saved suite at the
  target path.

Manifest fields specific to the quick path: `source: "cli-quick"`
(or `"shell-quick"` from the TUI), and `ad_hoc_prompt: true`. See
[`custom-tests-spec.md`](custom-tests-spec.md) → "Quick (ad-hoc
one-shot prompts)" for the full design.

### What `--allow-auto-detected` actually does

All four CLI surfaces share `model_registry.resolve_runnable_models`:

- **off (default)** — only experiment YAML models are runnable. Best for
  reproducible runs and CI, where you want the manifest to match a
  reviewed lineup.
- **on** — also probes Ollama, classifies every detected tag (vision /
  embedding tags are still skipped), and synthesizes a `ModelConfig` for
  the rest using Ollama defaults. Synthesized entries carry
  `notes=["auto-detected from Ollama (no YAML config)"]` so the run
  manifest, report, and Markdown summary all flag the entries that
  weren't reviewed.

The curses TUI (`spark-bench shell`) always behaves as if the flag were
on — the operator is in front of the screen and can see the
`auto-detected` label. The same is true of the `Custom` menu entry,
which mirrors `run-custom`'s default.

## Current scaffold includes

- repository structure
- validated YAML config loading via Pydantic
- CLI with run, aggregate, report, and dashboard commands
- backend and telemetry base interfaces
- Spark-only sample experiment, platform, backend, and model configs
- first working reliability suite runner: `spark-bench run --experiment configs/experiments/spark-ollama-baseline.yaml --platform spark --run-suite hallucination_grounding` loads `data/reliability/hallucination_grounding_v1.json`, runs every task against every configured model, writes one row per (model, task) to `results.jsonl`, and emits `summary.json` + `summary.md` with per-model pass rates using simple heuristics for `answer_from_context`, `abstain`, and `correct_user`
- code generation suite (`--run-suite code_generation`): canonical HumanEval-style problems with sandboxed execution (`subprocess` + `resource.setrlimit` + timeout) and pass@k unbiased estimator; reference-score validator at `data/code/reference_scores.yaml` emits warnings when results drift from published baselines. See [`extensions-spec.md`](extensions-spec.md) for the full long-context / sustained-throughput / code-generation extension plan
- placeholder suite structure for quality, performance, reliability, and practical task checks
- bring-your-own-test (BYOT) custom suites — `spark-bench run-custom` plus `spark-bench validate-custom`. v0.2.0 ships Mode A (pass-through, no scoring); see [`custom-tests-spec.md`](custom-tests-spec.md) for the roadmap.

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
| `tests/test_shell.py` | Curses-shell classifier surface — `classify_models(ShellContext, …)` backwards-compat wrapper, `SUITE_REGISTRY` ↔ fixture wiring sanity |
| `tests/test_model_registry.py` | Shared model registry — `slugify_tag` / `synthesize_model_config` defaults, `classify_detected` filtering (vision / embedding / auto-synth), `resolve_runnable_models` with and without `--allow-auto-detected` (collisions skip auto entries), `find_config_by_name_or_tag` lookup order |
| `tests/test_custom_suites.py` | Custom (BYOT) suites — Pydantic validation (duplicate IDs, empty prompts, empty tasks), YAML / JSON loaders, soft validation (long prompts, bad sampling, unknown models, `mode: scored` rejection), end-to-end runner with resume + error recording, summary aggregation, Markdown rendering, `slugify_suite_name` |
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
