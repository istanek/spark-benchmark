# spark-benchmark architecture

Companion to `README.txt` (plain-language overview) / `docs/README.md`
(markdown version with badges) and `METHODOLOGY.md` (how we score). This
doc is the **how the code is laid out**: which module owns what, how data flows
end-to-end, and what extends where.

If you only read one diagram, read this one:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                ENTRY POINTS                                  │
│   cli.py (Typer)         shell.py (curses TUI)                               │
│   ├─ run                 ├─ Run     → do_run                                 │
│   ├─ run-custom          ├─ Custom  → do_custom (BYOT, 0.2.1+)               │
│   ├─ validate-custom     ├─ Models  → show_models                            │
│   ├─ console             ├─ Suites  → show_suites                            │
│   ├─ benchmark           ├─ Info    → show_info                              │
│   ├─ wizard              ├─ Chat    → do_chat / chat_command                 │
│   ├─ aggregate           └─ Refresh / Quit                                   │
│   ├─ report                                                                  │
│   └─ dashboard                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CONFIG + CONTEXT LOADING                            │
│  config.py        load_experiment / load_platform / load_backend             │
│                   load_model_config            (YAML → Pydantic)             │
│  models.py        ExperimentSpec, PlatformConfig, BackendConfig,             │
│                   ModelConfig, SamplingConfig, RunManifest, …                │
│  shell.py         load_default_context()       (bundles the above into a     │
│                                                 ShellContext)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATION                                   │
│  orchestration.py                                                            │
│   ├─ parse_benchmark_request   NL → BenchmarkPlan (models + suites)          │
│   └─ run_benchmark_bundle      iterates plan.selected_suites, dispatches     │
│                                each one to its suite runner                  │
│  shell.py        detect_ollama_models / classify_models                      │
│                   (Ollama /api/tags → DetectedOllamaModel → OllamaModelInfo) │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                          ▼
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│  SUITE RUNNERS                   │      │  BACKEND ADAPTERS                │
│  reliability.py                  │      │  runners/base.py (Protocol)      │
│   ├─ hallucination_grounding     │      │   ├─ ollama.py  (HTTP)           │
│   └─ practical_structured_output │      │   ├─ llamacpp.py (subprocess)    │
│  orchestration.py                │      │   └─ stub.py    (no-op)          │
│   └─ openclaw_speed              │      │  runners/registry.py             │
│  code_generation.py              │      │   └─ build_backend(BackendConfig)│
│   └─ run_code_generation_suite   │      │                                  │
│  sustained_throughput.py         │      │  Each adapter implements:        │
│   └─ run_sustained_throughput_…  │      │   load_model / generate /        │
│      + TelemetrySampler (NVML)   │      │   get_metrics / unload           │
└──────────────────────────────────┘      └──────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RESULTS + REPORTING                               │
│  results_bundle.py   make_run_id, ensure_run_dir, write_manifest,            │
│                      write_result (JSONL), write_json                        │
│  runtime.py          build_manifest / build_environment_snapshot             │
│  reporting.py        aggregate_runs → render_markdown_report                 │
│                                     → render_html_report                     │
│                                     → render_cli_benchmark_summary           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. Layout

```
spark-benchmark/
├─ src/spark_benchmark/
│   ├─ cli.py                   Typer app, sub-commands, runtime context glue
│   ├─ shell.py                 curses TUI shell (Run/Models/Suites/Chat/…)
│   ├─ models.py                Pydantic data classes (configs + results)
│   ├─ config.py                YAML → Pydantic loaders
│   ├─ suites.py                SuiteDefinition + SuiteTask (JSON suite fixtures)
│   ├─ orchestration.py         NL parser, suite dispatch, openclaw_speed runner
│   ├─ reliability.py           hallucination_grounding + practical_structured_output
│   ├─ code_generation.py       HumanEval-style sandboxed code suite
│   ├─ sustained_throughput.py  long-decode soak + NVML/nvidia-smi telemetry
│   ├─ reporting.py             aggregate JSONL → md / html / CLI summary
│   ├─ results_bundle.py        run-id, write_manifest, write_result (JSONL)
│   ├─ runtime.py               build_manifest / EnvironmentSnapshot
│   ├─ evaluator.py             (stub; reserved)
│   ├─ runners/
│   │   ├─ base.py              BackendAdapter Protocol
│   │   ├─ registry.py          build_backend(BackendConfig)
│   │   ├─ ollama.py            HTTP adapter (used by v1)
│   │   ├─ llamacpp.py          subprocess adapter
│   │   └─ stub.py              fallback
│   └─ telemetry/               base / registry / stub (placeholder; real
│                               telemetry lives in sustained_throughput.py)
├─ configs/
│   ├─ experiments/*.yaml       what to run (backend + models + suites + sampling)
│   ├─ platforms/*.yaml         where we're running (spark.yaml)
│   ├─ backends/*.yaml          how we talk to inference (ollama, llamacpp, trt-llm)
│   └─ models/*.yaml            per-model ModelConfig (name, tag, quant, ctx, …)
├─ data/                        Suite fixtures (input prompts + references)
│   ├─ performance/             openclaw_speed_v1.json, sustained_throughput_v1.json
│   ├─ reliability/             hallucination_grounding_v1.json
│   ├─ practical/               practical_structured_output_v1.json
│   └─ code/                    code_generation_v1.json + reference_scores.yaml
├─ results/                     Output artifacts (one run-id directory per run)
├─ suites/                      Reserved for future per-suite scratch
├─ tests/                       Plain-python tests (run with `python3 <file>`)
├─ docs/                        This doc + extensions-spec.md
└─ pyproject.toml
```

## 2. Data flow end-to-end

A canonical "run a benchmark" path:

1. **User invokes an entry point**
   - CLI: `python3 -m spark_benchmark.cli wizard --experiment … --platform spark`
   - TUI: `… cli console …` → `shell.py` curses loop.
2. **Context load** (`cli.load_runtime_context` or `shell.load_default_context`)
   - Reads `configs/experiments/<name>.yaml` → `ExperimentSpec`.
   - Reads `configs/platforms/<platform>.yaml` → `PlatformConfig`.
   - Resolves `configs/backends/<experiment.backend>.yaml` → `BackendConfig`.
   - Loads every `configs/models/<model>.yaml` listed in the experiment.
3. **Model selection (TUI)** (`shell.detect_ollama_models` + `classify_models`)
   - GET `<endpoint>/api/tags` → list of `DetectedOllamaModel(tag, family, …)`.
   - For each YAML config whose `artifact_path` matches a detected tag → `OllamaModelInfo(config=cfg)`.
   - For each remaining detected tag, classify via `is_vision_model` / `is_embedding_model`:
     - vision/embedding → `OllamaModelInfo(config=None, disable_reason=…)` (greyed in picker).
     - anything else → `OllamaModelInfo(config=_synthesize_model_config(detected), auto_detected=True)`
       (pre-selected, runs against the default sampling and `context_length=131072`).
4. **Plan** (`orchestration.parse_benchmark_request` for NL, or direct picker)
   - Produces a `BenchmarkPlan(request, selected_models, selected_suites, rationale)`.
5. **Backend build** (`runners.registry.build_backend`)
   - `BackendKind.OLLAMA` → `OllamaAdapter`; `LLAMACPP` → `LlamaCppAdapter`;
     anything else → `StubBackendAdapter`.
6. **Dispatch** (`orchestration.run_benchmark_bundle`)
   - Creates `results/benchmarks/<run-id>/` (`results_bundle.make_run_id`).
   - For each suite in the plan: makes `<bundle>/<suite>/`, writes `manifest.json`
     (`runtime.build_manifest`), then calls the matching suite runner:

| Suite name                        | Runner location                                         |
|----------------------------------|---------------------------------------------------------|
| `hallucination_grounding`        | `reliability.run_hallucination_grounding_suite`         |
| `practical_structured_output`    | `reliability.run_practical_structured_output_suite`     |
| `openclaw_speed`                 | `orchestration.run_openclaw_speed_suite`                |
| `code_generation`                | `code_generation.run_code_generation_suite`             |
| `sustained_throughput`           | `sustained_throughput.run_sustained_throughput_suite`   |

7. **Per-suite runner loop** (shared shape across suites)
   - For each `ModelConfig`: `backend.load_model(...)` → for each `SuiteTask`:
     `backend.generate(prompt, sampling)` → score → `write_result(...)` appends a
     JSON line to `results.jsonl`. Then `backend.unload()` (Ollama frees weights
     via `keep_alive=0`).
   - Each runner finishes by writing `summary.json` (`build_summary`) and
     usually `summary.md` (`write_summary_markdown`).
8. **Aggregation / reporting**
   - `reporting.aggregate_runs(runs_root)` scans every run directory, joins
     `manifest.json` + `summary.json` + `results.jsonl`, and aggregates by suite
     × model (passes/total, ttft, decode tokens, etc.).
   - `render_markdown_report` / `render_html_report` / `render_cli_benchmark_summary`
     consume the aggregate.

## 3. Module reference

### Entry points

- **`cli.py` (`spark-bench` Typer app)** — sub-commands:
  - `run` — execute one suite directly against the configured experiment.
  - `console` — single-model REPL; `--model` accepts experiment name, raw
    Ollama tag, or slugified tag (resolved via
    `model_registry.find_config_by_name_or_tag`).
  - `benchmark` — natural-language flow: parses a Czech/English sentence into a
    `BenchmarkPlan` (`orchestration.parse_benchmark_request`) and runs the bundle.
  - `wizard` — interactive multi-select picker (curses) → bundle run.
  - `aggregate` — fold a `results/` tree into a single JSON.
  - `report` — render aggregate to markdown or HTML.
  - `dashboard` — placeholder for future live view.
  - Common flag: every command that runs models (`run`, `console`, `benchmark`,
    `wizard`) accepts `--allow-auto-detected`; off by default so manifests
    only contain reviewed entries.
  - Helpers: `load_runtime_context`, plus a thin
    `detect_ollama_model_tags(backend) -> set[str]` wrapper kept for callers
    that only need raw tag strings.

- **`model_registry.py`** — shared model-pool resolver, used by all four
  CLI surfaces and by the curses TUI:
  - `DetectedOllamaModel` / `OllamaModelInfo` dataclasses carry the picker
    state (`has_config`, `auto_detected`, `disable_reason`).
  - `detect_ollama_models(backend_config)` hits `/api/tags` and captures
    family info; returns `[]` on any error so callers can probe non-fatally.
  - `classify_detected(model_configs, detected)` joins YAML configs +
    detected tags, auto-synthesizes `ModelConfig` for unknown
    non-vision/non-embedding tags, marks vision and embedding extras as
    disabled.
  - `resolve_runnable_models(...)` is the canonical entrypoint: returns
    `(configs, classified)` where `configs` is the list a CLI command
    should expose (curated YAML, plus auto-detected extras when the flag
    is on).
  - `find_config_by_name_or_tag(needle, ...)` resolves a `--model X`
    string against curated names → tags → slugified tags →
    classification list.
  - `is_vision_model` / `is_embedding_model` — family- and tag-substring
    heuristics (see `_VISION_FAMILY_HINTS`, `_EMBEDDING_FAMILY_HINTS`).

- **`shell.py`** — curses TUI built around `ShellContext`:
  - Re-exports `DetectedOllamaModel`, `OllamaModelInfo`,
    `detect_ollama_models`, `is_vision_model`, `is_embedding_model` from
    `model_registry` for backwards compatibility.
  - `classify_models(ctx, detected)` is a thin wrapper over
    `model_registry.classify_detected(ctx.model_configs, detected)`.
  - Menu actions: `do_run`, `do_custom` (BYOT, since 0.2.1),
    `show_models`, `show_suites`, `show_info`, `do_chat`, `chat_command`
    (readline-style chat outside the curses loop).
  - `discover_custom_suites(repo_root)` — lists shipped templates in
    `examples/custom-tests/**/suite.yaml` plus prior runs from
    `results/custom/<slug>/<run-id>/manifest.json` (deduped on
    absolute `suite_path`, newest run-id per suite). Returns
    `CustomSuiteCandidate` items with an `origin` of `"example"` or
    `"recent"`. `do_custom` shows them in a single-select, then
    delegates the actual run to `custom_suites.run_custom_suite_quick`
    with `--allow-auto-detected` implicitly on.

- **`custom_suites.py`** — Bring-Your-Own-Test (BYOT) subsystem,
  introduced in v0.2.0 and specced in detail at
  `docs/custom-tests-spec.md`. Deliberately separate from
  `suites.SuiteDefinition` so user-driven schema changes never perturb
  the canonical suites:
  - `CustomSuiteTask` / `CustomSuiteDefinition` — Pydantic models for
    user-supplied suites. v0.2.0 honours `mode: quick` (Mode A,
    pass-through, no scoring); `mode: scored` is rejected at load time
    with a pointer to the v0.3.0 roadmap.
  - `load_custom_suite(path)` — YAML or JSON loader (auto-detects on
    suffix; falls back to YAML).
  - `validate_custom_suite(suite, available_models=...)` — soft checks
    beyond the Pydantic schema (long prompts, sampling out of range,
    unknown model references).
  - `run_custom_suite_quick(...)` — runs every `(model, task)`,
    appends rows to `results.jsonl`, supports resume via
    `already_completed_pairs(run_dir)`, records errors as rows
    instead of aborting.
  - `build_custom_summary(...)` + `render_custom_summary_markdown(...)`
    — per-model telemetry table plus a side-by-side per-task block.
  - Bundle layout: `results/custom/<slug>/<run-id>/` with
    `manifest.json` tagged `kind: "custom"` so reporting can keep
    custom runs visually distinct.

### Data model

- **`models.py`** — every Pydantic type used downstream:
  - **Inputs**: `ExperimentSpec`, `PlatformConfig`, `BackendConfig`, `ModelConfig`,
    `SamplingConfig`. `BackendKind` enum (`llamacpp | trt-llm | vllm | ollama`).
  - **Runtime**: `GenerationResult` (output + metrics + raw payload), `InferenceMetrics`
    (prefill/decode tokens + times, TTFT, peak mem).
  - **Manifests**: `EnvironmentSnapshot`, `RunManifest`.
- **`config.py`** — thin YAML→Pydantic shim (`load_yaml_model[T]`).
- **`suites.py`** — `SuiteCategory`, `SuiteTask`, `SuiteDefinition`, plus
  `load_suite_definition(path)` for the JSON suite fixtures in `data/`.

### Orchestration

- **`orchestration.py`**:
  - `BenchmarkPlan` — request + selected models + selected suites + rationale.
  - `parse_benchmark_request(text, available_models)` — keyword-based router
    (Czech + English tokens like `rychlost`/`speed`, `spolehliv`/`reliab`,
    `kod`/`code`, `dlouhodob`/`sustained`). Includes aliasing (`qwen` →
    `qwen-3.6`, etc.).
  - `run_openclaw_speed_suite` — performance probe (no quality scoring, every
    row passes; the point is the captured metrics).
  - `run_benchmark_bundle` — top-level dispatcher; writes `plan.json` and
    iterates per-suite.
- **`runtime.py`** — `build_environment_snapshot` + `build_manifest`.

### Suite runners

Each runner follows the same shape: take `run_dir`, `suite`, `backend`,
`backend_config`, `model_configs`, `sampling`, optional `progress_callback`;
write `results.jsonl` row-by-row; finish with `summary.json` (+ optionally
`summary.md`).

- **`reliability.py`** — `hallucination_grounding`,
  `practical_structured_output`. Scoring:
  - `score_hallucination_task` understands three `expected_behavior` flags:
    `answer_from_context`, `abstain`, `correct_user`. Uses
    `ABSTAIN_PHRASES`/`NEGATION_PHRASES` and a token-overlap heuristic.
  - `score_structured_output_task` requires `json_exact_match`; uses
    `extract_json_value` (handles ```json fences, trailing prose detection).
  - `build_summary` / `write_summary_markdown` are reused by the other
    pass/fail-style suites (`openclaw_speed` calls `build_summary` even though
    every row is a synthetic pass).

- **`code_generation.py`** — HumanEval-shaped:
  - `pass_at_k` unbiased estimator (`n`, `c`, `k`).
  - `extract_code` parses the model output; `_build_program` glues code + tests.
  - `sandbox_run` executes the candidate in a subprocess with
    `resource.setrlimit` (RSS + CPU) + timeout. Failures get classified by
    `_classify_failure` (`timeout`, `compile_error`, `assertion`, `runtime`, …).
  - `validate_reference_scores` cross-checks measured pass@k against
    `data/code/reference_scores.yaml` with a tolerance; mismatches log warnings
    but don't fail the suite.

- **`sustained_throughput.py`** — long-decode soak:
  - `TelemetrySampler` — background thread polling at `DEFAULT_TELEMETRY_HZ`,
    detection order: `pynvml` → `nvidia-smi` → `none` (degrades gracefully).
  - `GenerationRecord` per attempt; `compute_windows` slices the run into walls
    (e.g. first-minute vs sustained); `compute_derived_metrics` computes
    throttling and energy.
  - `run_sustained_throughput_suite` orchestrates per-model decode storms,
    writes per-window summary plus a `summary.md`.

### Backend adapters

- **`runners/base.py`** — `BackendAdapter` Protocol (`load_model`, `generate`,
  `get_metrics`, `unload`).
- **`runners/registry.py`** — `build_backend(BackendConfig)` selects the
  concrete adapter by `BackendKind`.
- **`runners/ollama.py`** (the production path):
  - POSTs to `<endpoint>` (`/api/generate`); reads `prompt_eval_*` /
    `eval_*` fields to populate `InferenceMetrics`.
  - `unload()` posts `{"keep_alive": 0}` so the next model fits in VRAM.
  - Errors raise `RuntimeError` with the HTTP body (so suite runners can
    abort/log cleanly).
- **`runners/llamacpp.py`** — subprocess `llama-cli` (or configured
  `executable`) with `-m <gguf>`, sampling flags, `--ctx-size`. Requires
  `ModelConfig.artifact_path` to be a real local GGUF file.
- **`runners/stub.py`** — used for `trt-llm` / `vllm` until they get real
  adapters. Returns a synthetic response so the orchestration pipeline can be
  exercised without a backend.

### Results + reporting

- **`results_bundle.py`** — pure I/O helpers:
  - `make_run_id()` → `YYYYMMDDTHHMMSSZ-<8hex>`.
  - `write_result(run_dir, row)` appends a JSON line to `results.jsonl`.
  - `write_manifest(run_dir, manifest)` writes `manifest.json`.
- **`reporting.py`** — read-only aggregation:
  - `aggregate_runs(runs_root)` traverses every directory that has both a
    `manifest.json` and `summary.json`, joins them with `results.jsonl`, and
    bins by suite × model.
  - Handles two row shapes: flat `row.generation.metrics` (reliability/speed)
    vs nested `row.samples[*].generation.metrics` (code_generation).
  - Render layers: `render_markdown_report` (publication), `render_html_report`
    (dashboard-ish), `render_cli_benchmark_summary` (TUI/CLI live output).

### Telemetry

- **`telemetry/{base,registry,stub}.py`** — interface placeholders. The real
  GPU sampling currently lives inside `sustained_throughput.TelemetrySampler`;
  the package-level telemetry registry is reserved for a future refactor that
  promotes it into its own module.

## 4. Config and data layout

### Experiment YAML
`configs/experiments/spark-ollama-baseline.yaml` (canonical example):

```yaml
experiment:
  name: spark-ollama-v1-baseline
  description: ...
  platforms: [spark]
  backend: ollama          # picks configs/backends/ollama.yaml
  backend_version: local
  models:                  # each name must have configs/models/<name>.yaml
    - qwen-3.6
    - gemma-4
    - nemotron-3
  suites: [...]            # any of openclaw_speed, hallucination_grounding,
                           # practical_structured_output, code_generation,
                           # sustained_throughput
  sampling: { temperature: 0.0, top_p: 1.0, seed: 42, max_tokens: 512 }
  context_lengths: [...]
  repetitions: 1
  warmup_runs: 0
```

### Model YAML
`configs/models/<model>.yaml`:

```yaml
name: qwen-3.6
family: qwen
revision: qwen3.6:35b
quantization: ollama-default
source: ollama-local
context_length: 131072
artifact_path: qwen3.6:35b     # Ollama tag, or path to GGUF for llamacpp
notes: [ ... ]
```

Auto-detected models (no YAML present) get a synthesized `ModelConfig` with
`name = slugified tag`, `family` from Ollama details, `context_length = 131072`,
`artifact_path = tag`, and `notes = ["auto-detected from Ollama (no YAML config)"]`.

### Suite JSON
`data/<category>/<suite>_v<n>.json` (validated through `SuiteDefinition`):

```json
{
  "name": "...",
  "category": "reliability",
  "description": "...",
  "version": "0.0.1",
  "tasks": [
    { "task_id": "...", "prompt": "...", "context": "...",
      "reference": "...", "tags": [...], "metadata": {"expected_behavior": "..."}}
  ]
}
```

`metadata.expected_behavior` is the contract between fixtures and scorers
(`answer_from_context`, `abstain`, `correct_user`, `json_exact_match`).

### Results directory shape

```
results/benchmarks/<run-id>/
├─ plan.json                          # full BenchmarkPlan (request + rationale)
├─ <suite_name>/
│   ├─ manifest.json                  # ExperimentSpec + Platform + Backend + env
│   ├─ results.jsonl                  # one row per (model, task[, sample])
│   ├─ summary.json                   # per-model passes/total/pass_rate (+ extras)
│   └─ summary.md                     # human-readable summary (where supported)
└─ report.md                          # optional aggregate (via `report` command)
```

`results/runs/` is an older flat layout (one run = one suite, no bundle). Both
shapes are accepted by `aggregate_runs`.

## 5. Extension recipes

### Adding a new suite

1. Add fixture JSON under `data/<category>/<name>_v<n>.json` (must validate
   against `SuiteDefinition`).
2. Implement `run_<name>_suite(*, run_dir, suite, backend, backend_config,
   model_configs, sampling, progress_callback=None)` following the existing
   per-row write pattern (`backend.load_model` → loop → `backend.generate` →
   `write_result` → `backend.unload`).
3. Register dispatch in `orchestration.run_benchmark_bundle`'s suite chain.
4. Register name in `shell.SUITE_REGISTRY` (label + data_path) so the TUI picker
   shows it.
5. Add NL keywords to `orchestration.parse_benchmark_request` so the
   `benchmark` CLI command can pick it up from a sentence.

### Adding a new backend

1. Create `runners/<backend>.py` implementing the `BackendAdapter` Protocol
   (`load_model`, `generate`, `get_metrics`, `unload`).
2. Add `BackendKind.<NAME>` in `models.py` and wire it in
   `runners/registry.build_backend`.
3. Drop a default config in `configs/backends/<name>.yaml`.
4. Reference it from an experiment YAML (`backend: <name>`).

### Adding a new model

For Ollama-served models you no longer need a YAML — pull the model and the
TUI auto-detects it (skips vision / embedding tags). To pin sampling, context
length, or aliasing in the wizard or NL parser, add `configs/models/<name>.yaml`
and reference it in an experiment's `models:` list.
