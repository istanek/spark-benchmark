# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Project home moved from GitLab to GitHub
  (`https://github.com/istanek/spark-benchmark`).** All Git history
  and both release tags (`v0.1.0`, `v0.2.0`) carry over unchanged
  (identical commit hashes). Knock-on edits in this commit:
  - ``CHANGELOG.md`` compare/tag link references repointed from
    ``gitlab.com/.../-/compare`` and ``-/tags`` to
    ``github.com/.../compare`` and ``releases/tag``.
  - ``docs/README.md`` swaps the GitLab pipeline badge for a GitHub
    Actions CI badge and rewords the "landing page" note.
  - ``README.txt`` "Help, support, bugs" now points at GitHub Issues
    and a pull request workflow; the install snippet uses the new
    GitHub URL.
  - ``CONTRIBUTING.md`` switches "Merge Request" / "MR" / "GitLab
    issues" wording to "Pull Request" / "PR" / "GitHub Issues" and
    points the cloning snippet at GitHub.
  - ``.gitlab-ci.yml`` was removed and replaced with
    ``.github/workflows/ci.yml`` running the same two stages
    (YAML/JSON fixture lint + ``pytest tests/``) on every push and
    pull request to ``main``.
  - ``scripts/release.sh`` was rewritten against the GitHub Release
    API (``POST /repos/<owner>/<repo>/releases``,
    ``Authorization: Bearer …``, ``Accept: application/vnd.github+json``).
    It now reads ``GITHUB_TOKEN`` first, then falls back to
    ``gh auth token`` and finally to ``~/.git-credentials`` for
    ``github.com``. The CHANGELOG-extraction logic and tag/push
    flow are unchanged.

### Added

- **Custom (BYOT) menu item in the curses TUI (`spark-bench shell`).**
  A new ``Custom`` entry sits next to ``Run`` and walks the user
  through the same flow as ``spark-bench run-custom`` on the CLI:
  it discovers suite YAMLs (shipped templates under
  ``examples/custom-tests/`` plus prior runs under
  ``results/custom/<slug>/<run-id>/``), loads + validates the
  selected suite, asks for models in a multiselect that respects
  the suite's ``models:`` list when present, and streams progress
  into the log as the run executes. The run bundle is written to
  ``results/custom/<slug>/<run-id>/`` with ``manifest.json`` tagged
  ``source: shell`` so reporting can tell TUI runs apart from CLI
  runs. ``--allow-auto-detected`` is implicitly on for the TUI
  entry, matching ``spark-bench run-custom``.
- **`shell.discover_custom_suites(repo_root)` helper** — pure
  function that returns ``CustomSuiteCandidate`` items, dedupes
  recent runs by absolute ``suite_path`` (newest ``run-id`` wins),
  and silently skips manifests pointing at deleted suite files.
  Covered by three new tests in ``tests/test_shell.py``.

## [0.2.0] - 2026-05-22

Second public release. Adds the Bring-Your-Own-Test (BYOT) subsystem
in Mode A (pass-through, no scoring), unifies model auto-detection
across every CLI surface behind one shared registry, and switches the
GitLab project page to the plain-language `README.txt`.

### Changed

- **Project landing page is now `README.txt` (plain language).** The
  markdown version moved to `docs/README.md` so the GitLab project page
  renders the human-friendly plain-text overview by default. PyPI
  metadata (`pyproject.toml::readme`) was repointed to the new
  location and serves the same markdown content. All cross-references
  in the `docs/` tree were updated to mention both paths; root
  `README.md` was deleted.

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

[Unreleased]: https://github.com/istanek/spark-benchmark/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/istanek/spark-benchmark/releases/tag/v0.2.0
[0.1.0]: https://github.com/istanek/spark-benchmark/releases/tag/v0.1.0
