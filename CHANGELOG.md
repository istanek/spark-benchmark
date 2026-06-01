# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Design spec for `long_context_retrieval` (v0.4.0 target).** New
  `docs/long-context-spec.md` is the implementation-ready plan for the
  single-needle NIAH suite, written against the real v0.3.0 codebase.
  Locks in: Project Gutenberg / Apache-2.0 public-domain haystacks,
  Part A (substring scoring) only — no LLM judge, per-model
  tokenization with honest reported lengths, a 4×4×8 grid (128
  tasks/model, 8 samples/cell) for statistically meaningful heatmaps,
  inline-SVG heatmaps via a new `_svg_heatmap` helper (no matplotlib /
  no new runtime dependency), deterministic needle/haystack selection,
  and three-state cells (pass / N/A-unsupported / OOM). Supersedes the
  aspirational Suite 1 sketch in `docs/extensions-spec.md`. Also records
  the agreed release ladder: v0.4.0 long-context → v0.5.0
  quantization sweep → v0.6.0 concurrent serving.

## [0.3.0] - 2026-06-01

### Added

- **Marketing-grade HTML reports — second pass.** The HTML reports
  picked up a complete visual overhaul on top of the standalone-file
  foundation introduced earlier in this release. Same single-file
  invariants (no JavaScript, no CDN, no external assets), but the
  visual quality is now "would happily attach this to a board deck"
  rather than "minimum viable HTML":
  - **Hero banner** at the top of every report — radial-gradient
    purple/indigo background (cyan-leaning for custom runs so the
    two flavours are visually distinct), display-size H1, subtitle
    pulled from the request prompt, and a glassmorphism "Recommended
    pick" winner card with the model name and a one-line justification
    ("perfect grounding reliability; TTFT 120 ms; 42.0 tok/s").
  - **Stat-tile strip** under the hero — five tiles for canonical
    bundles (models tested, suites run, total tasks, overall pass
    rate colour-graded, top model score) and four for custom runs
    (completed pairs, errored pairs, fastest decode model, lowest
    TTFT model).
  - **Verdict card** with a soft gradient background and indigo accent
    border, replacing the bare-bones verdict paragraph.
  - **Color-coded pass-rate cells.** Every percentage cell in the
    canonical report tables now carries a CSS ``--cell-pct`` custom
    property and a ``data-band`` (good / warn / bad / na) attribute,
    rendering a proportional fill behind the value (≥95 % green,
    80–95 % amber, <80 % red).
  - **Sticky table headers** so column labels stay visible while
    scrolling long rankings.
  - **Print stylesheet** (``@media print``) — gradients flatten to
    flat colours, shadows vanish, ``<details>`` collapses cleanly,
    every ``break-inside`` is set to avoid cutting tables / cards.
- **Per-suite dashboard cards with suite-specific charts.** Each of
  the five canonical suites now renders into a dashboard card with
  a 3-up grid of charts tailored to *that* suite, plus the shared
  per-model results table:
  - **``openclaw_speed``** — pass rate (good-bg gradient) + TTFT bar
    chart with **inverted colour** (lower = greener = better) +
    decode throughput (tok/s) bars in green.
  - **``hallucination_grounding``** and
    **``practical_structured_output``** — pass rate + TTFT
    (inverted) + a wide **per-task pass-fail strip** (green / red /
    grey squares per task) loaded lazily from the suite's
    ``results.jsonl``.
  - **``code_generation``** — aggregate pass@1 + **per-benchmark
    stacked bars** (HumanEval / MBPP / …) + a wide **sandbox-status
    breakdown** (passed / failed / timeout / oom / compile_error /
    runtime_error) loaded from per-row sandbox status fields.
  - **``sustained_throughput``** — initial vs sustained
    **dual-bar** per model + per-model **throttle-ratio gauges**
    (semicircle SVG arcs, colour-graded) + per-model **peak-temp
    thermometers** + a wide **tps-over-time line chart** with an
    optional GPU-temperature overlay (dashed secondary axis) loaded
    from ``telemetry-<model>.jsonl``.
- **New SVG primitives** in ``reporting_html``: ``_svg_line_chart``
  (multi-series with optional secondary axis, adaptive grid lines,
  inline legend), ``_svg_gauge`` (180° semicircle, colour-graded,
  optional invert), ``_svg_dual_bars`` (paired thin bars with shared
  scale), ``_svg_stacked_bars`` (segmented bar with hint counts),
  ``_svg_thermometer`` (vertical bar + bulb), ``_pass_fail_strip_html``
  (per-model task-by-task dot strip). All inline-SVG, all
  ``viewBox``-based so they scale with the container.
- **Lazy data loaders** for the renderer:
  ``_load_results_rows(run_dir)`` reads ``results.jsonl`` (skips
  malformed lines, returns empty on missing file),
  ``_load_telemetry_samples(run_dir, model, max_points=240)`` reads
  ``telemetry-<model>.jsonl`` and uniformly downsamples (a
  30-minute soak with ~18 000 points compresses to a 240-point
  curve under a few KB).
- **Custom (BYOT) / quick run polish.** ``summary.html`` now ships
  with the same hero / stat-tile chrome (cyan-tinted gradient so it's
  visually distinct from a canonical bundle), each ``<details>``
  task block opens with a 2-up mini-chart row showing **TTFT
  comparison** (lower is better, inverted colour) and **output
  length** (decode tokens) per model, and the task summary header
  carries a small **error strip** of dots so the user can scan a
  long suite for failures without expanding every block.
- **Plumbing in ``aggregate_runs``.** Per-model entries now forward
  ``windows`` (sustained-throughput per-window throughput series),
  ``benchmarks`` (code-generation per-benchmark breakdown), and
  ``run_dir`` (so the HTML renderer can lazy-load
  ``results.jsonl`` / ``telemetry-*.jsonl`` without re-walking the
  filesystem). Markdown / CLI summaries are unchanged.
- **27 new tests** in ``tests/test_reporting_html.py`` covering
  every new SVG helper (line chart with secondary axis, gauge with
  invert, dual bars, stacked bars normalising per-row, thermometer,
  pass-fail strip), color helpers (``_gradient_color_for_ratio``,
  ``_band_for_pass_rate``, ``_cell_pct_html``), lazy loaders
  (results.jsonl + telemetry downsampling + missing-file
  fallbacks), suite-specific dispatch (all 5 suites + unknown-suite
  fallback), color-coded ranking cells, and a true end-to-end
  reliability render that builds a ``results.jsonl`` on disk and
  asserts the per-task strip survives the round-trip. Total HTML
  test count: 38 (was 11).

- **Polished standalone HTML reports.** New
  ``spark_benchmark.reporting_html`` module renders both flavours of
  run output as a single self-contained HTML page — no JavaScript,
  no CDN, no external assets. Open the file from a USB stick, attach
  it to an email, paste it into a wiki: it just works.
  - Canonical bundles now emit ``report.html`` next to ``report.md``
    with overall ranking, per-suite tables, narrative commentary,
    verdict / recommendation, and inline SVG bar charts for overall
    score and per-suite pass rates.
  - Custom (BYOT) and ``quick`` runs now emit ``summary.html`` next
    to ``summary.md`` / ``summary.json`` with a per-model telemetry
    table, mean-decode-tps bar chart, and one collapsible
    ``<details>`` block per task showing every model's reply
    side-by-side. Errored cells are highlighted in red.
  - ``write_report`` learned a new ``"both"`` format that writes the
    ``.md`` and ``.html`` siblings in one call. Existing
    ``"markdown"`` / ``"html"`` paths continue to work unchanged.
  - All renderers HTML-escape user content (prompt text, model
    output, error messages) so YAML suites containing ``<script>``
    or ``<img onerror=...>`` payloads can't escape the ``<pre>`` /
    ``<code>`` containers.
- **CLI / TUI surface for HTML.** ``spark-bench benchmark``,
  ``wizard``, ``aggregate``, ``run-custom``, and ``quick`` all log
  the HTML path next to the existing markdown / JSON paths.
  ``aggregate``'s JSON output gained ``"aggregate_html"`` and the
  custom commands gained ``"summary_html"``. The TUI ``Run`` /
  ``Custom`` / ``Quick`` flows print the HTML path in their final
  log block.
- **``tests/test_reporting_html.py``** — 11 plain-Python tests
  covering document well-formedness (doctype, no script tags,
  embedded ``<style>``), canonical-renderer ranking / verdict /
  per-suite blocks, custom-renderer telemetry / per-task details,
  HTML-escaping of user-supplied prompts and outputs, SVG bar-chart
  edge cases (empty input, value formatting, clamping), and a
  ``write_report(..., "both")`` integration assertion that the
  ``.md`` and ``.html`` siblings land next to each other.

- **Quick (ad-hoc one-shot prompts).** New
  ``spark_benchmark.quick`` module surfaces the lightest BYOT
  workflow yet — type one prompt, fan it out to every model you
  picked, get the same ``summary.md`` ``run-custom`` produces. No
  YAML required up front.
- **CLI command ``spark-bench quick "your prompt here"``.** Builds
  a one-task ``CustomSuiteDefinition`` in memory
  (``task_id="ad-hoc"``) and feeds it to the existing
  ``run_custom_suite_quick`` runner — single runner, single
  summary format, single results layout. Flags: ``--models``,
  ``--allow-auto-detected`` (default ON), ``--name`` (overrides the
  ``quick-<slug>`` default), ``--save`` / ``--save-path`` /
  ``--overwrite`` to persist the prompt as a reusable suite YAML,
  and ``--output-dir``.
- **TUI menu entry ``Quick``.** Sits between ``Custom`` and
  ``Models``. Walks the user through model multi-select → drops
  out of curses to read a single-line prompt on the regular TTY →
  runs ``run_custom_suite_quick`` with progress streaming into the
  log → asks ``Save this prompt as a reusable custom suite?
  [y/N]`` afterwards. If saved, the run's ``manifest.json`` is
  patched in place so ``suite_path`` points at the saved YAML and
  ``discover_custom_suites`` surfaces it next time.
- **Saved-quick layout.** ``examples/custom-tests/quick-saved/`` is
  the default save root. The directory is **git-ignored** (added
  to ``.gitignore``) so personal one-shots stay out of source
  control while still being findable by the existing TUI discovery
  helper.
- **Manifest provenance fields.** Quick runs carry
  ``source: "cli-quick" | "shell-quick"`` and
  ``ad_hoc_prompt: true`` so reports can tell quick runs apart
  from canonical custom-suite runs (which use ``"cli"`` /
  ``"shell"``).
- **``tests/test_quick.py``** — 12 plain-Python tests covering
  ``build_quick_suite`` (one task, ad-hoc id, default-name slug,
  empty-prompt rejection, sampling pass-through, punctuation-only
  fallback), ``save_quick_suite_as_yaml`` (round-trip through
  ``load_custom_suite``, refuses to clobber by default,
  ``overwrite=True`` replaces, empty optional fields trimmed), and
  an end-to-end run against a fake backend asserting one row per
  ``(model, ad-hoc)`` pair.

### Changed

- ``shell.MENU_ITEMS`` gained ``("quick", "Quick")``; dispatch
  routes Enter on it to ``TUIApp.do_quick``.
- ``CONTRIBUTING.md`` is unchanged; ``README.txt``,
  ``docs/README.md``, ``docs/architecture.md``,
  ``docs/custom-tests-spec.md``, and
  ``.cursor/rules/project-overview.mdc`` were extended to cover
  the new entry point.
- ``reporting.render_html_report`` is now a thin delegate to
  ``reporting_html.render_canonical_report_html``. The previous
  unstyled stub (a bare ``<html><body>`` with one un-themed table
  per suite) is gone — anything that called it now produces the
  full styled report instead, which is a deliberate behaviour
  change with no downstream API change.
- The benchmark / wizard / shell-run flows now write the report
  bundle as ``"both"`` (``report.md`` *and* ``report.html``) by
  default. ``aggregate`` likewise writes both.

### Fixed

- **Sustained-throughput dual bars overflowed their panel.** The
  "Initial vs sustained throughput" SVGs in
  ``_svg_dual_bars`` sat one level deeper inside an extra flex
  wrapper, so the ``.bar-row svg { flex: 1 1 auto }`` rule didn't
  reach them. With no explicit ``width``, browsers fell back to
  the inline-SVG default (300 px) and the bars spilled into the
  neighbouring "Throttle ratio" card. The track wrapper now uses
  ``flex: 1 1 0; min-width: 0; overflow: hidden`` and the inner
  SVGs carry ``width: 100%; display: block``. Added matching
  CSS safety nets — ``.bar-row svg`` now also sets
  ``min-width: 0``, ``.dual-bars-track svg`` enforces full-width,
  and a global ``svg.lines { width: 100%; height: auto }`` rule
  keeps the throughput line chart honest in narrow containers.

## [0.2.1] - 2026-05-28

Polish release on top of 0.2.0. Surfaces the BYOT subsystem in the
curses TUI (so users no longer have to type ``run-custom`` flags),
moves the project home from GitLab to GitHub (history and tags carry
over with identical commit hashes), and fixes a long-standing
"ESC needs two presses to leave a submenu" bug.

### Fixed

- **ESC double-press in the curses TUI.** ``ncurses`` defaults to
  ``ESCDELAY=1000``, so a bare ESC sat in the read buffer for a full
  second while the library waited to see if it was the start of an
  escape sequence (arrow keys, F-keys). Users learned to hit ESC
  twice. ``shell.TUIApp.run`` now calls ``curses.set_escdelay(25)``
  right after ``curses.curs_set(0)`` (the value vim and htop use),
  with a graceful fallback for environments where the symbol is
  missing. Single ESC now leaves singleselect / multiselect overlays
  immediately.

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

[Unreleased]: https://github.com/istanek/spark-benchmark/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/istanek/spark-benchmark/releases/tag/v0.2.1
[0.2.0]: https://github.com/istanek/spark-benchmark/releases/tag/v0.2.0
[0.1.0]: https://github.com/istanek/spark-benchmark/releases/tag/v0.1.0
