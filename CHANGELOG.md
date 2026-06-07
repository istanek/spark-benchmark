# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1] - 2026-06-06

### Added

- **`spark_benchmark.quant_sweep` — post-processor and HTML tradeoff table.**
  - `aggregate_quant_sweep(aggregate, model_configs, fixture)` groups
    `aggregate_runs()` output by `base_model × quantization` using the
    `base_model` field already present on `ModelConfig`. Suite-name version
    suffixes (e.g. `hallucination_grounding_v1`) are stripped automatically.
  - `check_quant_regressions(sweep, fixture)` returns warning strings when
    any non-reference variant drops more than 5 pp below the reference
    threshold for a suite. Only fires when `enforce: true` on the base-model
    spec (currently `false` for all v1 entries until baselines are measured
    on hardware).
  - `load_quant_sweep_fixture(path)` loads and validates
    `data/quant/quantization_sweep_v1.json` via Pydantic.
  - `_render_quant_sweep_card` and `_render_quant_sweep_section` added to
    `reporting_html`. The card renders a per-base-model tradeoff table
    (Variant · Quant · Hallucination · Struct. output · Code pass@1 · TTFT ·
    tok/s · VRAM). Quality cells are colour-coded relative to the reference
    variant (green ≥ ref, amber within 5 pp, red > 5 pp below). Reference
    row is sorted first; remaining variants follow fixture order.
  - `render_canonical_report_html` now accepts `aggregate["quant_sweep"]` or
    an explicit `quant_sweep=` kwarg — the section is omitted when absent,
    so existing reports are unchanged.
  - `enrich_with_quant_sweep(aggregate, model_configs, repo_root, fixture_path)`
    added to `quant_sweep`. Detects quant-sweep runs automatically (any model
    with `base_model` set), loads the fixture, calls `aggregate_quant_sweep`,
    and injects the result into the aggregate dict in place. No-ops silently
    when no model carries a `base_model` or the fixture file is missing.
  - `cli.py` `benchmark` and `wizard` commands and `shell.py` TUI `do_run`
    now call `enrich_with_quant_sweep` after `aggregate_runs`, so the quant
    tradeoff table appears automatically in the HTML report whenever quant
    variants are benchmarked together.
- **21 new tests in `tests/test_quant_sweep.py`** covering fixture loading,
  aggregation grouping, suite-version stripping, missing-suite → null,
  regression enforcement (enforce=False silent, enforce=True fires, within-5pp
  silent, null-threshold skipped), and HTML smoke tests for the card and
  section renderers.

### Fixed

- `tests/test_long_context.py`: `test_existing_model_yaml_still_loads_without_base_model`
  renamed to `test_model_config_loads_without_base_model` and switched to a
  synthetic inline YAML so the test doesn't break when the real `qwen-3.6.yaml`
  gains a `base_model` field (as it now has).

## [0.5.0] - 2026-06-05

### Added

- **BYOT `mode: scored` — deterministic scorers (v0.3.0 milestone, shipped
  in v0.5.0 alongside the quant sweep infrastructure).**
  `CustomSuiteTask` now accepts an optional `scoring:` block. Five scorers:
  - `exact_match` — normalised case-insensitive string equality.
  - `substring_match` — all items in `must_contain` must appear in the output.
  - `regex_match` — a Python `re` pattern must match somewhere in the output.
  - `json_fields_match` — output must parse as JSON and contain all
    `expected_fields` keys with the expected values. Markdown fences are
    stripped automatically.
  - `multiple_choice` — a letter/word expected answer must appear as a whole
    word in the output.
  A suite-level `scoring:` block provides the default for tasks that don't
  specify their own. Tasks without any scorer are still run but produce
  `passed: null`. `validate_custom_suite` now warns when `mode: scored` is
  used but no scorer is configured for a task.
  `load_custom_suite` no longer rejects `mode: scored`.
- **`--dry-run` flag on `spark-bench run-custom`.** Executes one task against
  one model and stops without writing any files. The JSON output carries
  `"dry_run": true` and a single row. Useful for sanity-checking backend
  connectivity and suite config before a long sweep.
- **Per-task `timeout_s` enforcement.** `run_custom_suite_quick` now wraps
  each backend call in a `_task_timeout` context manager. On Unix platforms
  (where `SIGALRM` is available) a timeout triggers `TimeoutError`, which is
  recorded as an error row and the run continues. On Windows / non-main
  threads the timeout is a no-op (field is still recorded for forensics).
- **Quantization sweep infrastructure (v0.5.0 preview).**
  - `docs/quantization-sweep-spec.md` — implementation-ready spec.
  - `data/quant/quantization_sweep_v1.json` — fixture with reference
    thresholds for the v1 lineup (all `enforce: false` until baselines run).
  - `configs/experiments/spark-quant-sweep.yaml` — experiment covering Q8_0
    and Q4_K_M variants of all three base models.
  - Six new model config YAMLs: `qwen-3.6-q8`, `qwen-3.6-q4`, `gemma-4-q8`,
    `gemma-4-q4`, `nemotron-3-q8`, `nemotron-3-q4`.
  - `base_model` field added to the three existing curated model YAMLs
    (`qwen-3.6`, `gemma-4`, `nemotron-3`).
  The post-processor and HTML tradeoff table shipped in v0.5.1.
  Pull the quant variants (`ollama pull qwen3.6:35b-q8_0` etc.) and run the
  experiment; canonical suites produce per-model results that
  `aggregate_quant_sweep` groups into the tradeoff card automatically.
- **Long-context empirical findings in `METHODOLOGY.md`.** Documents the
  "lost in the middle" pattern across all four tested models (depth drives
  retrieval more than context length), prefill throughput numbers, and the
  v0.4.2 prompt-improvement rationale.

### Changed

- `run_custom_suite_quick` progress callback now includes the scorer method
  in the task line (`[exact_match]`) and prints `PASS` / `FAIL` with a
  reason snippet after each scored task.
- `render_custom_summary_markdown` renames "Per-model telemetry" to
  "Per-model summary" and adds Pass / Pass rate / Scored columns in
  `mode: scored`.
- `build_custom_summary` per-model buckets now always include `passes`,
  `scored`, and `pass_rate` fields (all zero/null in `mode: quick`).
- `validate_custom_suite` no longer emits an error for `mode: scored`
  (it now validates instead of rejecting).

### Tests

- 25 new tests in `tests/test_custom_suites.py` covering:
  - `ScoringConfig` schema validation (all 5 methods, missing required fields).
  - `score_response` for all scorers including edge cases (case-sensitivity,
    word-boundary for multiple_choice, JSON with markdown fences, invalid
    regex, missing JSON keys).
  - Runner integration: mode: scored pass/fail, suite-level default scorer,
    unscored task → null verdict, dry_run no files written.
  - `validate_custom_suite` warning for unscored tasks in scored mode.

## [0.4.5] - 2026-06-02

### Added

- **`run --model` (repeatable).** The `run` command can now target specific
  models by name/tag instead of always running the full resolved lineup. It
  accepts a curated experiment name, the raw Ollama tag, its slugified form,
  or an explicit Ollama Cloud `-cloud` tag — the latter is synthesized on the
  fly, so cloud models work without an experiment YAML entry or `/api/tags`
  listing. Example:
  `run --experiment … --platform spark --run-suite hallucination_grounding --model gpt-oss:120b-cloud`.

### Fixed

- **Docs:** the Ollama Cloud examples in `README.txt`, `docs/README.md`, and
  the v0.4.4 release notes used a non-existent `run --suite … --model …`
  invocation. Corrected to real commands (`quick --models …` and
  `run … --run-suite … --model …`) and listed the valid `--run-suite` values.

## [0.4.4] - 2026-06-02

### Added

- **Ollama Cloud support.** Benchmark hosted models (e.g.
  `gpt-oss:120b-cloud`, `deepseek-v3.1:671b-cloud`) with no config changes —
  set `OLLAMA_HOST=https://ollama.com` and `OLLAMA_API_KEY=…` and pick a
  cloud model by tag. The Ollama adapter now resolves its base URL from
  `$OLLAMA_HOST` (falling back to the configured endpoint, then localhost)
  and sends an `Authorization: Bearer` header from `$OLLAMA_API_KEY` on every
  request (generate / unload / `/api/ps` / `/api/tags`). New helpers
  `resolve_ollama_base`, `ollama_auth_headers`, `is_cloud_endpoint`. Model
  detection is auth-aware, and `find_config_by_name_or_tag` synthesizes a
  config for an explicit `-cloud` tag even when `/api/tags` doesn't list it
  (`synthesize_cloud_model_config`).
- **Security:** the API key is read from the environment only and is never
  copied into `BackendConfig.options`, manifests, or report payloads.

### Notes

- Cloud runs report speed (tokens/s, TTFT) and pass rates, but **no local
  GPU telemetry** — memory/power/temperature are unavailable remotely, so
  those charts stay empty. Cloud calls are billed and traverse the network,
  so long suites are slower.

## [0.4.3] - 2026-06-02

### Added

- **Long-context HTML report: pass-rate by needle type.** A multi-model
  run showed needle *category* drives retrieval as much as position does —
  e.g. alphanumeric codes (~17% across all models) are far harder than
  dates (~67%), because models garble or loop on codes even when they're
  clearly trying. `build_long_context_summary` now emits a per-model
  `categories` breakdown ({category, passes, n, pass_rate}), it's forwarded
  through `aggregate_runs`, and the report renders a model × needle-type
  pass-rate heatmap. `_svg_heatmap` gained a `left_pad` argument so long
  model names fit the row gutter.
- **`scripts/probe_refusal.py`** — a standalone diagnostic (stdlib +
  Ollama HTTP, no install needed) that varies only the prompt wording
  (baseline / instruction-at-end / forceful) at a fixed short context. It
  established that the depth-0 retrieval collapse is **not** a wording or
  refusal artifact (all three variants score identically), so the
  anti-refusal prompt is not the lever — needle type and position are.

## [0.4.2] - 2026-06-01

### Changed

- **Long-context scoring now ignores thousands separators.** A correct
  numeric answer was being failed purely on formatting — e.g. a model
  answering `1840` when the planted fact said `1,840`. `score_niah` now
  strips digit-group separators (comma, space, NBSP, narrow NBSP,
  apostrophe) that sit *between two digits*, so `1840` == `1 840` ==
  `1,840`. Non-numeric punctuation (such as the comma in
  `November 14, 2023`) is untouched, and genuinely different numbers still
  fail.
- **Needle-in-a-haystack prompt now states the fact is present.** A
  multi-model run showed every model scoring 0 % whenever the needle was
  *not* at the very end of the context — they treated the planted fact as
  out-of-place in the public-domain filler and refused ("not answerable").
  The prompt now uses standard NIAH framing ("a specific fact … has been
  inserted … the answer is present, do not reply that it is missing"), so
  the suite measures retrieval rather than refusal. Note: this changes
  prompt semantics, so long-context pass rates from 0.4.0/0.4.1 are not
  directly comparable to 0.4.2+.

## [0.4.1] - 2026-06-01

### Added

- **Fast / full profiles for the long-context suite.** The full
  needle-in-a-haystack grid is 4×4×8 = 128 cells per model and, dominated
  by long-context prefill (the 131k row alone is ~half the time), runs
  roughly an hour per model. There's now a **fast preview profile**
  (3 lengths `4096 / 32768 / 131072` × 3 depths `0/50/100` × 2 needles =
  18 cells, ~10 min/model) that still spans the full range up to 131k. It
  shows up in the TUI suite picker as a separate
  `long_context_retrieval_fast` entry and works on the CLI via
  `run --suite long_context_retrieval_fast`. Profiles live under a new
  `profiles` block in the fixture and are validated like the default grid;
  `LongContextFixture` gained `profiles`, plus `resolve_profile_matrix` /
  `profile_for_suite_name` helpers and a `matrix` override on the runner.

### Changed

- **Long-context suites are now opt-in in the TUI "Run" picker.** Because
  they're slow and the two profiles are mutually redundant, neither
  long-context entry is preselected by default — the five quick canonical
  suites stay checked and you tick the one long-context profile you want.
  The `Suites`/`Info` screens report the fast entry's actual (smaller)
  grid.

## [0.4.0] - 2026-06-01

### Added

- **Long-context retrieval in the interactive TUI + HTML reports (layer 3).**
  `long_context_retrieval` now shows up in the `spark-bench shell` suite
  picker and `Suites`/`Info` screens (grid-aware task counts:
  lengths × depths × needles/cell). A preflight checks the git-ignored
  Project Gutenberg corpora are present and cleanly skips the suite with a
  "run scripts/fetch_haystacks.sh" hint instead of crashing mid-run. The
  HTML report gains a dedicated long-context card: a per-model
  length×depth **pass-rate heatmap** (red→amber→green, N/A tiles for
  lengths a model can't load) via a new dependency-free `_svg_heatmap`
  helper, a **prefill-throughput-vs-length** line chart, and a
  **resident-memory-vs-length** chart (shown only when Ollama `/api/ps`
  yielded memory), plus a first-failure-length KPI strip. The aggregator
  forwards the per-cell grid through `aggregate_runs`.
- **`long_context_retrieval` runner (layer 2 of the v0.4.0 suite).**
  Single-needle NIAH execution across the fixture grid. For each model it
  iterates the `length × depth × needles_per_cell` matrix and writes one
  of three states per cell: `pass`/`fail` (ran and substring-scored),
  `skipped_unsupported` (length exceeds the model's claimed context), or
  `error` (backend raised — e.g. OOM — captured, never fatal). Built on
  the probe findings: each cell prompt carries a deterministic-but-unique
  nonce to defeat Ollama's prefill cache, `options.num_ctx` is set
  explicitly per request, the backend request timeout is bumped to
  ≥ 600 s for long prefills, and the reported context length is the
  backend's actual `prompt_eval_count` (never the char-based estimate).
  Memory is sourced from Ollama `/api/ps` (new
  `OllamaAdapter.memory_snapshot()`) since `nvidia-smi` reports N/A on the
  Spark's unified memory. Summary aggregates per-cell pass rates, average
  prefill tok/s, peak VRAM, and a first-failure-length per model, plus a
  per-model length×depth markdown heatmap table. Wired into both the
  orchestration bundle and the `run --suite long_context_retrieval` path;
  added `SamplingConfig.num_ctx`. Reporting/HTML heatmaps land in layer 3.
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

### Fixed

- **TUI: Esc now cleanly exits a selection back to the menu.** Pressing
  Esc/`q` in the model or suite multiselect (Run / Custom / Quick) was
  conflated with confirming an empty selection, so it printed a stray
  "(no models selected)" notice that looked like a dead-end sub-screen.
  Esc-cancel (`None`) is now distinguished from an empty confirm and
  returns silently to the main menu.

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
