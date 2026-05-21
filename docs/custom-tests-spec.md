# Custom user tests — design spec

Companion to `docs/architecture.md`, `docs/extensions-spec.md`,
`README.txt` (plain-language) / `docs/README.md` (markdown), and
`METHODOLOGY.md`. This document describes the
**Bring-Your-Own-Test (BYOT) subsystem** that lets users answer "how do
these models perform on **my** workload?", on top of the harness's
existing model + backend + telemetry infrastructure.

> **Scope.** The harness is **Spark-only**. Every example, default, and
> piece of advice in this document assumes you are running on an NVIDIA
> DGX Spark with Ollama serving the models locally. Custom tests inherit
> that scope — they exist to compare local model variants on Spark, not
> to do any cross-platform marketing.

## Why this exists

The canonical suites in `data/**/*.json` answer
**"which model is best on a fixed academic benchmark?"**. They are
useful as reference data, but they are not the question most people care
about.

Most people care about

- "Does this model write valid JSON for **my** schema?"
- "Does it correctly translate the **technical** Czech I work with?"
- "Does it write working PySide6 code for **my** stack?"
- "Is it fast enough on **my** prompt distribution under sustained load?"

Custom suites turn the harness from "interesting reference data" into
"a tool you actually use to make decisions about which local model to
pin to which workload".

## Two user journeys (collapsed into one CLI)

The early proposal split this into two top-level CLI commands
(`spark-bench quick` for "just show me the answers" and
`spark-bench run-custom` for "score them against expectations"). We
collapsed that to **one entrypoint** (`spark-bench run-custom`) with a
`mode:` field in the YAML so users only have to learn one command. The
two journeys still exist, just under one roof:

| Mode | What it does | Typical input | Output |
| --- | --- | --- | --- |
| **`quick`** | Pass-through: each prompt is sent to each model, telemetry is captured, no scoring. ~80 % of expected use. | A handful of prompts | Side-by-side Markdown summary you read on screen |
| **`scored`** | Each prompt is scored against an `expected:` block, a regex, a JSON schema, a custom Python function, or an LLM judge. | Tasks with `expected:` payloads | Pass/fail aggregate report + per-task drill-down |

`quick` is the only mode implemented in **v0.2.0**. `scored` ships in
v0.3.0+ — see the roadmap below.

## Schema (v0.2.0)

Every custom suite is one YAML or JSON file. No multi-file fan-out in
v0.2.0 — most users have ≤ 30 tasks and a single file is easier to
share, diff, and review. (`prompts_dir:` is on the v0.4.0 list for
suites that genuinely need hundreds of tasks.)

```yaml
name: my-czech-rag-test          # required, used for slug + report title
version: "1.0"                   # free-form; appears in manifest
description: |                   # optional one-liner shown at report top
  RAG-style queries against my domain corpus.
mode: quick                      # only "quick" is honoured in v0.2.0
models:                          # optional default model lineup
  - qwen-3.6
  - phi4-14b                     # auto-detected slug works here
sampling:                        # default sampling for every task
  temperature: 0.0
  top_p: 1.0
  seed: 42
  max_tokens: 1024
tasks:
  - task_id: q1                  # required, must be unique inside the suite
    prompt: |                    # required, multi-line OK
      Translate to Czech: "The early bird catches the worm."
    tags: [translation, czech]   # optional, copied to result rows
    sampling:                    # optional per-task sampling override
      temperature: 0.2
      max_tokens: 256
    timeout_s: 120               # accepted, NOT enforced in v0.2.0
```

### What's enforced today

| Check | When | Behaviour |
| --- | --- | --- |
| `name` non-empty | Pydantic load | `ValueError` |
| At least one task | Pydantic load | `ValueError` |
| Unique `task_id` per suite | Pydantic load | `ValueError` listing the duplicate |
| Non-empty `prompt` per task | Pydantic load | `ValueError` naming the task |
| `mode == "quick"` | `load_custom_suite` | `ValueError` pointing at v0.3.0 |
| `temperature` in `0..2` | `validate_custom_suite` | error issue |
| `max_tokens > 0` | `validate_custom_suite` | error issue |
| Long prompt (>32 k chars) | `validate_custom_suite` | warning issue |
| `models[]` references resolve | `validate_custom_suite(available_models=...)` | error issue if any unknown |

### What's accepted but not enforced today

- `timeout_s` per task — recorded in the result row for forensics; v0.3+
  will actually wrap the backend call in a watchdog.
- `mode: scored` — explicitly rejected at load time today so users get a
  clear error pointing to v0.3.0, not silent success.

## CLI surface (v0.2.0)

Two new commands, both backed by `spark_benchmark.custom_suites`:

```bash
# Schema + soft validation. Exits non-zero on any error issue.
spark-bench validate-custom path/to/suite.yaml \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark

# End-to-end run. --allow-auto-detected is ON by default for custom
# suites (the user explicitly opted into a non-canonical workload).
spark-bench run-custom path/to/suite.yaml \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark \
  [--models qwen-3.6,phi4-14b] \
  [--no-allow-auto-detected] \
  [--no-resume] \
  [--output-dir ./somewhere/else/]
```

Defaults:

- `--allow-auto-detected` is **ON** for `run-custom`. The flag default
  is the only place across the four CLI surfaces (`run`, `console`,
  `benchmark`, `wizard`) where auto-detection is on by default — and
  for good reason: custom suites only exist because the user is asking
  about something the curated YAML doesn't cover.
- `--output-dir` defaults to `results/custom/<slug>/<run-id>/` where
  `<slug>` is `slugify_suite_name(suite.name)` and `<run-id>` is the
  same `YYYYMMDDTHHMMSSZ-<8hex>` shape as the canonical run bundles.
- `--no-resume` starts fresh. Without it, the runner reads the existing
  `results.jsonl` and skips every `(model, task_id)` pair that's already
  there.

## Run bundle layout

Identical shape to canonical suites, plus `kind: custom` in the
manifest so reporting code can render a "Custom Test" badge instead of
the canonical headline framing:

```
results/custom/<slug>/<run-id>/
├─ manifest.json     # {kind: "custom", suite, suite_version, models, ...}
├─ results.jsonl     # one row per (model, task_id) — append-only
├─ summary.json      # per-model aggregate metrics + raw rows
└─ summary.md        # side-by-side Markdown (telemetry table + per-task)
```

A row in `results.jsonl` looks like:

```json
{
  "suite": "my-czech-rag-test",
  "suite_version": "1.0",
  "mode": "quick",
  "model": "phi4-14b",
  "model_tag": "phi4:14b",
  "task_id": "q1",
  "tags": ["translation", "czech"],
  "prompt": "...",
  "sampling": {"temperature": 0.2, "top_p": 1.0, "seed": 42, "max_tokens": 256},
  "timeout_s": 120,
  "generation": { "prompt": "...", "output": "...", "metrics": {...} },
  "error": null
}
```

If the backend raises, `generation` is `null` and `error` is
`{"type": "RuntimeError", "message": "..."}`. **The runner does not
abort on a single task failure** — every other `(model, task)` pair
keeps going, the error row is on disk, and the per-model summary
counts the failure under `tasks_errored`.

## Reporting

Mode A's report is intentionally not a pass/fail leaderboard. The
intent is "here are the answers, judge them yourself". The Markdown
output has:

1. **Suite metadata** — name, version, description, backend, task count.
2. **Per-model telemetry table** — completed / errored counts, mean
   TTFT, mean decode tps, total decode tokens, total wall time.
3. **One section per task**, with the prompt rendered as a fenced block
   followed by every model's reply (also fenced) prefixed with TTFT,
   tokens-per-second, and finish reason.

The same `summary.json` is structured so future `scored`-mode reports
can extend the same shape (add `score`, `passed`, `details` keys) and
still render with the same template.

Custom test reports are visually distinct from canonical reports so
they cannot be mistaken for the headline benchmark numbers — see
`reporting.write_report` (canonical) vs. the bundle-local `summary.md`
written directly by the custom runner.

## Reuse, not parallel infrastructure

Custom suites share **everything** with canonical suites:

| Concern | Where it lives | Reused unchanged |
| --- | --- | --- |
| Backend adapters | `runners/registry.py` | yes |
| Sampling / generation contract | `models.py:GenerationResult` | yes |
| Telemetry capture | per-backend adapter (e.g. `OllamaAdapter.generate`) | yes |
| Model resolution + auto-detection | `model_registry.resolve_runnable_models` | yes |
| Run-bundle helpers (`make_run_id`, `write_result`, `write_json`) | `results_bundle.py` | yes |

What is **not** reused:

- `suites.SuiteDefinition` — that's the canonical JSON-only schema with
  `expected_behavior` flags hard-wired into the reliability suite. The
  user-facing custom format lives in
  `spark_benchmark.custom_suites.CustomSuiteDefinition` so canonical
  suites are not perturbed by user-driven schema changes.

## Roadmap beyond v0.2.0

Each phase is intended to be a single release boundary, not a
multi-feature blob. Phases land in order; the harness stays usable
between releases.

### v0.3.0 — `mode: scored` with deterministic scorers

Add `scoring:` to `CustomSuiteTask` and `CustomSuiteDefinition` (default
inherited from suite, optional per-task override):

| Scorer | Config |
| --- | --- |
| `exact_match` | `expected: "string"` |
| `substring_match` | `must_contain: ["substring", "..."]` |
| `regex_match` | `pattern: "..."` |
| `json_fields_match` | `expected_fields: {...}` (subset match against JSON) |
| `multiple_choice` | `expected: "B", choices: [A,B,C,D]` |

Plus the `dry-run` flag on `run-custom` that picks one task and one
model and runs them end-to-end before committing to the full bundle.

Out of scope for v0.3.0: `numeric_tolerance`, `json_schema` (the latter
needs a `jsonschema` dependency).

### v0.4.0 — custom Python scorers

User-supplied scoring functions (`./scorers/my_scorer.py:score`) with
the **same sandbox** the `code_generation` suite already uses
(`subprocess + resource.setrlimit + timeout`). Sandbox is **on by
default**; opt-out via `--unsafe-scorers`. Imported test bundles always
sandbox.

Also in v0.4.0: per-task timeout enforcement (the field is already in
the schema; v0.4.0 wraps the backend call in a watchdog so a single bad
prompt cannot hold a five-model run hostage).

### v0.5.0 — local LLM-as-judge

`judge_rubric` and `judge_binary` scorers, but **only with a local
Ollama judge model**. The judge runs against a separate
`configs/models/<judge-name>.yaml` so the harness stays fully offline.
Cloud judges (Claude / GPT) are explicitly **not** in v0.5.0; they would
introduce credentials, rate limits, and cost tracking that aren't worth
the complexity for a Spark-only project.

Pairwise / preference / multi-judge agreement is research territory and
is not on the v0.5.0 list either.

### v0.6.0+ — sharing & ecosystem

`prompts_dir:` for large suites, suite bundle export/import via plain
zip with a security warning when imported bundles contain custom Python,
optional `compare` command for diffing two custom runs, and (much
later) a `suites/community/` namespace for promoting battle-tested
custom suites into the canonical tree.

## Out of scope (everywhere)

- **Web UI for building tests.** Custom suites are YAML files; if a UI
  ever happens it is a separate tool that emits YAML.
- **Test marketplace / registry.** Share via git or zip. The harness is
  not a package manager.
- **Multimodal custom tests** (image + text). Text-only across every
  phase above.
- **Cross-platform comparisons.** v1 of this entire project is
  Spark-only; custom suites do not change that.

## Acceptance criteria for v0.2.0

All of these must hold before `0.2.0` is tagged:

- `spark-bench run-custom examples/custom-tests/quick/suite.yaml --experiment ... --platform spark` produces a non-empty `results.jsonl`, `summary.json`, and `summary.md` under `results/custom/example-quick-test/<run-id>/`.
- `spark-bench validate-custom <bad.yaml>` exits non-zero and prints a clear `ERROR ...` line for: duplicate task IDs, empty prompts, `mode: scored`, unknown `models[]`.
- The runner records errors as rows and keeps going (covered by `test_run_custom_suite_records_errors_without_aborting`).
- Re-running against the same `--output-dir` skips already-completed pairs (covered by `test_run_custom_suite_resume_skips_already_done_pairs`).
- A copy of the example template is committed under
  `examples/custom-tests/quick/` with a `README.md` explaining the
  shape and the run command.
- The CLI commands are documented in `README.txt`, `docs/README.md`,
  `CHANGELOG.md`, and `docs/architecture.md`.
