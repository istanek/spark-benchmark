# Contributing

Thanks for poking around. The harness is small on purpose; the rules below
keep it small as it grows.

## Workflow

- Branch off `main`. Use feature branches like `feat/<short-name>` or
  `fix/<short-name>`. Don't push directly to `main` — it is protected.
- Open a Pull Request from your branch. Keep PRs small and focused.
- Use **conventional commit** prefixes in titles and commit messages:
  `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `ci:`.
- Reference GitHub issues with `#<issue-number>` when relevant.

## Local development

```bash
git clone https://github.com/istanek/spark-benchmark.git
cd spark-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/
```

Or run a single test file without pytest:

```bash
PYTHONPATH=src python3 tests/test_reliability.py
```

The CLI is wired up by the package install but also runnable raw:

```bash
PYTHONPATH=src python3 -m spark_benchmark.cli wizard \
  --experiment configs/experiments/spark-ollama-baseline.yaml --platform spark
```

## Code conventions

See `.cursor/rules/python-conventions.mdc` for the canonical list. Short
version:

- Python ≥ 3.11. Use `from __future__ import annotations` in new modules.
- Pydantic v2 only.
- Backends raise `RuntimeError` with the HTTP body / stderr — never silently
  swallow.
- `code_generation.sandbox_run` must keep `subprocess + setrlimit + timeout`.
  Never run generated code in-process.

## Adding a new suite

Long form lives in `docs/architecture.md` § 5 "Extension recipes". Short form:

1. Drop a fixture JSON under `data/<category>/<name>_v<n>.json`. It must
   validate against `SuiteDefinition` and use the right
   `metadata.expected_behavior` for the scorer you want
   (`.cursor/rules/fixtures-and-configs.mdc` has the full table).
2. Implement `run_<name>_suite(*, run_dir, suite, backend, backend_config,
   model_configs, sampling, progress_callback=None)` following the existing
   per-row write pattern (`backend.load_model` → loop → `backend.generate` →
   `write_result` → `backend.unload`).
3. Wire dispatch in `orchestration.run_benchmark_bundle`.
4. Add the suite to `shell.SUITE_REGISTRY` (label + data path) so the TUI
   picker shows it.
5. Add NL keywords to `orchestration.parse_benchmark_request` so the
   `benchmark` CLI command can pick it up from a sentence.
6. Add a test under `tests/` that loads the fixture and exercises the scorer.

## Adding a new backend

1. Create `runners/<backend>.py` implementing the `BackendAdapter` protocol
   (`load_model`, `generate`, `get_metrics`, `unload`).
2. Add `BackendKind.<NAME>` in `models.py` and wire it in
   `runners/registry.build_backend`.
3. Drop a default config in `configs/backends/<name>.yaml`.
4. Reference it from an experiment YAML (`backend: <name>`).
5. Extend `tests/test_backend_registry.py` with a dispatch check.

## Adding a new model

For Ollama-served models you no longer need a YAML — pull the model and the
TUI auto-detects it (vision / embedding tags get greyed out). To pin
sampling, context length, or aliasing in the wizard or NL parser, add
`configs/models/<name>.yaml` and reference it in an experiment's `models:`
list.

## Releasing

Releases are cut from `main` by the maintainer. The process:

1. Move the relevant entries from `## [Unreleased]` to a new
   `## [X.Y.Z] - YYYY-MM-DD` section in `CHANGELOG.md`.
2. Commit on `main`:
   ```bash
   git add CHANGELOG.md && git commit -m "docs: prepare X.Y.Z changelog"
   git push origin main
   ```
3. Run the release script. It validates the working tree, extracts the
   matching CHANGELOG section, creates an annotated tag, pushes it, and
   creates a GitHub Release with the same body:
   ```bash
   scripts/release.sh 0.2.0              # cut v0.2.0
   scripts/release.sh 0.2.0 --dry-run    # preview without changing anything
   ```

Requirements: `jq`, `curl`, `python3`, and a GitHub token. The script
reads the token from `GITHUB_TOKEN` first, then falls back to
`gh auth token` (if the GitHub CLI is installed and signed in), and
finally to a token stored in `~/.git-credentials` for `github.com`.
The token must have the `repo` scope (or, on a fine-grained PAT,
`Contents: read & write`) to create a Release.

## Tests + CI

- The CI runs YAML / JSON fixture validation and `pytest tests/` on every
  push and pull request to `main`
  (see `.github/workflows/ci.yml`).
- Keep tests deterministic. No reliance on Ollama, NVML, or the network
  inside `tests/` — those code paths are exercised manually via the CLI.

## Honesty notes (from `METHODOLOGY.md`)

Publish failures, not just wins. If a model fails a suite or a backend
returns garbage, document it in the report rather than dropping the row.
