# `examples/custom-tests/quick`

A minimal Mode A custom suite — three prompts, two models, no scoring.

This is the canonical "Bring Your Own Test" starting point. Copy it,
edit the prompts and models, and run.

## How to use it

```bash
# 1. Make a copy you can edit freely
cp -r examples/custom-tests/quick my-czech-rag-test
$EDITOR my-czech-rag-test/suite.yaml

# 2. Validate before you run (catches typos, missing fields, unknown models)
PYTHONPATH=src python3 -m spark_benchmark.cli validate-custom \
  my-czech-rag-test/suite.yaml \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark

# 3. Run it
PYTHONPATH=src python3 -m spark_benchmark.cli run-custom \
  my-czech-rag-test/suite.yaml \
  --experiment configs/experiments/spark-ollama-baseline.yaml \
  --platform spark
```

The output is one bundle directory under `results/custom/<slug>/<run-id>/`
containing:

- `manifest.json` — what was run (suite, suite version, models, backend).
- `results.jsonl` — one row per `(model, task_id)` with the raw output
  and per-call telemetry.
- `summary.json` — per-model aggregates (mean TTFT, mean decode tps,
  total decode tokens, wall time).
- `summary.md` — side-by-side Markdown report. Per-model telemetry
  table at the top, then one section per task with each model's reply
  rendered as a fenced block.

## What's in `suite.yaml`

| Field | Meaning |
| --- | --- |
| `name` | Human-readable suite name. Used for the report title and the slug under `results/custom/`. |
| `version` | Free-form version string. Appears in the manifest and report. |
| `description` | Optional one-liner shown at the top of the Markdown summary. |
| `mode` | Always `quick` in v0.2.0. `scored` will be accepted in v0.3.0. |
| `models` | Optional default lineup. Override with `--models a,b,c` on the CLI. |
| `sampling` | Default sampling. Per-task `sampling:` entries override individual fields. |
| `tasks[].task_id` | Unique identifier — must not repeat inside the suite. |
| `tasks[].prompt` | The actual prompt sent to the model. Multi-line YAML strings work fine. |
| `tasks[].tags` | Free-form tags, copied to the result row for filtering later. |
| `tasks[].sampling` | Optional per-task override (e.g. `max_tokens: 1024` for a longer answer). |
| `tasks[].timeout_s` | Per-task timeout. **Recorded in v0.2.0 but not enforced** — see `docs/custom-tests-spec.md`. |

## Resume

If a run dies half-way through (Ollama crashes, somebody Ctrl-C's the
process), point `run-custom` at the same `--output-dir` and the runner
skips every `(model, task_id)` pair that is already in `results.jsonl`.
Pass `--no-resume` to start fresh.

## Auto-detected models

By default `run-custom` lets you reference any chat-capable Ollama tag
even if it has no curated YAML — same `--allow-auto-detected` rules as
`spark-bench wizard` and `spark-bench benchmark`. So
`--models phi4-14b` works as long as `ollama pull phi4:14b` has run on
the box.

Pass `--no-allow-auto-detected` to restrict the lineup to the curated
experiment YAML.
