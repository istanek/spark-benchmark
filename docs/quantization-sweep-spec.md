# Quantization sweep — design spec (v0.5.0)

Companion to `docs/architecture.md`, `docs/long-context-spec.md`,
`docs/custom-tests-spec.md`, `README.txt` / `docs/README.md`, and
`METHODOLOGY.md`. This document is the implementation-ready plan for the
`quantization_sweep` suite introduced in v0.5.0.

> **Scope.** The harness is Spark-only. Every quantization variant tested
> here runs locally on a DGX Spark via Ollama. No cross-platform comparison,
> no cloud calls for the base runs (Ollama Cloud models may appear as a
> reference point but are labelled distinctly).

---

## Where this sits in the roadmap

| Release | Suite | Status |
|---|---|---|
| v0.3.0 | Marketing-grade HTML reports | ✅ done |
| v0.4.0 | `long_context_retrieval` | ✅ done |
| **v0.5.0** | `quantization_sweep` (this doc) | next |
| v0.6.0 | `concurrent_serving` | future |

`quantization_sweep` is deliberately before `concurrent_serving` because it
requires almost no new runner code — it re-runs *existing* quality and performance
suites against multiple quantization variants of the same base model, then groups
the results by `base_model` for a quality-vs-speed-vs-VRAM tradeoff table.

---

## What this suite measures, in plain terms

Take one base model (e.g. Qwen-3-35B), pull multiple quantization variants
(`q4_k_m`, `q8_0`, `fp16`), and run them through the same canonical suites.
The output answers:

1. **How much quality do you lose** going from FP16 to Q4_K_M?
2. **How much faster / smaller** is Q4_K_M compared to FP16?
3. **At what quantization level does quality visibly degrade?**

This is the most common practical question a Spark user faces when selecting a
model for a workflow.

---

## Design decisions

### 1. Re-use existing suite runners, no new runner code

`quantization_sweep` is a **post-processor and reporting layer**, not a new
runner. It:

1. Resolves a set of `ModelConfig` entries that share the same `base_model`
   field and differ only in `quantization`.
2. Runs the existing suite runners (one or more of: `hallucination_grounding`,
   `practical_structured_output`, `code_generation`, `openclaw_speed`) for
   each variant.
3. Calls a new `aggregate_quant_sweep` function that groups results by
   `base_model` and emits a tradeoff table + charts.

No changes to the runner core, no new backend calls, no new scoring logic.

### 2. `base_model` is an explicit YAML field

Already added to `ModelConfig` in v0.4.0. Example:

```yaml
# configs/models/qwen-3.6-q4.yaml
name: qwen-3.6-q4
family: qwen
base_model: qwen3-35b          # grouping key — matches across all quant variants
revision: qwen3.6:35b-q4_k_m
quantization: Q4_K_M
source: ollama-local
context_length: 131072
artifact_path: qwen3.6:35b-q4_k_m
```

### 3. Suites to run

Not all suites are equally informative for a quant comparison. Recommended set:

| Suite | Why |
|---|---|
| `hallucination_grounding` | Quality signal — does quantisation cause more hallucination? |
| `practical_structured_output` | JSON compliance — does Q4 corrupt structured output? |
| `code_generation` | Pass@1 — is there a visible cliff in code quality? |
| `openclaw_speed` | TTFT + decode tok/s — the primary performance axis |

`sustained_throughput` and `long_context_retrieval` are optional (slow, but
will show VRAM headroom differences across quants).

### 4. Tradeoff table format

Per `base_model`, the HTML report renders a table like:

| quant | halluc pass% | JSON pass% | code pass@1 | TTFT ms | tok/s | VRAM MB |
|---|---|---|---|---|---|---|
| FP16 | 100 % | 100 % | 80 % | 200 ms | 25 | 70 000 |
| Q8_0 | 100 % | 100 % | 80 % | 120 ms | 42 | 38 000 |
| Q4_K_M | 89 % | 94 % | 70 % | 90 ms | 55 | 20 000 |

Cells are colour-coded: quality columns (green = ≥ FP16, amber = within 5 pp,
red = > 5 pp below FP16). Speed/VRAM columns are inverted (lower = greener).

### 5. No composite score

Do not blend quality and speed into a single number. Report each axis
separately and let the user make the tradeoff. The METHODOLOGY.md principle
"no composite quality score" applies here too.

### 6. Fixture

The suite uses the *same fixtures* as the canonical runs
(`hallucination_grounding_v1.json`, etc.) — no new fixture file needed for
the basic sweep. A dedicated `quantization_sweep_v1.json` is needed only for:
- metadata (which `base_model` groupings are expected),
- reference quality thresholds per quant level (so the report can flag
  unexpected regressions), and
- the list of recommended suites to run.

This document defines that fixture schema below.

---

## Fixture schema: `data/quant/quantization_sweep_v1.json`

```json
{
  "name": "quantization_sweep_v1",
  "category": "quant",
  "version": "0.5.0",
  "description": "Reference quality thresholds per base model × quantization level.",
  "notes": [
    "Quality thresholds come from the v0.4.x canonical runs (FP16 / Ollama-default).",
    "Populate reference_pass_rates once the first full sweep is complete.",
    "enforce: false until baselines are confirmed on real hardware."
  ],
  "recommended_suites": [
    "hallucination_grounding",
    "practical_structured_output",
    "code_generation",
    "openclaw_speed"
  ],
  "base_models": [
    {
      "base_model": "qwen3-35b",
      "display_name": "Qwen-3 35B",
      "variants": ["qwen-3.6", "qwen-3.6-q8", "qwen-3.6-q4"],
      "reference_variant": "qwen-3.6",
      "reference_pass_rates": {
        "hallucination_grounding": null,
        "practical_structured_output": null,
        "code_generation": null
      },
      "enforce": false
    },
    {
      "base_model": "gemma4-27b",
      "display_name": "Gemma-4 27B",
      "variants": ["gemma-4", "gemma-4-q8", "gemma-4-q4"],
      "reference_variant": "gemma-4",
      "reference_pass_rates": {
        "hallucination_grounding": null,
        "practical_structured_output": null,
        "code_generation": null
      },
      "enforce": false
    },
    {
      "base_model": "nemotron3-33b",
      "display_name": "Nemotron-3 33B",
      "variants": ["nemotron-3", "nemotron-3-q8", "nemotron-3-q4"],
      "reference_variant": "nemotron-3",
      "reference_pass_rates": {
        "hallucination_grounding": null,
        "practical_structured_output": null,
        "code_generation": null
      },
      "enforce": false
    }
  ]
}
```

---

## Experiment YAML: `configs/experiments/spark-quant-sweep.yaml`

See `configs/experiments/spark-quant-sweep.yaml` (created alongside this doc).

---

## Model config YAML stubs

Three model configs per base model × 3 quant levels = up to 9 new YAML files
under `configs/models/`. The Ollama tags follow the `<model>:<size>-<quant>`
convention. Tags are placeholders until confirmed via `ollama list`.

Example stub:

```yaml
# configs/models/qwen-3.6-q4.yaml
name: qwen-3.6-q4
family: qwen
base_model: qwen3-35b
revision: qwen3.6:35b-q4_k_m
quantization: Q4_K_M
source: ollama-local
context_length: 131072
artifact_path: qwen3.6:35b-q4_k_m
notes:
  - quantization sweep variant (Q4_K_M)
  - pull with: ollama pull qwen3.6:35b-q4_k_m
```

---

## New code — `spark_benchmark.quant_sweep`

A new module (≤ 150 lines) with two public functions:

```python
def aggregate_quant_sweep(
    run_dirs: list[Path],
    fixture: QuantSweepFixture,
) -> dict[str, BaseModelSweepResult]:
    """Group multi-suite results by base_model × quantization.

    Reads summary.json from each suite dir, matches ModelConfig.base_model,
    and emits per-base-model tradeoff tables. Called by the aggregator and
    the HTML renderer.
    """

def check_quant_regressions(
    sweep: dict[str, BaseModelSweepResult],
    fixture: QuantSweepFixture,
) -> list[str]:
    """Return warning strings for quality regressions vs reference thresholds.

    Returns empty list when enforce=False on the relevant base model.
    """
```

The HTML renderer gains a `_render_quant_sweep_card` helper that emits the
tradeoff table with colour-coded cells (reuses `_band_for_pass_rate`) and
bar charts for speed and VRAM.

---

## CLI surface

No new CLI command. The existing `spark-bench run` and `spark-bench wizard`
surfaces work unchanged — the quant sweep is just another bundle in the
canonical report. The `aggregate` command already forwards `run_dir` so the
HTML renderer can pull the right suite summaries.

Optional convenience: `spark-bench benchmark "otestuj qwen na rychlost a
kvalitu pro q4 q8 fp16"` routes to a `BenchmarkPlan` with all three quant
variants selected (once model names are resolvable). Alias routing
(`qwen-3.6-q4` → `qwen3-35b Q4_K_M`) does not require NL changes — the
existing `--allow-auto-detected` path handles it.

---

## Implementation order

1. Add `data/quant/quantization_sweep_v1.json` fixture (this document).
2. Add model config YAML stubs for Q8 and Q4 variants of the v1 lineup.
3. Add `configs/experiments/spark-quant-sweep.yaml`.
4. Pull the quantization variants on Spark: `ollama pull qwen3.6:35b-q8_0` etc.
5. Run one sweep, populate `reference_pass_rates` in the fixture.
6. Implement `spark_benchmark.quant_sweep` (aggregate + regression check).
7. Wire `_render_quant_sweep_card` into the HTML report.
8. Write tests (fixture loading + schema, aggregation grouping, regression
   check enforce=False/True, HTML card).
9. Update README.txt, docs/README.md, CHANGELOG.md.

Estimated effort: ~2 weeks.

---

## Out of scope for v0.5.0

- Mixed-precision or layer-wise quantization (not Ollama-native).
- Automatic tag discovery for quant variants (user must list them in YAML).
- Benchmarking quantization of vision models (text-only throughout v1).
- Automated `ollama pull` — the harness never pulls models; it reports
  which ones are missing and exits with a clear error.
