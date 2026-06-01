# Long-context retrieval — design spec (v0.4.0)

Companion to `docs/architecture.md`, `docs/extensions-spec.md` (the
original, aspirational long-context sketch), `docs/custom-tests-spec.md`,
`README.txt` / `docs/README.md`, and `METHODOLOGY.md`. This document is
the **implementation-ready** version of the `long_context_retrieval`
suite, written against the real v0.3.0 codebase. Where it disagrees with
the older sketch in `docs/extensions-spec.md` (Suite 1), **this document
wins** — the older one predates the harness and assumes structure the
code does not have.

> **Scope.** The harness is **Spark-only**. Every length, default, and
> hypothesis here assumes an NVIDIA DGX Spark with its 128 GB unified
> memory, serving models locally via Ollama (or llama.cpp). There is no
> Mac mini, no cross-platform marketing, no "Spark vs. X" framing in
> this suite. The long-context story *is* the Spark story, but we tell
> it by reporting honest Spark numbers, not by staging a comparison the
> harness is not built to run.

---

## Where this sits in the roadmap

The agreed release ladder (smallest-to-largest engineering cost, so the
repo ships something every cycle and the runner abstraction stays stable
until the suite that actually needs to bend it):

| Release | Suite | Why this slot | Rough cost |
| --- | --- | --- | --- |
| **v0.3.0** ✅ | Marketing-grade HTML reports | Already shipped/tagged | — |
| **v0.4.0** | `long_context_retrieval` (this doc) | The core Spark value proposition; mostly additive | 3–4 weeks |
| **v0.5.0** | `quantization_sweep` | Almost no new runtime code — reuses existing quality suites; high community value | ~2 weeks |
| **v0.6.0** | `concurrent_serving` | The only suite that reworks the runner core (threaded clients, `llama-server` mode, `OLLAMA_NUM_PARALLEL`) — gets its own cycle | 4–6 weeks |

`quantization_sweep` is deliberately **before** `concurrent_serving`
(the reverse of the original sketch): it is cheap, data-heavy, and does
not touch the runner. Concurrency is real engineering and should not
block two easy wins.

---

## What this suite measures, in plain terms

Give a model a long document with one specific sentence (a "needle")
hidden inside it, then ask a question whose answer is only in that
sentence. If the model's reply contains the expected answer, it passed.
Repeat across:

- **Context lengths** — how big the document is (in *the model's own*
  tokens).
- **Needle depths** — where in the document the needle sits (start →
  end).

The output is a **heatmap** per model: length on one axis, depth on the
other, colour = pass rate. A model that "supports 128k context" but goes
blind past 32k shows up as a red bottom band — exactly the failure the
Spark memory story needs to surface honestly.

---

## Decisions locked in (approved)

1. **Public-domain haystacks.** Source text comes from **Project
   Gutenberg** (public domain) — e.g. Joseph Conrad / Mark Twain prose
   for the "literary" haystack and a permissively-licensed technical
   corpus (Kubernetes docs, Apache-2.0) for the "technical" haystack.
   **No Paul Graham essays** (copyrighted). Each shipped haystack file
   records its source URL and license in the fixture.
2. **`ModelConfig.base_model` is an explicit YAML field**, not parsed
   from the model name. Added as an optional field (back-compatible). We
   do *not* infer it with a regex — that is brittle and silently wrong
   for odd names. Used here only for grouping/labelling; it matters more
   for `quantization_sweep` in v0.5.0, but we add it now so the schema
   stops moving.
3. **No composite "quality score."** (This decision is about the
   v0.5.0 quant sweep, but the principle applies here too: report
   per-cell and per-length pass rates directly. The headline is the
   heatmap and the "first-failure length" table, never a single blended
   number that gets screenshotted out of context.)

---

## Design refinements vs. the original sketch

The `docs/extensions-spec.md` Suite 1 sketch needs these changes to be
buildable and statistically honest:

### Part A only — substring match, no LLM judge

The original split single-needle (Part A) and multi-needle reasoning
(Part B). **v0.4.0 ships Part A only.** Multi-needle requires an
LLM-as-judge, which the harness does not have and which introduces a new
dependency, a new failure mode, and non-determinism. Scoring is a
deterministic, case-insensitive, whitespace-normalised substring match —
the same heuristic-scorer philosophy as `reliability.py`. Multi-needle
defers to v0.5.0+ alongside the BYOT "scored" judge work.

### Smaller grid, more samples per cell

The sketch proposed 6 lengths × 5 depths × 3 needles × 2 haystacks =
180 tasks/model, which leaves only **6 samples per (length, depth)
cell** — a 95 % CI of roughly ±20 pp. The heatmap would be visual
noise.

**v0.4.0 grid:**

- Lengths: **4** → `[4096, 16384, 65536, 131072]`
- Depths: **4** → `[0, 33, 66, 100]` (% of context)
- Needles per cell: **8**
- Haystacks: **1 per cell**, but *rotated deterministically* across
  cells so both corpora get exercised.

→ **4 × 4 × 8 = 128 tasks/model**, with **8 samples per cell**. Fewer
cells, but each one means something. The grid is config-driven, so a
"deep dive" run can widen it later.

### Per-model tokenization (the big one)

Different model families tokenize the same text differently (Llama vs.
Qwen vs. Gemma can differ ~5–10 % on token counts). Targeting "65 536
tokens" with a fixed OpenAI tokenizer would build a *different* test for
each model. So:

- At suite start, for each model, take the raw haystack text and
  tokenize with **that model's tokenizer**, truncate to the requested
  length, cache the prepared haystack to disk (keyed by
  `(model, haystack, length)`).
- The fixture stores **raw text + a target length**, never pre-tokenized
  blobs.
- Every task's telemetry records the **actual** tokenized length, and
  the report shows actual (not target) counts.

Tokenizer access: prefer the backend's own tokenizer where exposed
(Ollama `/api/embed` or tokenize endpoints, llama.cpp tokenize). If a
model's tokenizer is unavailable, fall back to a character-per-token
heuristic and **flag the length as approximate** in the report rather
than silently lying.

### Inline SVG charts — no matplotlib

The sketch wanted matplotlib PNGs. **No.** The v0.3.0 reporting
philosophy is single-file, no-JS, no-CDN, no-extra-deps HTML, and we
already have an SVG helper kit in `reporting_html.py`
(`_svg_bars`, `_svg_line_chart`, `_svg_gauge`, `_svg_dual_bars`,
`_svg_stacked_bars`, `_svg_thermometer`, `_pass_fail_strip_html`,
`_gradient_color_for_ratio`, `_cell_pct_html`). Long-context adds:

- `_svg_heatmap(grid, …)` — an N×M grid of `<rect>`s coloured via
  `_gradient_color_for_ratio`. ~100 lines.
- Reuse `_svg_line_chart` for the prefill-tokens/sec-vs-length curve.

**Anti-goal: no new runtime dependency for reporting.** A markdown table
remains the graceful fallback when running headless.

### Deterministic needle selection

Which 8 needles land in a given cell must be reproducible. Selection is
`needle_idx = stable_hash((length, depth, repetition)) % len(needles)`,
and haystack rotation is `haystack_idx = stable_hash((length, depth)) %
len(haystacks)`. Same fixture + same grid → byte-identical task plan
across runs.

### Per-model context limits → three cell states

Read each model's claimed max from the **existing**
`ModelConfig.context_length` field (no new field needed; the sketch's
`max_context_tokens` does not exist — `context_length` does). For each
`(model, length)`:

- `length > model.context_length` → **`skipped_unsupported`** (don't
  run; render as N/A with the claimed limit as the reason).
- `length <= context_length` but the backend OOMs / errors → **`oom`**
  (capture telemetry, record the event, continue — never crash the
  suite).
- otherwise → run, score pass/fail.

The report shows three states per cell: **pass-rate fill**, **N/A
(unsupported)**, **OOM**. "Claims 128k, OOMs at 64k on Q4_K_M" is a
headline result, not a missing data point.

---

## Fixture format

Lives at `data/long_context/long_context_retrieval_v1.json`, following
the `data/<category>/<suite>_v<n>.json` convention. It is a normal
`SuiteDefinition`-adjacent JSON, but with long-context-specific blocks
(the suite loader reads it directly rather than via the generic
`load_suite_definition`, the same way `code_generation` carries its own
benchmark structure).

```json
{
  "name": "long_context_retrieval_v1",
  "category": "reliability",
  "version": "0.4.0",
  "description": "Single-needle NIAH at 4k–128k, public-domain haystacks, substring scoring.",
  "notes": [
    "Part A (single-needle) only; multi-needle reasoning deferred.",
    "Haystacks are Project Gutenberg (public domain) + Apache-2.0 docs."
  ],
  "haystacks": {
    "literary_conrad": {
      "source_url": "https://www.gutenberg.org/ebooks/219",
      "license": "Public Domain (Project Gutenberg)",
      "text_file": "data/long_context/haystacks/conrad_heart_of_darkness.txt"
    },
    "technical_k8s": {
      "source_url": "https://github.com/kubernetes/website",
      "license": "Apache-2.0",
      "text_file": "data/long_context/haystacks/k8s_docs_subset.txt"
    }
  },
  "needles": [
    {
      "id": "code_7B_MIRA",
      "category": "alphanumeric_code",
      "text": "The secret access code for the maintenance hatch is 7B-MIRA-4419.",
      "question": "What is the secret access code for the maintenance hatch?",
      "expected_substring": "7B-MIRA-4419"
    },
    {
      "id": "date_project_zenith",
      "category": "date",
      "text": "Project Zenith was officially launched on November 14, 2023.",
      "question": "When was Project Zenith officially launched?",
      "expected_substring": "November 14, 2023"
    }
  ],
  "test_matrix": {
    "context_lengths_tokens": [4096, 16384, 65536, 131072],
    "depth_percentages": [0, 33, 66, 100],
    "needles_per_cell": 8,
    "haystacks": ["literary_conrad", "technical_k8s"]
  }
}
```

Ship **≥ 8 needles spanning ≥ 3 categories** (alphanumeric code, date,
named entity) so a cell can draw 8 distinct needles without repeats.

---

## Scoring

```python
def score_niah(response: str, expected: str) -> tuple[bool, dict]:
    """Case-insensitive substring match with whitespace normalisation."""
    norm = lambda s: " ".join(s.lower().split())
    passed = norm(expected) in norm(response)
    return passed, {
        "matched": passed,
        "response_length": len(response),
        "expected": expected,
    }
```

Deterministic, no judge. A future v0.5.0 may add a fuzzy/judge fallback
for paraphrased answers, gated behind a config flag.

---

## Codebase integration (the real wiring)

This is the part the original sketch got structurally wrong. The repo
uses **flat modules**, a **single `reporting_html.py`**, a **central
fixture-path registry**, and a **`_CANONICAL_SUITES` tuple**. There is
no `suites/` subpackage, no `reporting/` subpackage, no
`post_processors/`. The suite must thread through all of these:

| Touch point | File | What changes |
| --- | --- | --- |
| Suite runner | **`src/spark_benchmark/long_context.py`** (NEW, flat module — sibling of `reliability.py`, `code_generation.py`) | Loader, per-model tokenize+truncate, needle insertion, run loop, substring scorer, three-state cell logic |
| Fixture registry | `src/spark_benchmark/reliability.py` → `fixture_path_for_suite_name` | Add `long_context_retrieval` → `data/long_context/long_context_retrieval_v1.json` |
| Orchestration | `src/spark_benchmark/orchestration.py` | Import + dispatch the runner; add NL aliases ("long context", "needle", "dlouhý kontext") to `parse_benchmark_request` |
| Aggregation | `src/spark_benchmark/reporting.py` → `aggregate_runs` | New if-branch extracting heatmap grid + first-failure length + prefill curve into the model bucket's `extra` |
| Canonical suite list | `src/spark_benchmark/reporting_html.py` → `_CANONICAL_SUITES` | Append `"long_context_retrieval"` |
| HTML dashboard | `src/spark_benchmark/reporting_html.py` | New `_svg_heatmap` helper + `_render_suite_long_context` (heatmap, prefill curve, first-failure table) wired into the suite dispatcher |
| Model schema | `src/spark_benchmark/models.py` → `ModelConfig` | Add optional `base_model: str | None = None` |
| Telemetry | `src/spark_benchmark/runners/{ollama,llamacpp}.py` | Audit/ensure `prefill_time_s` + a new `context_tokens_loaded` are populated honestly for long contexts |
| CLI/TUI | n/a (additive) | Suite appears automatically in multiselect + NL routing once registered |

### New files

```
src/spark_benchmark/long_context.py                 # suite runner
data/long_context/long_context_retrieval_v1.json    # fixture
data/long_context/haystacks/*.txt                   # PD/Apache-2.0 source text
configs/experiments/spark-long-context.yaml         # experiment config
tests/test_long_context.py                          # tests
```

### Experiment config

```yaml
experiment:
  name: spark-long-context-shootout
  description: Single-needle NIAH at 4k-128k on Spark-local models
  platforms: [spark]
  backend: ollama
  backend_version: local
  models: [qwen-3.6, gemma-4, nemotron-3]
  suites: [long_context_retrieval]
  sampling:
    temperature: 0.0
    top_p: 1.0
    seed: 42
    max_tokens: 256
  context_lengths: [4096, 16384, 65536, 131072]
  repetitions: 1
  warmup_runs: 0
```

(`context_lengths` already exists on `ExperimentSpec`; the suite reads
the grid from the fixture's `test_matrix` and intersects with what the
experiment requests + what each model claims to support.)

---

## Telemetry additions (per task)

On top of the standard power/memory/temp collectors:

- `prefill_time_s` — **audit existing adapters**; this is sneaky work.
  Ollama's API sometimes returns 0 on cache hits, and llama.cpp reports
  it differently. A weekend of per-adapter calibration is budgeted here,
  not assumed free.
- `prefill_tokens_per_sec` — derived: `context_tokens_loaded /
  prefill_time_s`.
- `context_tokens_loaded` — the **actual** tokenized length (verify the
  backend loaded the full context, don't trust the target).
- `peak_memory_mb` during prefill — already collected; this is where
  Spark's 128 GB matters most.

---

## Report output

Beyond the standard `report.md` / `report.html` bundle, the
`_render_suite_long_context` dashboard card shows:

- **NIAH heatmap** per model — inline SVG grid, X = depth %, Y = context
  length, fill = pass rate, with N/A and OOM cells visually distinct.
- **Prefill speed curve** — `_svg_line_chart`, one line per model, X =
  context length, Y = tokens/sec.
- **First-failure table** — per model, the shortest context length where
  pass rate drops below a threshold (default 50 %), plus claimed-vs-real
  support.
- **Markdown fallback** — equivalent tables when HTML is not requested.

---

## Estimated runtime

128 tasks/model. Long contexts dominate wall-clock (131k prefill is
slow). At ~30 s/task average: **~65 min/model**. For 3 models: ~3.5 h —
an overnight-friendly single run.

---

## Test coverage (`tests/test_long_context.py`)

Plain-Python tests, patterned after `tests/test_code_generation.py`.
Target ≥ 12:

- Fixture loader validates schema (needles, haystacks, matrix present).
- Tokenize+truncate hits the target length within tolerance (±1 token
  with a real tokenizer; flagged-approximate path when none available).
- Needle insertion lands at the requested depth % (positional check).
- Substring scorer: case, whitespace, punctuation, no-match.
- Deterministic needle/haystack selection: same inputs → same plan.
- Per-model context limit: `length > context_length` → `skipped_unsupported`,
  not a run.
- OOM path: backend raises → cell recorded `oom`, suite continues.
- `_svg_heatmap`: empty grid, all-pass, all-fail, mixed-with-NA render
  without crashing and contain the expected cell count.
- End-to-end with a fake backend returning canned needle answers →
  expected pass/fail grid.

---

## Acceptance criteria (v0.4.0)

- [ ] Fixture shipped with ≥ 8 needles (≥ 3 categories) and 2
      public-domain/permissive haystacks, each with source + license
      recorded.
- [ ] Suite runs all 4 context lengths without crashing the harness.
- [ ] Per-model context limits respected — no spurious OOMs from
      over-context requests.
- [ ] Per-model tokenization produces honest, reported actual lengths.
- [ ] NIAH heatmap renders as inline SVG for ≥ 1 model; N/A and OOM
      cells visually distinct.
- [ ] `ModelConfig.base_model` field added (optional, back-compatible).
- [ ] `tests/test_long_context.py` passes 100 %.
- [ ] One published run on Spark (Qwen 3.6 or available 70B-class model)
      as the reference artifact.

---

## Out of scope (v0.4.0)

- Multi-needle reasoning + LLM-as-judge (→ v0.5.0+, with BYOT scored
  mode).
- Czech-language haystack (→ later; ship English public-domain first).
- matplotlib / PNG export of any kind (permanent anti-goal).
- Cross-platform / Mac mini comparison (out of project scope entirely).

---

## Open questions deferred to implementation time

- Exact tokenizer access path per backend (Ollama tokenize endpoint vs.
  llama.cpp `tokenize` vs. heuristic fallback) — resolve when wiring the
  adapters; the three-tier fallback above is the contract.
- First-failure threshold default (50 % vs. 80 %) — pick during the
  first real run when we can see the shape of the curve.
