# Benchmark Extensions Spec: Long Context, Sustained Throughput, Code Generation

## Status

Adopted for v1 in Spark-only form. v1 runs only against the DGX Spark configuration,
consistent with the project's v1 scope (`README.txt` / `docs/README.md`, `METHODOLOGY.md`). Cross-platform
framing is explicitly out of scope — see also `docs/custom-tests-spec.md` for the
Spark-only "bring-your-own-test" subsystem.

Implementation order (smallest-to-largest payoff, easiest pipeline validation first):

1. `code_generation` — canonical, externally verifiable benchmark; validates the harness
   against published reference numbers.
2. `sustained_throughput` — pure performance/telemetry, no quality scoring.
3. `long_context_retrieval` — largest scope, most complex scoring, biggest payoff for the
   Spark memory story.

## Suite 1: long_context_retrieval

> **Superseded for implementation.** This section is the original
> aspirational sketch. The buildable, v0.4.0-targeted spec — written
> against the real codebase and with locked-in decisions (Project
> Gutenberg haystacks, Part A only, per-model tokenization, inline-SVG
> heatmaps, 4×4×8 grid) — lives in
> [`docs/long-context-spec.md`](long-context-spec.md). Where the two
> disagree, the newer document wins.

### Purpose

Quantify whether Spark's larger unified memory and bandwidth translate into a useful
long-context advantage in practice — both raw needle retrieval and multi-fact reasoning.

### Methodology

Hybrid NIAH (needle-in-a-haystack) plus multi-needle reasoning.

#### Part A: Single-needle retrieval

Insert a single distinctive fact at varying depths in a large unrelated context. Ask the
model to retrieve it.

- Context lengths (tokens): 4096, 8192, 16384, 32768, 65536, 131072
- Needle positions: 0%, 25%, 50%, 75%, 100% of context depth
- Repetitions: 3 per (length × depth) cell → 90 runs per model
- Haystack content: Paul Graham essays and technical documentation, reported separately
- Needle format: a verifiable fact with no semantic relationship to the haystack
  (e.g. `"The secret access code for the maintenance hatch is 7B-MIRA-4419."`)
- Scoring: exact string match on the code. Binary pass/fail.

#### Part B: Multi-needle reasoning

Insert 3-5 needles at different positions. The question requires combining facts from
multiple needles.

- Context lengths: 8192, 32768, 65536, 131072 (no 4k; multi-needle is meaningless there)
- 5 distinct templates per length, 3 repetitions → 60 runs per model
- Scoring: hybrid LLM-as-judge (Claude Opus / GPT-5 configurable), with deterministic
  substring fallback for clearly correct/incorrect cases.

### Per-model context limits

If a model's claimed maximum context is below a tested length, mark as N/A; do not fail
the suite. The N/A pattern is itself a useful platform/model result and must be reported.

### Telemetry (per run)

- Prefill time (s)
- Prefill tokens/sec (derived)
- TTFT (ms)
- Peak memory during prefill (MB) — this is where Spark's 128 GB matters most
- Decode tokens/sec for the (short) answer
- OOM events — log explicitly, never silently swallow

### Hypothesis (stated up front)

- Spark wins cleanly at 64k+ on 70B-class quantized models due to memory headroom.
- Prefill speed is bandwidth-bound; Spark's memory bandwidth should advantage it, but
  this is the most uncertain dimension and the most interesting to measure.
- Smaller-memory configurations are expected to fail at long context with large models.
  Document the failure point rather than hiding it.

### Output artifacts

- Heatmap: context length × depth, color = pass rate (per model)
- Line chart: prefill tokens/sec vs. context length (per model)
- Memory growth curve: peak memory vs. context length
- Table: first failure length per model

### YAML config sketch

```yaml
suite: long_context_retrieval
config:
  haystack_sources:
    - paul_graham_essays
    - technical_docs_subset
  single_needle:
    context_lengths_tokens: [4096, 8192, 16384, 32768, 65536, 131072]
    depth_percentages: [0, 25, 50, 75, 100]
    repetitions: 3
  multi_needle:
    context_lengths_tokens: [8192, 32768, 65536, 131072]
    templates_per_length: 5
    needles_per_template: [3, 4, 5]
    repetitions: 3
  scoring:
    single_needle: exact_match
    multi_needle: hybrid_judge
    judge_model: "claude-opus-4-7"
  per_model_max_context: auto_detect
```

### Estimated runtime

6 models × (90 single + 60 multi) runs × ~30s avg ≈ 6-8 hours. Plan for overnight runs.

## Suite 2: sustained_throughput

### Purpose

Quantify thermal and power throttling under realistic continuous workloads. A 10-second
throughput burst does not predict 5-10 minute behavior.

### Methodology

Continuous decode for a fixed wall-clock duration, per-minute throughput sampled,
throttling detected.

#### Part A: Single-stream sustained

1. Load model, warmup with one short generation (discard).
2. Issue a prompt that triggers ~2000 tokens of output.
3. On completion, immediately re-issue the same prompt.
4. Repeat for 10 minutes of wall-clock time.
5. Sample tokens/sec per generation and per 60-second window.

#### Part B: Batch sustained

Same as Part A with batch size 4 (where the backend supports it cleanly). If the backend
does not support batching, mark N/A; do not fake it with threading.

#### Part C: Power profiles (v2)

Cross-platform power-profile comparison deferred. For v1 Spark-only: document whether
NVIDIA power management on Spark changes behavior across exposed profiles (research at
implementation time).

### Prompts

Three distinct prompts cycled, to avoid pure cache effects:

1. "Explain how a transformer language model works in detail, starting from tokenization."
2. "Write a Python implementation of a B-tree with insert, delete, and search operations.
   Include docstrings."
3. "Summarize the key arguments in this passage and provide three counterarguments:
   [insert ~500 token excerpt from a known essay]"

Prompts are checked into the repo.

### Telemetry (sampled at 2 Hz)

- Tokens/sec (sliding 30-second window)
- GPU/NPU power (W)
- Total system power (W) — wall-plug via smart plug where possible
- Memory used (MB)
- GPU/NPU temperature (°C)
- Clock speeds (MHz) where exposed
- Fan RPM where exposed
- Throttle flags (NVML throttle reasons on Spark)

### Derived metrics per run

- Initial throughput — first 60 s, tokens/sec
- Sustained throughput — last 60 s, tokens/sec
- Throttle ratio — sustained / initial (1.0 = no throttling)
- Time to throttle — first window with > 10% drop from peak
- Average power (W)
- Energy per token (J/token) = avg_power × duration / total_tokens
- Peak temperature (°C)

### Scoring

No quality score. Pure multi-dimensional performance comparison.

### YAML config sketch

```yaml
suite: sustained_throughput
config:
  duration_minutes: 10
  warmup_generations: 1
  prompts:
    - explain_transformer
    - implement_btree
    - summarize_counterargue
  target_output_tokens: 2000
  batch_sizes: [1, 4]
  telemetry_hz: 2
  cool_down_between_runs_minutes: 5
```

### Output artifacts

- Line chart: tokens/sec over 10 minutes (per model, batch size 1)
- Line chart: temperature over 10 minutes
- Line chart: power (W) over 10 minutes
- Bar chart: throttle ratio across models
- Bar chart: energy per token (J/token)
- Table: thermal/throttle event log per run

### Honesty notes

- Document ambient temperature during runs.
- Document chassis position (rack vs. desk).
- Each test min 3 repetitions with 5-minute cooldown.

### Estimated runtime

6 models × 10 min × 2 batch sizes × 3 repetitions × 5 min cooldown ≈ 7.5 hours.

## Suite 3: code_generation

### Purpose

Provide externally-verifiable quality numbers on a canonical, well-known benchmark suite.
Published HumanEval and MBPP scores are widely known for major models; if our numbers
do not match published baselines within ±2-3%, our methodology is broken and we need to
know.

This suite is primarily a sanity check on the entire pipeline, disguised as a useful
benchmark.

### Methodology

Two well-established benchmarks: HumanEval (164 Python problems) and MBPP sanitized
subset (257 problems).

#### HumanEval

- 164 problems, official OpenAI release
- Metrics: pass@1 (T=0, single sample) and pass@10 (T=0.8, 10 samples)
- Scoring implements the canonical unbiased estimator from the HumanEval paper

#### MBPP (sanitized)

- 257 problems
- Metric: pass@1 only (pass@10 adds little signal)

#### HumanEval+ / MBPP+ (recommended)

- EvalPlus extended test cases (~80x more tests than the originals)
- Catches false positives where models pass weak tests but fail edge cases
- Modern reference standard since 2024

### Execution safety

Generated code must be executed in a sandbox. Defaults from least-to-most isolated:

1. **subprocess + resource.setrlimit + timeout** (minimum, v1 default — portable across
   environments without Docker)
2. **firejail** (lighter than Docker, available on Linux)
3. **Docker container** (production-recommended): no network, read-only filesystem
   except `/tmp`, CPU/memory limits, 10-second per-test timeout

Generated code must never be run on the host Python directly. Non-negotiable.

### Sampling

- pass@1: temperature 0, top_p 1.0, seed 42, single sample
- pass@10: temperature 0.8, top_p 0.95, 10 samples per problem, seed sequence [42..51]

### Prompt format

Canonical prompt format for each benchmark, not a custom one. For HumanEval: bare
function signature plus docstring. No system prompts, no "write good code" instructions —
published numbers don't use them, and we want comparable results.

Document the exact prompt format in `METHODOLOGY.md`.

### Scoring

Deterministic. Run generated code against test cases, count passes.

```
pass@k = E[ 1 - C(n-c, k) / C(n, k) ]
```

where `n` is the number of samples generated and `c` is the number of correct samples.
This is the unbiased estimator from the original HumanEval paper, not naive "any of k
passed."

### Reference-score validation

After a run completes, compare each model's pass@1 against a stored reference score
(`data/code/reference_scores.yaml`). If delta exceeds tolerance (default 3 pp), emit a
warning:

```
Llama 3.3 70B HumanEval pass@1: 71.3% (expected 78.2% ±3pp)
Possible causes: quantization quality, prompt format, sampling config
```

Catches: prompt format bugs, sampling config mistakes, quantization-induced quality
regression, backend bugs (e.g. broken stop tokens).

### YAML config sketch

```yaml
suite: code_generation
config:
  benchmarks:
    - humaneval
    - humaneval_plus
    - mbpp_sanitized
    - mbpp_plus
  metrics:
    humaneval: [pass_at_1, pass_at_10]
    humaneval_plus: [pass_at_1]
    mbpp_sanitized: [pass_at_1]
    mbpp_plus: [pass_at_1]
  sampling:
    pass_at_1:
      temperature: 0.0
      top_p: 1.0
      seed: 42
      samples: 1
    pass_at_10:
      temperature: 0.8
      top_p: 0.95
      seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
      samples: 10
  execution:
    sandbox: subprocess_rlimit
    timeout_seconds: 10
    memory_limit_mb: 512
  reference_check:
    enabled: true
    tolerance_pp: 3.0
```

### Output artifacts

- Table: pass@1 / pass@10 per model per benchmark
- Bar chart: pass@1 deltas vs. reference (validates pipeline integrity)
- Table: HumanEval+ vs. HumanEval pass@1 gap (measures overfitting to weak tests)
- Per-problem failure analysis (optional, useful for debugging)

### Honesty notes

- Quantized models score lower than FP16. Report the quant level explicitly with every
  score.
- Same model on different backends may produce slightly different scores due to sampling
  implementation differences. Document this.
- HumanEval is saturated for top models. HumanEval+ is more discriminative in 2026.
- Code-specific fine-tunes (Qwen-Coder, DeepSeek-Coder) should not be grouped in
  headline comparisons with general-purpose models; group them separately or note the
  asymmetry.

### Estimated runtime

- pass@1 only: (164 + 257) × 6 models × ~3 s ≈ 2 hours
- + pass@10 on HumanEval: + 164 × 10 × 6 models × ~3 s ≈ 3 hours
- Total: ~5 hours

## Integration into the existing harness

### File structure additions

```
docs/
└── extensions-spec.md             # this document

data/
├── code/                          # NEW
│   ├── code_generation_v1.json    # HumanEval/MBPP problems in SuiteDefinition format
│   └── reference_scores.yaml      # published pass@1 references per model
├── needles/                       # NEW (suite 1)
└── haystacks/                     # NEW (suite 1)

src/spark_benchmark/
├── code_generation.py             # NEW: sandbox, scoring, pass@k, runner
├── long_context.py                # NEW (suite 1)
└── sustained_throughput.py        # NEW (suite 2)

configs/experiments/
├── spark-code-generation.yaml     # NEW
├── spark-long-context.yaml        # NEW
└── spark-sustained.yaml           # NEW
```

### CLI additions

No new top-level commands. Each suite plugs into the existing `--run-suite` and
`benchmark` paths.

### Recommended experiment configs (Spark-only adaptation)

```yaml
# configs/experiments/spark-code-generation.yaml
experiment:
  name: spark-code-generation-v1
  platforms: [spark]
  models: [qwen-3.6, gemma-4, nemotron-3]
  suites: [code_generation]
  repetitions: 1

# configs/experiments/spark-long-context.yaml
experiment:
  name: spark-long-context-v1
  platforms: [spark]
  models: [qwen-3.6, gemma-4, nemotron-3]
  suites: [long_context_retrieval]
  repetitions: 3

# configs/experiments/spark-sustained.yaml
experiment:
  name: spark-sustained-v1
  platforms: [spark]
  models: [qwen-3.6, gemma-4, nemotron-3]
  suites: [sustained_throughput]
  repetitions: 3
```

## Acceptance criteria

- **Long context**: heatmaps render for all (model × length × depth) cells, with N/A
  clearly marked where unsupported; prefill speed curves plotted; OOM events logged with
  exact tokens loaded.
- **Sustained**: 10-minute curves for tokens/sec, power, temperature plotted per
  (model); throttle ratio computed; NVML throttle reasons captured.
- **Code**: pass@1 scores within ±3 pp of stored reference values, or warning emitted;
  sandbox isolation verified (deliberately submit malicious code, confirm contained).
- All three suites runnable via the existing CLI without modification to the core harness
  abstractions.

## Out of scope (these extensions, v1)

- Multimodal long-context (vision tokens) — separate suite.
- Code generation in languages other than Python — Python is the canonical reference.
- Long-context generation (32k-token outputs) — different problem, not addressed here.
- Inference-time adaptation (speculative decoding, draft models).
- Cross-platform comparisons of any kind — v1 is Spark-only by design.
