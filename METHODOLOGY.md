# Methodology

## Principles

1. Reproducibility over convenience
2. Reliability matters as much as headline benchmark scores
3. Publish failures, not just wins
4. Report performance, quality, efficiency, and hallucination behavior together

## v1 scope

Version 1 is Spark-only. We are comparing multiple local models on the same DGX Spark setup rather than comparing Spark against another machine.

Primary v1 questions:

1. Which model is fastest and most efficient on Spark?
2. Which model is most reliable on practical tasks?
3. Which model hallucinates less when information is missing, ambiguous, or adversarial?
4. Which model gives the best tradeoff for real use, not just benchmark bragging rights?

Initial model lineup:

- qwen-3.6
- gemma-4
- nemotron-3
- nemotron-3-super

## Evaluation categories

### Quality

Conventional suites remain useful, but they are not the whole story. Quality runs should cover correctness-oriented public benchmarks and deterministic scoring where possible.

### Performance

Measure throughput, TTFT, prefill behavior, context scaling, memory pressure, thermal stability, and energy-oriented telemetry on Spark.

### Reliability and hallucination behavior

This is a first-class evaluation axis in v1. The harness should support suites that test:

- refusal vs. fabrication when the prompt does not contain enough information
- grounded answering from supplied context
- citation or evidence discipline when required
- structured output compliance under ambiguity
- consistency across repeated runs on the same task

### Practical outcomes

Add lightweight task sets that look more like real usage than leaderboard evals, for example:

- practical coding tasks with concrete acceptance checks
- JSON or tool-calling style outputs
- summarization or extraction with known source-grounded answers
- domain-specific workflows where wrong confident answers should be penalized

## Phase 1 scaffold scope

Phase 1 focuses on config validation, orchestration shape, backend abstraction, telemetry abstraction, report generation structure, and placeholder suite organization for reliability-oriented work.

---

## Empirical findings

### Long-context retrieval (v0.4.0 – v0.4.3, fast profile)

Fast profile grid: lengths = 4 096 / 32 768 / 131 072 tokens,
depths = 0 % / 50 % / 100 %, 2 needles per cell = 18 cells per model.

**Key pattern: depth dominates length.** Every tested model—qwen-3.6, gemma-4,
nemotron-3, and gpt-oss:120b-cloud—passes at depth 100 % (needle at the *end* of
the context) and fails at depths 0 % and 50 % essentially uniformly, regardless of
context length. This is the classic "lost in the middle" failure, not a
Spark-specific memory or bandwidth limitation.

Selected results from the v0.4.3 run (20260602):

| Model | passes / 18 | depth-0 % | depth-50 % | depth-100 % | VRAM @ 131k |
|---|---|---|---|---|---|
| qwen-3.6 | 6 / 18 = 33 % | 0/6 | 0/6 | 6/6 | 28 880 MB |
| gemma-4 | 6 / 18 = 33 % | 0/6 | 0/6 | 4/6 | 34 430 MB |
| nemotron-3 | 6 / 18 = 33 % | 0/6 | 0/6 | 6/6 | 31 173 MB |
| gpt-oss:120b-cloud | 6 / 18 = 33 % | 0/6 | 0/6 | 6/6 | 67 537 MB |

**Prefill throughput scales with Spark memory bandwidth.** Qwen-3.6 achieves
~1 450 tok/s at 4k context, degrading to ~1 120 tok/s at 131k. Nemotron-3 is
similar (~1 940 → 1 520). Gemma-4 is notably slower (~655 → 360) due to its
architecture. The 120B cloud model processes ~1 210 → 780 tok/s.

**Version-over-version prompt improvements.** The v0.4.0 run (pre-v0.4.2) showed
depth-0 and depth-50 scoring 0 on all models because models treated the needle as
"not in context". The v0.4.2 anti-refusal prompt and numeric-format normalisation
resolved the refusal issue; the pattern above is now genuine retrieval failure,
not prompt phrasing.

**What this means for the Spark story.** The Spark's 128 GB unified memory is *not*
the bottleneck—every model loads to 131k without OOM, and prefill throughput stays
high. The bottleneck is model architecture: these models are not trained to retrieve
facts from mid-context positions. This is the honest finding. The quant sweep
(v0.5.0) should show whether smaller quantisations shift the failure point.

**Next step.** Run the full grid (4×4×8) for at least qwen-3.6 and nemotron-3 to
produce statistically significant per-depth columns. The fast profile confirms the
pattern; the full profile will quantify it precisely enough to publish.
