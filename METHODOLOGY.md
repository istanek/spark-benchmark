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
