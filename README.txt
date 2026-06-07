================================================================================
  spark-benchmark — what is this and how do I use it?
================================================================================

This is a tool for testing local language models on a single machine — an
NVIDIA DGX Spark in particular. If you have several large models installed
locally (Qwen, Gemma, Nemotron, …) and you want to know which one is
fastest, which one tells the truth more often, and which one writes the
best Python code, this tool runs that comparison for you and writes a
report.

It does not call any cloud APIs. Everything runs against models you have
already pulled to your machine through Ollama or llama.cpp.


--------------------------------------------------------------------------------
  Why would I use it?
--------------------------------------------------------------------------------

Three concrete questions:

  1. Which of my local models answers the fastest?
       (time to first token, decoded tokens per second)

  2. Which of my local models makes things up the least?
       (does it answer only from the context it was given, does it know
       when to say "I don't know", does it correct a wrong claim?)

  3. Which of my local models actually writes correct Python code?
       (real HumanEval problems, sandboxed execution, pass / fail)

You point the tool at the experiment configuration, it spins up each
model in turn, runs every test, and gives you a side-by-side table.


--------------------------------------------------------------------------------
  What you need
--------------------------------------------------------------------------------

  - An NVIDIA DGX Spark (any Linux machine works for development, but
    the throughput numbers are only meaningful on Spark in v1).

  - Python 3.11 or newer.

  - Ollama running locally on http://localhost:11434, with at least one
    of the v1 models pulled:

        ollama pull qwen3.6:35b
        ollama pull gemma4:31b
        ollama pull nemotron3:33b

  - Optional: a llama.cpp build with `llama-cli` on PATH, if you want
    to use the llama.cpp backend instead of Ollama.


--------------------------------------------------------------------------------
  Ollama Cloud (optional)
--------------------------------------------------------------------------------

  You can benchmark Ollama Cloud models (the big ones you can't fit on the
  box, e.g. gpt-oss:120b-cloud, deepseek-v3.1:671b-cloud) instead of a local
  Ollama. No config edits needed — just two environment variables:

      export OLLAMA_HOST=https://ollama.com
      export OLLAMA_API_KEY=sk-...        # from https://ollama.com/settings/keys

  Then pick a cloud model by its tag. A few ways:

      # ad-hoc: one prompt, compare against a cloud model
      # (run from the repo dir, or give --experiment an absolute path)
      spark-bench quick "Summarize the CAP theorem." \
                        --experiment configs/experiments/spark-ollama-baseline.yaml \
                        --platform spark \
                        --models gpt-oss:120b-cloud

      # a built-in suite against a specific cloud model
      spark-bench run --experiment configs/experiments/spark-ollama-baseline.yaml \
                      --platform spark --run-suite hallucination_grounding \
                      --model gpt-oss:120b-cloud

  Valid --run-suite values: hallucination_grounding,
  practical_structured_output, code_generation, sustained_throughput,
  long_context_retrieval (and long_context_retrieval_fast).

  The API key is read from the environment only — it is never written to
  configs, manifests, or reports.

  Two caveats for cloud runs:
    - No local GPU telemetry. Memory/temperature/power are unavailable
      remotely, so those charts stay empty; speed (tokens/s, time-to-first-
      token) and pass rates work normally.
    - Cloud calls are billed and go over the network, so long suites (e.g.
      131k long-context) are slower and cost credits.


--------------------------------------------------------------------------------
  Install
--------------------------------------------------------------------------------

  git clone https://github.com/istanek/spark-benchmark.git
  cd spark-benchmark
  pip install -e .


--------------------------------------------------------------------------------
  How do I actually run it?
--------------------------------------------------------------------------------

You have six ways to use the tool. Pick whichever fits how you like to
work.


  Way 1 — full screen menu (recommended for first use)
  ----------------------------------------------------

    spark-bench

  This opens a colourful full-screen menu. Use arrow keys to move, Enter
  to choose. The menu shows EVERY chat-capable model you have pulled in
  Ollama, not just the three the project is shipped with — anything new
  you "ollama pull" appears here automatically, labeled "auto-detected".
  You then tick the ones you want to test, tick which test suites to
  run, and watch a scrolling log while the benchmark runs.

  The menu items are:
    Run      — pick models + canonical suites, run them now.
    Custom   — pick one of your own YAML test suites (BYOT, see Way 4).
    Quick    — type ONE prompt right now, fan it out to every model
               you tick, see the answers side by side. No YAML
               required. (See Way 5.)
    Models   — list every Ollama model and whether it is ready to run.
    Suites   — list the canonical suites and their task counts.
    Info     — show the JSON metadata of one canonical suite.
    Chat     — open a small chat with a single model.
    Refresh  — re-read configs / re-probe Ollama.
    Quit     — leave the TUI (q also works anywhere).

  Anything visual / blue / boxed comes from this mode. It is the easiest
  way to figure out what the tool can do.


  Way 2 — interactive wizard
  --------------------------

    spark-bench wizard --experiment configs/experiments/spark-ollama-baseline.yaml \
                       --platform spark

  Same idea as Way 1 but lighter. Two questions:
    1. Which models do you want to test?     (pick with Space, Enter)
    2. Which test suites do you want to run? (pick with Space, Enter)

  By default the wizard only offers the curated v1 lineup (qwen-3.6,
  gemma-4, nemotron-3). Add --allow-auto-detected to also offer any
  other chat model you have pulled in Ollama:

    spark-bench wizard --experiment configs/experiments/spark-ollama-baseline.yaml \
                       --platform spark --allow-auto-detected

  Then it runs everything and prints a summary at the end.


  Way 3 — chat with a single model
  --------------------------------

    spark-bench console --experiment configs/experiments/spark-ollama-baseline.yaml \
                        --platform spark \
                        --model gemma-4

  Opens a prompt loop. Type a question, press Enter, the model answers.
  Type /exit when you are done. Useful for sanity-checking a model
  before running a long benchmark — if it does not answer here, the
  benchmark won't either.

  --model accepts the curated experiment name (gemma-4), the raw Ollama
  tag (gemma4:31b), or the slugified form (gemma4-31b). Add
  --allow-auto-detected to chat with any chat-capable Ollama tag, even
  one that has no YAML config:

    spark-bench console --experiment configs/experiments/spark-ollama-baseline.yaml \
                        --platform spark --allow-auto-detected \
                        --model phi4:14b


  Way 4 — bring your own test
  ---------------------------

    # 1. Copy the example as a starting point
    cp -r examples/custom-tests/quick my-test
    $EDITOR my-test/suite.yaml

    # 2. Validate before running (catches typos, missing fields,
    #    duplicate task IDs, unknown models). Free, takes < 1 second.
    spark-bench validate-custom my-test/suite.yaml \
                                --experiment configs/experiments/spark-ollama-baseline.yaml \
                                --platform spark

    # 3. Run it
    spark-bench run-custom my-test/suite.yaml \
                           --experiment configs/experiments/spark-ollama-baseline.yaml \
                           --platform spark

  This is the path for "I have my own prompts and I want to see how
  the models I have actually do on them." Drop a YAML file like the
  one in examples/custom-tests/quick/, list your prompts, and the
  harness runs each prompt against each model and writes a
  side-by-side Markdown report you can read on screen.

  Don't want to type the flags? The full-screen menu (Way 1) has a
  "Custom" entry that walks you through the same flow:
    spark-bench
      → arrow to "Custom" → Enter
      → pick a suite from the discovered list (your example
        templates + any suite you've already run once)
      → tick the models you want
      → watch the run scroll by, get the path to summary.md /
        summary.html at the end (open the .html in your browser
        for the prettiest view)

  Two modes are available:

    mode: quick (default)
        Shows the answers and how fast each model produced them.
        No pass/fail verdict — useful for exploratory comparisons.

    mode: scored
        Add a ``scoring:`` block to each task (or one default at
        suite level). Supported scorers:
          exact_match       — model must say exactly this (normalised)
          substring_match   — output must contain all listed substrings
          regex_match       — a Python regex must match somewhere
          json_fields_match — output must be valid JSON with these keys
          multiple_choice   — a letter choice must appear in the output

        Example task with scoring:
          - task_id: cap-france
            prompt: "Capital of France? Reply with just the city name."
            scoring:
              method: exact_match
              expected: Paris

        In scored mode the summary table shows Pass / Total and the
        per-task blocks show ✓ PASS or ✗ FAIL with a reason string.

  Use --dry-run to sanity-check your suite without committing to a
  long run — it executes one task against one model and stops without
  writing any files. See docs/custom-tests-spec.md for the full spec.

  If a long run dies half-way through (Ollama crashes, you Ctrl-C the
  process, the box reboots) just rerun the same command against the
  same output directory — the runner reads the existing results.jsonl
  and skips every (model, task) pair that's already on disk. Pass
  --no-resume to start fresh.

  Useful when the canonical benchmarks don't ask the question you
  actually care about.


  Way 5 — type one quick prompt, see all models reply
  ---------------------------------------------------

    spark-bench quick "Explain what 'throwing peas against a wall' means." \
                      --experiment configs/experiments/spark-ollama-baseline.yaml \
                      --platform spark

  This is "Way 4 but without the YAML." Use this when you have ONE
  thing you want to ask, and you want to see what every model on
  your machine says back. No file to write, no schema to learn —
  just type the prompt in quotes.

  By default it fans the prompt out to every chat-capable model
  Ollama reports (curated lineup + auto-detected). Restrict the
  lineup with --models like everywhere else:

    spark-bench quick "Compare these two paragraphs..." \
                      --experiment configs/experiments/spark-ollama-baseline.yaml \
                      --platform spark \
                      --models qwen-3.6,phi4-14b

  Want to keep the prompt for next time? Add --save and the harness
  writes a real reusable suite YAML to
  examples/custom-tests/quick-saved/<slug>/suite.yaml. That folder
  is git-ignored on purpose (your scratchpad, not shipped
  templates), but the TUI's Custom menu still picks it up because
  it walks the same examples/custom-tests tree. You can rename it
  with --name:

    spark-bench quick "Rate this code review snippet 1-5..." \
                      --experiment configs/experiments/spark-ollama-baseline.yaml \
                      --platform spark \
                      --save --name code-review-rating

  In the full-screen menu (Way 1) the Quick entry walks you through
  the same flow interactively:
    spark-bench
      → arrow to "Quick" → Enter
      → tick the models
      → type the prompt on the line you're given (single line)
      → watch the run scroll by
      → at the end, the harness asks "save this prompt as a
        reusable custom suite? [y/N]" — say yes if you want it back
        in the Custom menu next time.

  Output goes to results/custom/<slug>/<run-id>/, exactly like
  Way 4. summary.md is what you read in the terminal; summary.html
  is what you open in a browser — same content, but with a small
  telemetry chart and collapsible per-task blocks. Both are
  standalone files, no upload anywhere.


  Way 6 — describe what you want in a sentence
  --------------------------------------------

    spark-bench benchmark otestuj qwen a gemma na rychlost a spolehlivost \
                          --experiment configs/experiments/spark-ollama-baseline.yaml \
                          --platform spark

  The tool understands a few Czech and English keywords and routes the
  request to the right test suites. Keywords it recognises:

    speed / rychlost              → speed benchmark (TTFT, tokens/sec)
    reliab / spolehliv            → grounded-vs-hallucination tests
    json / structured / openclaw  → JSON output reliability
    code / kod / kód / humaneval  → Python code generation
    sustained / dlouhodob         → 5-minute decode soak with throttling

  And model aliases: "qwen" matches qwen-3.6, "gemma" matches gemma-4,
  "nemotron" matches nemotron-3. Add --allow-auto-detected to also
  match any other chat model in Ollama by its slugified tag (so
  "phi4:14b" can be referenced as "phi4-14b" in the sentence).

  Useful when you don't want to memorise flag names.


--------------------------------------------------------------------------------
  What tests does it run?
--------------------------------------------------------------------------------

Six test suites, all working today (plus one in preview):

  Speed             Short prompts to measure how quickly each model
                    starts answering and how many tokens per second it
                    produces.

  Hallucination     The model is given a paragraph of context and a
  grounding         question. It is supposed to answer only from that
                    context, say "I don't know" if the answer isn't
                    there, and correct the user if the user asks a
                    leading question with a wrong premise.

  Structured        The model has to return a strict JSON object
  output            matching a specific shape. No prose, no markdown
                    fences. Wrong keys, missing keys, or trailing text
                    all count as a failure.

  Code generation   Real HumanEval-style Python problems. The model
                    writes a function, the tool runs the canonical test
                    cases against it in a sandboxed subprocess, and
                    counts how many problems pass.

  Sustained         The model is asked to generate ~2000 tokens
  throughput        repeatedly for five minutes. The tool watches for
                    thermal throttling and reports tokens/sec at the
                    start versus at the end, peak GPU temperature, and
                    energy per token.

  Long-context      Needle-in-a-haystack retrieval across context
  retrieval         lengths up to 131k tokens. The tool inserts a
                    specific fact at different positions and depths
                    and tests whether the model can retrieve it.
                    Available in a fast preview profile (18 cells /
                    model, ~10 min) and a full profile (128 cells).

  Custom (BYOT)     Your own prompts, your own model lineup. Two modes:
                    quick (side-by-side answers, no scoring) and
                    scored (each task gets a pass/fail verdict via
                    exact_match, substring_match, regex, JSON fields,
                    or multiple_choice scorers). See Way 4 above.


--------------------------------------------------------------------------------
  Where do the results go?
--------------------------------------------------------------------------------

Canonical benchmark runs (Way 1, 2, 6) write to

    results/benchmarks/<timestamp>-<random>/

with one subdirectory per test suite. Inside each you get:

    manifest.json   what was run (models, backend, environment)
    results.jsonl   one row per (model, task) — the raw answers
    summary.json    per-model pass rate, latency, throughput
    summary.md      a readable table

The whole run also gets two top-level reports with the overall ranking:

    report.md       Markdown — good for committing to a wiki / pasting
                    into GitHub
    report.html     standalone HTML — open it in any browser, share by
                    email, attach to a PR. Single file. No JavaScript,
                    no internet connection required. Includes:
                      - a hero banner with the recommended model + why,
                      - stat tiles (models tested, suites run, total
                        tasks, overall pass rate),
                      - the overall ranking with color-coded cells
                        (green / amber / red based on pass rate),
                      - per-suite dashboard cards with charts tailored
                        to each test (bar charts, line charts, gauges,
                        thermometers, per-task pass/fail strips),
                      - a quantization tradeoff table (when a quant
                        sweep has been run) — quality and speed columns
                        colour-coded relative to the reference variant,
                      - the verdict and recommendation.
                    Prints to PDF cleanly (gradients flatten, layout
                    survives).

Custom (Bring-Your-Own-Test) and Quick runs (Way 4 and Way 5) write
to a separate tree so they cannot be confused with the canonical
numbers:

    results/custom/<suite-slug>/<run-id>/
        manifest.json   tagged kind: "custom"
        results.jsonl   one row per (model, task)
        summary.json    per-model telemetry aggregates
        summary.md      side-by-side Markdown (telemetry table on top,
                        then one section per task with each model's
                        reply rendered as a fenced block)
        summary.html    same content as standalone HTML — telemetry
                        table, mean-decode-tps bar chart, and one
                        collapsible block per task with each model's
                        reply side-by-side. Errored cells in red.
                        Tip: this is usually the nicest thing to look
                        at after a quick run.

Either way, nothing is uploaded anywhere. The results stay on your
machine.


--------------------------------------------------------------------------------
  Common problems
--------------------------------------------------------------------------------

  "No configured experiment models were detected in Ollama."
      You haven't pulled any of the models the experiment expects.
      Run "ollama list" to see what you have, then either:
        - pull what you're missing,
        - pick a different experiment YAML, or
        - re-run with --allow-auto-detected so the harness uses
          whatever you DO have.

  "Ollama HTTP 500 ..."
      Ollama is running but choked on the request. Most often: the model
      you're asking for doesn't exist locally, or it's still loading.
      Check the message; the harness includes the exact body Ollama
      returned.

  Curses TUI looks broken in tmux / screen
      Set TERM=xterm-256color before launching. The TUI uses default
      colours, so it works in most terminals, but some multiplexers
      need a hint.


--------------------------------------------------------------------------------
  Where to read more
--------------------------------------------------------------------------------

  docs/README.md        Same content as this file but with markdown
                        formatting and a few extra technical sections
                        (badges, tests overview table, suite mix).

  METHODOLOGY.md        Why we measure what we measure. The principles
                        we hold ourselves to (publish failures, not
                        just wins).

  docs/architecture.md  How the code is laid out. Module map, data
                        flow, extension recipes for adding your own
                        suites or backends.

  docs/extensions-spec.md
                        The full spec for long-context, sustained
                        throughput, and code generation suites.

  docs/custom-tests-spec.md
                        Bring-Your-Own-Test (BYOT) subsystem. v0.2.0
                        ships Mode A; the doc covers the roadmap to
                        v0.3.0+ (deterministic scorers, sandboxed
                        custom Python, local-only judge).

  CONTRIBUTING.md       Workflow, conventions, and how to cut a
                        release.

  CHANGELOG.md          What changed in each version.


--------------------------------------------------------------------------------
  Help, support, bugs
--------------------------------------------------------------------------------

Open an issue on GitHub:
    https://github.com/istanek/spark-benchmark/issues

Or send a pull request — see CONTRIBUTING.md for the workflow.

License: MIT. See the LICENSE file.
