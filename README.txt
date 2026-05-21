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
  Install
--------------------------------------------------------------------------------

  git clone https://gitlab.com/istanek/spark-benchmark.git
  cd spark-benchmark
  pip install -e .


--------------------------------------------------------------------------------
  How do I actually run it?
--------------------------------------------------------------------------------

You have four ways to use the tool. Pick whichever fits how you like to
work.


  Way 1 — full screen menu (recommended for first use)
  ----------------------------------------------------

    spark-bench

  This opens a colourful full-screen menu. Use arrow keys to move, Enter
  to choose. The menu shows you which of your locally installed models
  match the benchmark, lets you tick the ones you want to test, lets you
  tick which test suites to run, and then runs everything while you
  watch a scrolling log.

  Anything visual / blue / boxed comes from this mode. It is the easiest
  way to figure out what the tool can do.


  Way 2 — interactive wizard
  --------------------------

    spark-bench wizard --experiment configs/experiments/spark-ollama-baseline.yaml \
                       --platform spark

  Same idea as Way 1 but lighter. Two questions:
    1. Which models do you want to test?     (pick with Space, Enter)
    2. Which test suites do you want to run? (pick with Space, Enter)

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


  Way 4 — describe what you want in a sentence
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
  "nemotron" matches nemotron-3.

  Useful when you don't want to memorise flag names.


--------------------------------------------------------------------------------
  What tests does it run?
--------------------------------------------------------------------------------

Five test suites, all working today:

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


--------------------------------------------------------------------------------
  Where do the results go?
--------------------------------------------------------------------------------

Every run creates one directory under

    results/benchmarks/<timestamp>-<random>/

with one subdirectory per test suite. Inside each you get:

    manifest.json   what was run (models, backend, environment)
    results.jsonl   one row per (model, task) — the raw answers
    summary.json    per-model pass rate, latency, throughput
    summary.md      a readable table

The whole run also gets a top-level report.md with the overall ranking.

Nothing is uploaded anywhere. The results stay on your machine.


--------------------------------------------------------------------------------
  Common problems
--------------------------------------------------------------------------------

  "No configured experiment models were detected in Ollama."
      You haven't pulled any of the models the experiment expects.
      Run "ollama list" to see what you have, then either pull what
      you're missing or pick a different experiment YAML.

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

  README.md             Same content as this file but with markdown
                        formatting and a few extra technical sections.

  METHODOLOGY.md        Why we measure what we measure. The principles
                        we hold ourselves to (publish failures, not
                        just wins).

  docs/architecture.md  How the code is laid out. Module map, data
                        flow, extension recipes for adding your own
                        suites or backends.

  docs/extensions-spec.md
                        The full spec for long-context, sustained
                        throughput, and code generation suites.

  CONTRIBUTING.md       Workflow, conventions, and how to cut a
                        release.

  CHANGELOG.md          What changed in each version.


--------------------------------------------------------------------------------
  Help, support, bugs
--------------------------------------------------------------------------------

Open an issue on GitLab:
    https://gitlab.com/istanek/spark-benchmark/-/issues

Or send a merge request — see CONTRIBUTING.md for the workflow.

License: MIT. See the LICENSE file.
