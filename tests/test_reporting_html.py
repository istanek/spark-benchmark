"""Tests for ``spark_benchmark.reporting_html``.

Coverage focus:

* document well-formedness (doctype, single ``<html>`` / ``<body>``,
  embedded CSS, no script tags)
* canonical (bundle) renderer surfaces overall ranking + per-suite
  blocks + verdict when the aggregate has data
* custom (BYOT) renderer surfaces telemetry table + side-by-side
  per-task blocks and HTML-escapes user content (XSS hygiene)
* SVG bar-chart helper handles empty input, picks the correct value
  format, clamps negatives, and emits the right number of bars
* ``write_report(..., "both")`` emits both ``.md`` and ``.html`` next
  to each other so callers don't have to invoke the renderers twice
"""

from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from spark_benchmark.reporting import write_report
from spark_benchmark.reporting_html import (
    _svg_bars,
    render_canonical_report_html,
    render_custom_summary_html,
)


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


def _aggregate_minimal() -> dict:
    """A canonical aggregate with one model on every scoring suite.

    Shape mirrors ``aggregate_runs`` so we can poke the renderer
    directly without spinning up a real run.
    """
    return {
        "runs_root": "/tmp/results/benchmarks/run-x",
        "total_runs": 3,
        "suites": [
            {
                "suite": "openclaw_speed",
                "models": [
                    {
                        "model": "llama3.1-8b",
                        "passes": 4,
                        "total": 4,
                        "pass_rate": 1.0,
                        "runs": 1,
                        "avg_ttft_ms": 120.0,
                        "avg_tokens_per_s": 42.0,
                    }
                ],
            },
            {
                "suite": "hallucination_grounding",
                "models": [
                    {
                        "model": "llama3.1-8b",
                        "passes": 9,
                        "total": 10,
                        "pass_rate": 0.9,
                        "runs": 1,
                        "avg_ttft_ms": 130.0,
                        "avg_tokens_per_s": 41.0,
                    }
                ],
            },
            {
                "suite": "practical_structured_output",
                "models": [
                    {
                        "model": "llama3.1-8b",
                        "passes": 5,
                        "total": 5,
                        "pass_rate": 1.0,
                        "runs": 1,
                        "avg_ttft_ms": 110.0,
                        "avg_tokens_per_s": 45.0,
                    }
                ],
            },
        ],
        "runs": [
            {
                "run_id": "run-001",
                "experiment": "spark-default",
                "backend": "ollama",
                "platform": "spark",
                "suite": "openclaw_speed",
                "suite_version": "1.0.0",
                "row_count": 4,
                "model_count": 1,
                "models": [],
            }
        ],
    }


def _custom_summary_minimal() -> dict:
    """Sample BYOT summary with one task, two models, one error.

    Errors deliberately go through the renderer so the ``error-line``
    class assertion has something to bite on.
    """
    return {
        "suite": "demo-suite",
        "suite_version": "0.0.1",
        "mode": "quick",
        "description": "Side-by-side <demo>",
        "backend": "ollama",
        "task_count": 1,
        "per_model": [
            {
                "model": "llama3.1-8b",
                "model_tag": "llama3.1:8b",
                "tasks_completed": 1,
                "tasks_errored": 0,
                "mean_ttft_ms": 100.0,
                "mean_decode_tps": 35.0,
                "total_decode_tokens": 200,
                "wall_time_s": 6.5,
            },
            {
                "model": "qwen2.5-7b",
                "model_tag": "qwen2.5:7b",
                "tasks_completed": 0,
                "tasks_errored": 1,
                "mean_ttft_ms": None,
                "mean_decode_tps": None,
                "total_decode_tokens": 0,
                "wall_time_s": 0.0,
            },
        ],
        "rows": [
            {
                "task_id": "t1",
                "model": "llama3.1-8b",
                "model_tag": "llama3.1:8b",
                "prompt": "Translate to French: <hello>",
                "generation": {
                    "output": "Bonjour <world>",
                    "finish_reason": "stop",
                    "metrics": {
                        "ttft_ms": 100.0,
                        "decode_time_s": 1.5,
                        "decode_tokens": 200,
                    },
                },
            },
            {
                "task_id": "t1",
                "model": "qwen2.5-7b",
                "model_tag": "qwen2.5:7b",
                "prompt": "Translate to French: <hello>",
                "error": {"type": "TimeoutError", "message": "deadline exceeded"},
            },
        ],
    }


# --------------------------------------------------------------------- #
# Document wrapper / canonical renderer                                 #
# --------------------------------------------------------------------- #


def test_canonical_report_is_a_complete_html_document() -> None:
    html = render_canonical_report_html(_aggregate_minimal())
    assert html.startswith("<!doctype html>")
    assert html.count("<html") == 1
    assert html.count("</html>") == 1
    assert "<style>" in html
    # No script tags should ever leak in — the report is JS-free.
    assert "<script" not in html.lower()
    assert "</body></html>" in html


def test_canonical_report_includes_overall_ranking_and_verdict() -> None:
    html = render_canonical_report_html(
        _aggregate_minimal(),
        request="What is the fastest grounded model?",
        selected_models=["llama3.1-8b"],
        selected_suites=["openclaw_speed", "hallucination_grounding"],
    )
    assert "Overall ranking" in html
    assert "llama3.1-8b" in html
    assert "Verdict" in html
    # Per-suite block headers
    assert "Suite: <code>openclaw_speed</code>" in html
    assert "Suite: <code>hallucination_grounding</code>" in html
    assert "Recent runs" in html


def test_canonical_report_handles_empty_aggregate() -> None:
    empty = {"runs_root": "/tmp/empty", "total_runs": 0, "suites": [], "runs": []}
    html = render_canonical_report_html(empty)
    # Still a valid document, no exceptions, no ranking section.
    assert html.startswith("<!doctype html>")
    assert "Overall ranking" not in html
    assert "Recent runs" not in html


# --------------------------------------------------------------------- #
# Custom (BYOT) renderer                                                #
# --------------------------------------------------------------------- #


def test_custom_summary_html_renders_telemetry_and_per_task() -> None:
    html = render_custom_summary_html(_custom_summary_minimal())
    assert html.startswith("<!doctype html>")
    assert "Custom suite: demo-suite" in html
    # Telemetry table headers + both models
    assert "Per-model telemetry" in html
    assert "llama3.1-8b" in html
    assert "qwen2.5-7b" in html
    # Side-by-side details block per task
    assert "<details>" in html
    assert "Task <code>t1</code>" in html
    # Errored row carries the error-line CSS class.
    assert "error-line" in html
    assert "TimeoutError" in html


def test_custom_summary_html_escapes_user_content() -> None:
    summary = _custom_summary_minimal()
    summary["rows"][0]["prompt"] = "<script>alert('xss')</script>"
    summary["rows"][0]["generation"]["output"] = "<img src=x onerror=alert(1)>"
    html = render_custom_summary_html(summary)
    # Dangerous tag forms must not be present as live HTML — the angle
    # brackets must come out escaped. (We do *not* assert that the words
    # "onerror" / "alert" are absent: html.escape leaves attribute-like
    # text intact, and that's fine because the surrounding tag is
    # already neutralized.)
    assert "<script>alert" not in html
    assert "<img src=x" not in html
    # Escaped form must appear.
    assert "&lt;script&gt;alert" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_custom_summary_html_handles_empty_rows() -> None:
    summary = _custom_summary_minimal()
    summary["per_model"] = []
    summary["rows"] = []
    html = render_custom_summary_html(summary)
    assert "No models recorded" in html
    # Side-by-side section is omitted when there are no rows.
    assert "Side-by-side outputs" not in html


# --------------------------------------------------------------------- #
# SVG bar chart helper                                                  #
# --------------------------------------------------------------------- #


def test_svg_bars_empty_input_returns_empty_string() -> None:
    assert _svg_bars([]) == ""


def test_svg_bars_emits_one_bar_per_row_and_uses_pct_when_capped_at_one() -> None:
    rows = [("alpha", 0.5), ("bravo", 1.0), ("charlie", 0.0)]
    html = _svg_bars(rows, max_value=1.0)
    # One bar-row per input row.
    assert html.count('class="bar-row"') == 3
    assert html.count("<svg") == 3
    # Auto pct formatting kicks in at max=1.0.
    assert "50.0%" in html
    assert "100.0%" in html
    # Labels appear escaped.
    assert ">alpha<" in html
    assert ">bravo<" in html


def test_svg_bars_picks_num_format_when_max_value_is_not_one() -> None:
    html = _svg_bars([("a", 12.5), ("b", 0.0)])
    # Default auto picks num because max != 1.0.
    assert "12.5" in html
    assert "%" not in html  # no pct sign anywhere


def test_svg_bars_clamps_negative_and_oversize_values() -> None:
    # Negative values clamp to 0; values exceeding cap clamp to cap.
    html = _svg_bars([("neg", -10.0), ("big", 999.0)], max_value=10.0)
    # Two bar rows rendered without raising.
    assert html.count('class="bar-row"') == 2
    # The "big" bar should fill 100% of its width.
    assert re.search(r'width="100\.00"', html)


# --------------------------------------------------------------------- #
# write_report integration: format == "both"                            #
# --------------------------------------------------------------------- #


def test_write_report_both_emits_md_and_html_next_to_each_other() -> None:
    aggregate = _aggregate_minimal()
    with TemporaryDirectory() as tmp:
        out_md = Path(tmp) / "report.md"
        write_report(out_md, "both", aggregate)
        assert out_md.exists(), "Markdown copy should be written."
        out_html = out_md.with_suffix(".html")
        assert out_html.exists(), "HTML copy should be written next to .md."
        html_text = out_html.read_text(encoding="utf-8")
        assert html_text.startswith("<!doctype html>")
        assert "llama3.1-8b" in html_text


def _run_all() -> int:
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  ERR  {name}: {exc.__class__.__name__}: {exc}")
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_all())
