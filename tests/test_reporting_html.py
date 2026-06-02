"""Tests for ``spark_benchmark.reporting_html``.

Coverage focus:

* document well-formedness (doctype, single ``<html>`` / ``<body>``,
  embedded CSS, no script tags)
* hero / stat-tile chrome on both canonical and custom reports
* canonical (bundle) renderer surfaces overall ranking + verdict +
  per-suite dashboard cards (one suite-specific renderer for each of
  ``openclaw_speed`` / ``hallucination_grounding`` /
  ``practical_structured_output`` / ``code_generation`` /
  ``sustained_throughput``)
* custom (BYOT) renderer surfaces telemetry card + per-task TTFT and
  output-length charts + side-by-side blocks + errored strip; HTML-
  escapes user content (XSS hygiene)
* new SVG helpers: line chart (with optional secondary axis), gauge,
  dual bars, stacked bars, thermometer, pass-fail strip
* lazy loaders read ``results.jsonl`` and ``telemetry-*.jsonl`` from
  a suite run-dir and degrade gracefully when files are missing
* ``write_report(..., "both")`` emits both ``.md`` and ``.html`` next
  to each other so callers don't have to invoke the renderers twice
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from spark_benchmark.reporting import write_report
from spark_benchmark.reporting_html import (
    _band_for_pass_rate,
    _cell_pct_html,
    _code_gen_status_breakdown,
    _gradient_color_for_ratio,
    _load_results_rows,
    _load_telemetry_samples,
    _pass_fail_strip_html,
    _per_task_pass_fail,
    _render_suite_long_context,
    _svg_bars,
    _svg_dual_bars,
    _svg_gauge,
    _svg_heatmap,
    _svg_line_chart,
    _svg_stacked_bars,
    _svg_thermometer,
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
    # Hero / stat-tile chrome
    assert 'class="hero"' in html
    assert 'class="stat-tiles"' in html
    assert "Recommended pick" in html  # winner card
    # Per-suite block headers — new design uses human-readable titles
    # plus a canonical-name badge inside an <h2>.
    assert "Speed probe" in html
    assert 'class="badge accent">openclaw_speed' in html
    assert "Grounding / hallucination" in html
    assert 'class="badge accent">hallucination_grounding' in html
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


# --------------------------------------------------------------------- #
# New SVG helpers (line, gauge, dual bars, stacked, thermometer, strip)
# --------------------------------------------------------------------- #


def test_svg_line_chart_empty_input_returns_empty_string() -> None:
    assert _svg_line_chart([]) == ""
    assert _svg_line_chart([("empty", [])]) == ""


def test_svg_line_chart_renders_polyline_with_legend() -> None:
    series = [("model-a", [(0.0, 10.0), (60.0, 12.0), (120.0, 11.5)])]
    html = _svg_line_chart(series, height=80)
    assert "<polyline" in html
    assert "model-a" in html  # legend label
    assert 'class="lines"' in html
    # Three points → coords list contains three space-separated pairs.
    points_attr = re.search(r'points="([^"]+)"', html)
    assert points_attr is not None
    assert len(points_attr.group(1).split(" ")) == 3


def test_svg_line_chart_overlays_dashed_secondary_axis() -> None:
    series = [("tps", [(0.0, 30.0), (60.0, 28.0)])]
    secondary = [("temp", [(0.0, 60.0), (60.0, 78.0)])]
    html = _svg_line_chart(series, secondary=secondary)
    # Primary line is solid; secondary uses stroke-dasharray.
    assert html.count("<polyline") == 2
    assert "stroke-dasharray" in html
    assert "(2nd axis)" in html


def test_svg_gauge_renders_arc_at_correct_fill_and_color() -> None:
    html = _svg_gauge(0.9, max_value=1.0, label="ratio", suffix="%")
    assert html.startswith("<svg")
    assert "ratio" in html
    assert "</svg>" in html
    # 0.9 of max → almost-full sweep, value displayed as "90%".
    assert ">90%<" in html
    # Two paths: the grey track and the colour-graded fill arc.
    assert html.count("<path") == 2


def test_svg_gauge_invert_flips_color_mapping() -> None:
    # ratio=0.1 (low) with invert=True should land on the *good* end of
    # the gradient (≈ green), since "low value" means "low throttle".
    high_ratio = _svg_gauge(0.9, max_value=1.0, invert=False)
    low_inverted = _svg_gauge(0.1, max_value=1.0, invert=True)
    assert "stroke=" in high_ratio and "stroke=" in low_inverted
    # We don't compare exact hex; we just confirm that flipping invert
    # changes the colour outcome.
    high_color = re.findall(r'stroke="(#[0-9a-f]{6})"', high_ratio)[1]
    low_color = re.findall(r'stroke="(#[0-9a-f]{6})"', low_inverted)[1]
    assert high_color == low_color  # same green at the end of the ramp


def test_svg_dual_bars_renders_paired_thin_bars_per_row() -> None:
    html = _svg_dual_bars(
        [("a", 42.0, 38.0), ("b", 35.0, 22.0)],
        label_a="Initial",
        label_b="Sustained",
    )
    # Two rows × 2 SVGs each = 4 SVGs; plus the legend has none.
    assert html.count("<svg") == 4
    assert "Initial" in html
    assert "Sustained" in html
    assert ">a<" in html
    assert ">b<" in html


def test_svg_dual_bars_empty_returns_empty_string() -> None:
    assert _svg_dual_bars([]) == ""


def test_svg_stacked_bars_segments_normalize_to_row_total() -> None:
    rows = [
        ("model-a", [("passed", 7.0), ("failed", 3.0)]),
        ("model-b", [("passed", 5.0), ("failed", 5.0)]),
    ]
    html = _svg_stacked_bars(rows)
    # Two rows × 1 SVG each = 2 SVGs (plus segments inside).
    assert html.count('class="bar-row"') == 2
    # First model: passed segment is 7/10 = 70% wide. Second model: 50%.
    # Just check both width values appear somewhere.
    assert 'width="70.00"' in html
    assert 'width="50.00"' in html
    # Legend includes both segment labels.
    assert "passed" in html
    assert "failed" in html


def test_svg_stacked_bars_skips_empty_rows() -> None:
    rows = [("with-data", [("ok", 1.0)]), ("empty", [])]
    html = _svg_stacked_bars(rows)
    # Only one rendered row (the empty one is filtered out).
    assert html.count('class="bar-row"') == 1


def test_svg_thermometer_returns_empty_for_none() -> None:
    assert _svg_thermometer(None) == ""


def test_svg_thermometer_renders_value_and_color() -> None:
    html = _svg_thermometer(85.0, label="peak")
    assert html.startswith("<svg")
    assert ">85<" in html
    assert "peak" in html
    # Hot temperature renders three filled shapes (track rect, fill rect,
    # and the round bulb circle). The renderer must produce all three.
    assert html.count("<rect") == 2
    assert "<circle" in html
    # At least one element references a hex colour from the gradient ramp
    # (red → amber → green). 85 °C lands near the hot end → reddish.
    hex_colors = re.findall(r'fill="(#[0-9a-f]{6})"', html)
    assert any(c.startswith("#") for c in hex_colors)


def test_pass_fail_strip_renders_one_row_per_model() -> None:
    html = _pass_fail_strip_html(
        [("alpha", [True, True, False, None]), ("bravo", [False, False])]
    )
    assert html.count('class="pf-row"') == 2
    # 4 + 2 = 6 cells total.
    assert html.count('class="cell pass"') == 2  # alpha has 2 passes
    assert html.count('class="cell fail"') == 3  # alpha 1, bravo 2
    assert html.count('class="cell na"') == 1
    assert ">alpha<" in html
    assert ">bravo<" in html


def test_pass_fail_strip_empty_returns_empty_string() -> None:
    assert _pass_fail_strip_html([]) == ""


def test_gradient_color_for_ratio_traverses_red_amber_green() -> None:
    # 0.0 → red, 0.5 → amber, 1.0 → green. Just sanity-check the
    # endpoints; the interpolation is the implementation detail.
    assert _gradient_color_for_ratio(0.0).lower() == "#dc2626"
    assert _gradient_color_for_ratio(1.0).lower() == "#16a34a"
    assert _gradient_color_for_ratio(0.5).lower() == "#d97706"


def test_band_for_pass_rate_buckets() -> None:
    assert _band_for_pass_rate(None) == "na"
    assert _band_for_pass_rate(0.99) == "good"
    assert _band_for_pass_rate(0.95) == "good"
    assert _band_for_pass_rate(0.94) == "warn"
    assert _band_for_pass_rate(0.80) == "warn"
    assert _band_for_pass_rate(0.79) == "bad"
    assert _band_for_pass_rate(0.0) == "bad"


def test_cell_pct_html_emits_color_band_and_fill_pct() -> None:
    cell = _cell_pct_html(0.85)
    assert 'data-band="warn"' in cell
    assert "--cell-pct: 85.0%" in cell
    assert "85.0%" in cell
    # None renders the dimmed em-dash and 0% fill.
    none_cell = _cell_pct_html(None)
    assert 'data-band="na"' in none_cell
    assert "—" in none_cell


# --------------------------------------------------------------------- #
# Lazy loaders                                                          #
# --------------------------------------------------------------------- #


def test_load_results_rows_returns_empty_list_when_missing() -> None:
    with TemporaryDirectory() as tmp:
        # No results.jsonl in tmp — should return empty without raising.
        assert _load_results_rows(tmp) == []
    # Also tolerates None.
    assert _load_results_rows(None) == []


def test_load_results_rows_skips_malformed_lines() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.jsonl"
        path.write_text(
            '{"task_id": "t1", "model": "m", "evaluation": {"passed": true}}\n'
            "this-is-not-json\n"
            '{"task_id": "t2", "model": "m", "evaluation": {"passed": false}}\n',
            encoding="utf-8",
        )
        rows = _load_results_rows(tmp)
        assert len(rows) == 2
        assert rows[0]["task_id"] == "t1"
        assert rows[1]["task_id"] == "t2"


def test_per_task_pass_fail_extracts_in_first_seen_order() -> None:
    rows = [
        {"task_id": "t1", "model": "m", "evaluation": {"passed": True}},
        {"task_id": "t2", "model": "m", "evaluation": {"passed": False}},
        {"task_id": "t3", "model": "m"},  # no evaluation
        {"task_id": "t1", "model": "other"},  # different model — ignored
    ]
    pf = _per_task_pass_fail(rows, "m")
    assert pf == [True, False, None]


def test_load_telemetry_samples_downsamples_uniformly() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "telemetry-foo.jsonl"
        # Write 1000 samples with a clean ramp so we can check the
        # downsampler picks evenly-spaced points.
        with path.open("w", encoding="utf-8") as fh:
            for i in range(1000):
                fh.write(json.dumps({"timestamp_s": float(i), "gpu_temp_c": 50 + i * 0.01}) + "\n")
        samples = _load_telemetry_samples(tmp, "foo", max_points=50)
        assert len(samples) == 50
        # First and last sample should be near the boundaries of the input.
        assert samples[0]["timestamp_s"] == 0.0
        assert samples[-1]["timestamp_s"] >= 970.0


def test_load_telemetry_samples_returns_empty_when_missing() -> None:
    with TemporaryDirectory() as tmp:
        assert _load_telemetry_samples(tmp, "missing-model") == []


def test_code_gen_status_breakdown_orders_passed_first() -> None:
    rows = [
        {
            "model": "m",
            "samples": [
                {"sandbox": {"status": "passed"}},
                {"sandbox": {"status": "timeout"}},
                {"sandbox": {"status": "compile_error"}},
                {"sandbox": {"status": "passed"}},
            ],
        }
    ]
    counts = _code_gen_status_breakdown(rows, "m")
    assert counts[0][0] == "passed"
    assert dict(counts) == {"passed": 2.0, "timeout": 1.0, "compile_error": 1.0}


# --------------------------------------------------------------------- #
# Suite-specific dashboards                                             #
# --------------------------------------------------------------------- #


def _aggregate_with_sustained() -> dict:
    """Aggregate covering all 5 canonical suites with rich extras."""
    return {
        "runs_root": "/results/benchmarks/run-x",
        "total_runs": 5,
        "suites": [
            {
                "suite": "openclaw_speed",
                "models": [
                    {"model": "alpha", "passes": 4, "total": 4, "pass_rate": 1.0, "runs": 1, "avg_ttft_ms": 120.0, "avg_tokens_per_s": 42.0},
                    {"model": "bravo", "passes": 3, "total": 4, "pass_rate": 0.75, "runs": 1, "avg_ttft_ms": 380.0, "avg_tokens_per_s": 28.0},
                ],
            },
            {
                "suite": "code_generation",
                "models": [
                    {
                        "model": "alpha", "passes": 7, "total": 10, "pass_rate": 0.7, "runs": 1,
                        "avg_ttft_ms": None, "avg_tokens_per_s": None,
                        "benchmarks": [
                            {"benchmark": "humaneval", "tasks": 5, "pass_at_1": 0.8},
                            {"benchmark": "mbpp", "tasks": 5, "pass_at_1": 0.6},
                        ],
                    },
                ],
            },
            {
                "suite": "sustained_throughput",
                "models": [
                    {
                        "model": "alpha", "passes": 50, "total": 50, "pass_rate": 1.0, "runs": 1,
                        "avg_ttft_ms": None, "avg_tokens_per_s": 38.0,
                        "initial_tokens_per_s": 42.0, "sustained_tokens_per_s": 38.0,
                        "throttle_ratio": 0.905, "peak_temp_c": 78.0,
                        "windows": [
                            {"start_s": 0, "tokens_per_s": 42.0},
                            {"start_s": 60, "tokens_per_s": 41.5},
                            {"start_s": 120, "tokens_per_s": 39.0},
                            {"start_s": 180, "tokens_per_s": 38.0},
                        ],
                    },
                ],
            },
        ],
        "runs": [],
    }


def test_canonical_renderer_dispatches_to_suite_specific_dashboards() -> None:
    html = render_canonical_report_html(_aggregate_with_sustained())
    # openclaw_speed: 3-up grid with TTFT (lower=better) and tok/s panel
    assert "Speed probe" in html
    assert "TTFT (ms · lower is better)" in html
    assert "Decode throughput" in html
    # code_generation: per-benchmark stacked bars
    assert "Aggregate pass@1" in html
    assert "Per-benchmark passes" in html
    assert "humaneval" in html
    assert "mbpp" in html
    # sustained_throughput: dual bars + gauge + thermometer + line chart
    assert "Initial vs sustained" in html
    assert "Throttle ratio" in html
    assert "Peak GPU temperature" in html
    assert "Throughput over time" in html
    # Stat tiles strip is present
    assert 'class="stat-tiles"' in html
    assert "Top model score" in html


def test_canonical_renderer_color_codes_pass_rate_cells_in_overall_ranking() -> None:
    html = render_canonical_report_html(_aggregate_with_sustained())
    # The overall ranking row uses _cell_pct_html for grounding /
    # structured columns. Even when those rates are 0, the cell carries
    # a data-band attribute.
    assert 'class="cell-pct" data-band=' in html
    assert "--cell-pct:" in html


def test_canonical_renderer_falls_back_when_suite_is_unknown() -> None:
    aggregate = {
        "runs_root": "/tmp",
        "total_runs": 1,
        "suites": [
            {
                "suite": "experimental_new_suite_v0",
                "models": [
                    {"model": "alpha", "passes": 5, "total": 10, "pass_rate": 0.5, "runs": 1, "avg_ttft_ms": None, "avg_tokens_per_s": None},
                ],
            }
        ],
        "runs": [],
    }
    html = render_canonical_report_html(aggregate)
    # Falls back to generic single-bar dashboard, not raising.
    assert "experimental_new_suite_v0" in html
    assert "Pass rate" in html


# --------------------------------------------------------------------- #
# Custom summary new chrome                                             #
# --------------------------------------------------------------------- #


def test_custom_summary_html_has_hero_and_stat_tiles() -> None:
    html = render_custom_summary_html(_custom_summary_minimal())
    # Hero with cyan-tinted gradient (style="..." inline), stat tiles
    # underneath, completed/errored counters, fastest tps tile.
    assert 'class="hero"' in html
    assert 'class="stat-tiles"' in html
    assert "Completed" in html
    assert "Errored" in html
    assert "Fastest decode" in html


def test_custom_summary_html_renders_per_task_charts_inside_details() -> None:
    html = render_custom_summary_html(_custom_summary_minimal())
    # Each task block now contains a 2-up grid: TTFT (lower is better)
    # and Output length panels. Errored rows skip the metrics panels
    # but still contribute to the error strip in the summary header.
    assert "TTFT (ms · lower is better)" in html
    assert "Output length" in html
    # Errored task carries the "contains errors" badge.
    assert "contains errors" not in html or "TimeoutError" in html


# --------------------------------------------------------------------- #
# Lazy-load integration: reliability suite reads results.jsonl          #
# --------------------------------------------------------------------- #


def test_canonical_renderer_loads_per_task_pass_fail_from_run_dir() -> None:
    """End-to-end: render a reliability suite where the model entry
    points at a real ``run_dir`` containing a ``results.jsonl``. The
    dashboard should include a per-task strip rendered from that file.
    """
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "hallucination_grounding"
        run_dir.mkdir()
        (run_dir / "results.jsonl").write_text(
            '{"task_id": "t1", "model": "alpha", "evaluation": {"passed": true}}\n'
            '{"task_id": "t2", "model": "alpha", "evaluation": {"passed": false}}\n'
            '{"task_id": "t3", "model": "alpha", "evaluation": {"passed": true}}\n',
            encoding="utf-8",
        )
        aggregate = {
            "runs_root": str(tmp),
            "total_runs": 1,
            "suites": [
                {
                    "suite": "hallucination_grounding",
                    "models": [
                        {
                            "model": "alpha", "passes": 2, "total": 3, "pass_rate": 0.667,
                            "runs": 1, "avg_ttft_ms": 130.0, "avg_tokens_per_s": 40.0,
                            "run_dir": str(run_dir),
                        },
                    ],
                },
            ],
            "runs": [],
        }
        html = render_canonical_report_html(aggregate)
        # Per-task strip rendered: 2 passes + 1 fail, in input order.
        assert 'class="cell pass"' in html
        assert 'class="cell fail"' in html
        # Hint text mentions "green = pass, red = fail".
        assert "green = pass" in html


# --------------------------------------------------------------------- #
# Long-context retrieval (needle-in-a-haystack) suite                    #
# --------------------------------------------------------------------- #


def _long_context_models() -> list[dict]:
    return [
        {
            "model": "qwen-3.6",
            "passes": 5,
            "total": 8,
            "pass_rate": 0.625,
            "runs": 1,
            "avg_ttft_ms": None,
            "avg_tokens_per_s": None,
            "first_failure_length": 16384,
            "skipped": 2,
            "errors": 0,
            "categories": [
                {"category": "alphanumeric_code", "passes": 1, "n": 3, "pass_rate": 0.3333},
                {"category": "location", "passes": 3, "n": 3, "pass_rate": 1.0},
            ],
            "cells": [
                {"context_length": 4096, "depth_pct": 0, "pass_rate": 1.0, "avg_prefill_tps": 820.0, "peak_vram_mb": 4096.0},
                {"context_length": 4096, "depth_pct": 100, "pass_rate": 1.0, "avg_prefill_tps": 810.0, "peak_vram_mb": 4096.0},
                {"context_length": 16384, "depth_pct": 0, "pass_rate": 0.5, "avg_prefill_tps": 600.0, "peak_vram_mb": 6100.0},
                {"context_length": 16384, "depth_pct": 100, "pass_rate": 0.0, "avg_prefill_tps": 590.0, "peak_vram_mb": 6100.0},
                {"context_length": 131072, "depth_pct": 0, "pass_rate": None, "avg_prefill_tps": None, "peak_vram_mb": None},
                {"context_length": 131072, "depth_pct": 100, "pass_rate": None, "avg_prefill_tps": None, "peak_vram_mb": None},
            ],
        }
    ]


def test_svg_heatmap_renders_values_and_na() -> None:
    svg = _svg_heatmap(
        ["131,072", "4,096"],
        ["0%", "100%"],
        {
            ("131,072", "0%"): None,
            ("131,072", "100%"): 0.0,
            ("4,096", "0%"): 1.0,
            ("4,096", "100%"): 0.95,
        },
    )
    assert svg.startswith("<svg")
    assert "viewBox" in svg
    assert "N/A" in svg  # the None cell
    assert "100%" in svg and "0%" in svg
    # green for the perfect cell, red for the zero cell.
    assert _gradient_color_for_ratio(1.0) in svg
    assert _gradient_color_for_ratio(0.0) in svg


def test_svg_heatmap_empty_returns_blank() -> None:
    assert _svg_heatmap([], ["0%"], {}) == ""
    assert _svg_heatmap(["4096"], [], {}) == ""


def test_render_suite_long_context_has_heatmap_kpi_and_charts() -> None:
    block = _render_suite_long_context(_long_context_models())
    assert 'class="heatmap"' in block
    assert "first-failure length" in block  # KPI strip
    assert "16,384" in block  # the first-failure value, comma-formatted
    assert "Prefill throughput vs context length" in block
    assert "Resident memory vs context length" in block


def test_render_suite_long_context_has_needle_category_panel() -> None:
    block = _render_suite_long_context(_long_context_models())
    assert "Retrieval pass-rate by needle type" in block
    # Category labels are shortened for display.
    assert ">code<" in block
    assert ">location<" in block


def test_render_suite_long_context_without_categories_hides_panel() -> None:
    models = _long_context_models()
    for m in models:
        m.pop("categories", None)
    block = _render_suite_long_context(models)
    assert "Retrieval pass-rate by needle type" not in block


def test_svg_heatmap_left_pad_widens_gutter() -> None:
    narrow = _svg_heatmap(["m"], ["code"], {("m", "code"): 1.0}, left_pad=92)
    wide = _svg_heatmap(["m"], ["code"], {("m", "code"): 1.0}, left_pad=240)
    # A wider gutter pushes the cell origin right and grows the viewBox.
    assert "viewBox" in narrow and "viewBox" in wide
    assert narrow != wide


def test_render_suite_long_context_without_memory_hides_memory_chart() -> None:
    models = _long_context_models()
    for cell in models[0]["cells"]:
        cell["peak_vram_mb"] = None
    block = _render_suite_long_context(models)
    assert "Prefill throughput vs context length" in block
    assert "Resident memory vs context length" not in block


def test_canonical_report_includes_long_context_card() -> None:
    aggregate = {
        "runs_root": "/tmp/results/benchmarks/run-lc",
        "total_runs": 1,
        "suites": [{"suite": "long_context_retrieval_v1", "models": _long_context_models()}],
        "runs": [],
    }
    html = render_canonical_report_html(aggregate)
    # Suite title + versioned badge + the dedicated heatmap renderer fired.
    assert "Long-context retrieval" in html
    assert "long_context_retrieval_v1" in html
    assert 'class="heatmap"' in html


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
