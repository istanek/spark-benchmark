"""Standalone HTML reports — single-file, dependency-free.

This module renders both flavours of run output into a polished,
self-contained HTML page that can be opened in a browser, attached to
an email, or pasted into a wiki:

- :func:`render_canonical_report_html` — bundle-level rollup for
  canonical runs (``results/benchmarks/<bundle-id>/report.html``).
  Includes overall ranking, per-suite tables, narrative commentary,
  verdict, and inline SVG bar charts of pass-rates.
- :func:`render_custom_summary_html` — single-suite rollup for
  Bring-Your-Own-Test runs (``results/custom/<slug>/<run-id>/summary.html``).
  Includes per-model telemetry, optional per-model error highlights,
  and one ``<details>`` block per task with each model's reply
  side-by-side.

Design constraints (deliberate):

- **No JavaScript**. Pages must work with JS disabled. Collapsibles
  use the native ``<details>``/``<summary>`` element.
- **No CDN, no external assets**. Everything (CSS, charts) is inline
  so the file is portable. Open ``report.html`` from a USB stick on
  a plane — it works.
- **No template engine.** Plain Python f-strings and ``html.escape``.
  Pulling in jinja2 just for static reports is overkill.

The styling is intentionally low-key: a system-font stack, dark text on
light background, table-row hover, sparing accent colour. We are not
building a marketing site; the goal is to make the *numbers* legible.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any, Iterable

from spark_benchmark.reporting import (
    _find_suite,  # noqa: F401  - re-used by the canonical renderer below
    _overall_rank_rows,
    _suite_commentary,
    _verdict_paragraph,
)


# --------------------------------------------------------------------- #
# Shared CSS                                                            #
# --------------------------------------------------------------------- #


_CSS = """
:root {
  --fg: #0f172a;
  --fg-muted: #475569;
  --fg-faint: #94a3b8;
  --bg: #f8fafc;
  --bg-card: #ffffff;
  --bg-row-alt: #f1f5f9;
  --border: #e2e8f0;
  --accent: #0ea5e9;
  --accent-strong: #0369a1;
  --good: #16a34a;
  --warn: #d97706;
  --bad: #dc2626;
  --code-bg: #0f172a;
  --code-fg: #e2e8f0;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 24px 64px;
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.55;
}
.container { max-width: 1100px; margin: 0 auto; }
header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 20px;
  margin-bottom: 32px;
}
h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: -0.01em; }
h2 { margin: 36px 0 12px; font-size: 22px; }
h3 { margin: 24px 0 8px; font-size: 18px; }
p, li { color: var(--fg); }
small, .meta { color: var(--fg-muted); font-size: 13px; }
.meta-row {
  display: flex; flex-wrap: wrap; gap: 16px 24px;
  color: var(--fg-muted); font-size: 13px;
  margin-top: 8px;
}
.meta-row strong { color: var(--fg); font-weight: 600; }
.kvs { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; }
.kvs dt { font-weight: 600; color: var(--fg-muted); }
.kvs dd { margin: 0; }
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 20px;
  margin: 16px 0 28px;
}
.card.ok    { border-left: 4px solid var(--good); }
.card.warn  { border-left: 4px solid var(--warn); }
.card.bad   { border-left: 4px solid var(--bad); }
.card h2:first-child, .card h3:first-child { margin-top: 0; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}
th, td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
th { background: var(--bg-row-alt); font-weight: 600; }
tr:nth-child(even) td { background: var(--bg-row-alt); }
tr:hover td { background: #e0f2fe; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  background: var(--bg-row-alt);
  color: var(--fg-muted);
  margin-left: 6px;
}
.badge.ok   { background: #dcfce7; color: var(--good); }
.badge.warn { background: #fef3c7; color: var(--warn); }
.badge.bad  { background: #fee2e2; color: var(--bad); }
.commentary {
  font-style: italic;
  color: var(--fg-muted);
  margin: 8px 0 0;
}
pre, code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo,
               Consolas, "Liberation Mono", monospace;
  font-size: 13px;
}
pre {
  background: var(--code-bg);
  color: var(--code-fg);
  padding: 12px 14px;
  border-radius: 8px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 8px 0 16px;
}
details {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-card);
  padding: 6px 14px;
  margin: 10px 0;
}
details[open] { padding-bottom: 14px; }
details > summary {
  cursor: pointer;
  font-weight: 600;
  padding: 6px 0;
  outline: none;
}
details > summary::-webkit-details-marker { color: var(--fg-faint); }
.task-block { padding-top: 4px; }
.task-block .prompt { margin: 4px 0 12px; }
.model-reply { margin: 6px 0 14px; }
.model-reply h4 {
  margin: 12px 0 4px;
  font-size: 15px;
  display: flex; align-items: baseline; gap: 8px;
}
.model-reply h4 .telemetry {
  color: var(--fg-muted);
  font-size: 12px;
  font-weight: 400;
}
.error-line { color: var(--bad); font-weight: 600; }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.bar-row .label {
  flex: 0 0 180px; font-size: 13px; color: var(--fg);
  text-overflow: ellipsis; overflow: hidden; white-space: nowrap;
}
.bar-row .value {
  flex: 0 0 80px; text-align: right;
  font-variant-numeric: tabular-nums; font-size: 13px;
  color: var(--fg-muted);
}
.bar-row svg { flex: 1 1 auto; height: 14px; display: block; }
footer {
  margin-top: 48px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  color: var(--fg-faint); font-size: 12px;
}
"""


# --------------------------------------------------------------------- #
# Tiny helpers                                                          #
# --------------------------------------------------------------------- #


def _esc(value: Any) -> str:
    """``html.escape`` with str coercion, so callers don't have to."""
    if value is None:
        return "—"
    return html.escape(str(value), quote=True)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: float | int | None, *, places: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{places}f}"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# --------------------------------------------------------------------- #
# Inline SVG bar chart                                                  #
# --------------------------------------------------------------------- #


def _svg_bars(
    rows: list[tuple[str, float]],
    *,
    max_value: float | None = None,
    accent: str = "#0ea5e9",
    value_format: str = "auto",
) -> str:
    """Render a list of ``(label, value)`` pairs as inline-SVG bar rows.

    Outputs one ``<div class="bar-row">`` per pair containing the label,
    a horizontal SVG bar (filled proportionally to ``value / max_value``),
    and the numeric value on the right. Pure CSS layout — no JS, no
    flexbox tricks beyond what every browser shipped a decade ago.

    Empty input → empty string. ``max_value`` defaults to ``max(values)``,
    falling back to 1 to avoid divide-by-zero. ``value_format`` controls
    how the right-hand value is rendered: ``"pct"`` (e.g. ``58.2%``),
    ``"num"`` (e.g. ``42.7``), or ``"auto"`` — auto picks pct when
    ``max_value == 1.0``, otherwise num.
    """
    if not rows:
        return ""
    values = [max(0.0, float(v)) for _, v in rows]
    cap = max_value if (max_value is not None and max_value > 0) else max(values + [0.0])
    if cap <= 0:
        cap = 1.0
    if value_format == "auto":
        value_format = "pct" if max_value == 1.0 else "num"
    parts: list[str] = []
    for label, value in rows:
        ratio = max(0.0, min(1.0, float(value) / cap))
        # Use viewBox so the bar resizes with the flex container.
        svg = (
            f'<svg viewBox="0 0 100 1" preserveAspectRatio="none" aria-hidden="true">'
            f'<rect x="0" y="0" width="100" height="1" fill="#e2e8f0" />'
            f'<rect x="0" y="0" width="{ratio * 100:.2f}" height="1" fill="{accent}" />'
            f"</svg>"
        )
        formatted = _fmt_pct(value) if value_format == "pct" else _fmt_num(value)
        parts.append(
            f'<div class="bar-row">'
            f'<span class="label" title="{_esc(label)}">{_esc(label)}</span>'
            f"{svg}"
            f'<span class="value">{formatted}</span>'
            f"</div>"
        )
    return "".join(parts)


# --------------------------------------------------------------------- #
# Canonical (bundle-level) report                                       #
# --------------------------------------------------------------------- #


def render_canonical_report_html(
    aggregate: dict[str, Any],
    *,
    request: str | None = None,
    selected_models: Iterable[str] | None = None,
    selected_suites: Iterable[str] | None = None,
    title: str = "spark-benchmark report",
) -> str:
    """Render a polished HTML page from an ``aggregate_runs`` dict.

    ``request`` / ``selected_models`` / ``selected_suites`` come from
    the orchestrator (or the NL-routed ``benchmark`` command) and feed
    the narrative section. They are optional — when omitted, the
    renderer falls back to whatever the aggregate itself reveals.
    """
    suites = aggregate.get("suites") or []
    runs = aggregate.get("runs") or []
    selected_models_list = list(selected_models) if selected_models else _model_names_from_suites(suites)
    selected_suites_list = list(selected_suites) if selected_suites else [s["suite"] for s in suites]
    ranking = _overall_rank_rows(aggregate, selected_models_list)

    body: list[str] = []
    body.append('<div class="container">')

    # Header / meta
    body.append("<header>")
    body.append(f"<h1>{_esc(title)}</h1>")
    body.append('<div class="meta-row">')
    if request:
        body.append(f"<div><strong>Request:</strong> {_esc(request)}</div>")
    body.append(
        f"<div><strong>Runs root:</strong> "
        f"<code>{_esc(aggregate.get('runs_root', '—'))}</code></div>"
    )
    body.append(f"<div><strong>Total runs:</strong> {len(runs)}</div>")
    if selected_models_list:
        body.append(
            f"<div><strong>Models:</strong> {_esc(', '.join(selected_models_list))}</div>"
        )
    body.append(f"<div><strong>Generated:</strong> {_esc(_now_utc_iso())}</div>")
    body.append("</div>")
    body.append("</header>")

    # Verdict + recommendation
    if ranking:
        # ``_verdict_paragraph`` keys by canonical suite name (not the
        # potentially-versioned ``suite['suite']`` field), mirroring
        # ``render_cli_benchmark_summary`` so the prose stays in sync.
        suite_map = {
            "openclaw_speed": _find_suite(aggregate, "openclaw_speed"),
            "hallucination_grounding": _find_suite(aggregate, "hallucination_grounding"),
            "practical_structured_output": _find_suite(aggregate, "practical_structured_output"),
            "code_generation": _find_suite(aggregate, "code_generation"),
            "sustained_throughput": _find_suite(aggregate, "sustained_throughput"),
        }
        verdict = _verdict_paragraph(ranking, suite_map)
        winner = ranking[0]
        body.append('<section class="card ok">')
        body.append("<h2>Verdict</h2>")
        if verdict:
            body.append(f"<p>{_esc(verdict)}</p>")
        body.append(
            f"<p><strong>Recommendation:</strong> "
            f"<code>{_esc(winner['model'])}</code> is the current default pick — "
            f"strongest combined result across reliability and speed.</p>"
        )
        body.append("</section>")

    # Overall ranking table + bar chart
    if ranking:
        body.append("<h2>Overall ranking</h2>")
        body.append('<table aria-label="Overall ranking">')
        body.append(
            "<thead><tr>"
            "<th>#</th><th>Model</th>"
            "<th class='num'>Overall score</th>"
            "<th class='num'>Grounding</th>"
            "<th class='num'>Structured JSON</th>"
            "<th class='num'>Avg TTFT</th>"
            "<th class='num'>Avg tok/s</th>"
            "</tr></thead><tbody>"
        )
        for i, row in enumerate(ranking, start=1):
            body.append(
                "<tr>"
                f"<td class='num'>{i}</td>"
                f"<td><code>{_esc(row['model'])}</code></td>"
                f"<td class='num'>{_fmt_num(row['overall_score'], places=3)}</td>"
                f"<td class='num'>{_fmt_pct(row['grounding_rate'])}</td>"
                f"<td class='num'>{_fmt_pct(row['structured_rate'])}</td>"
                f"<td class='num'>"
                f"{('—' if row['avg_ttft_ms'] is None else f'{_fmt_num(row['avg_ttft_ms'])} ms')}"
                f"</td>"
                f"<td class='num'>"
                f"{('—' if row['avg_tokens_per_s'] is None else f'{_fmt_num(row['avg_tokens_per_s'])} tok/s')}"
                f"</td>"
                "</tr>"
            )
        body.append("</tbody></table>")

        bar_rows = [(r["model"], float(r["overall_score"])) for r in ranking]
        body.append('<div class="card">')
        body.append("<h3>Overall score</h3>")
        body.append(_svg_bars(bar_rows, max_value=1.0))
        body.append("</div>")

    # Per-suite breakdown
    for suite in suites:
        body.append(_render_suite_block_html(suite))

    # Recent runs tail
    if runs:
        body.append("<h2>Recent runs</h2>")
        body.append(
            "<table><thead><tr>"
            "<th>Run ID</th><th>Experiment</th><th>Backend</th>"
            "<th>Suite</th><th class='num'>Rows</th>"
            "</tr></thead><tbody>"
        )
        for run in reversed(runs):
            body.append(
                "<tr>"
                f"<td><code>{_esc(run.get('run_id'))}</code></td>"
                f"<td>{_esc(run.get('experiment'))}</td>"
                f"<td>{_esc(run.get('backend'))}</td>"
                f"<td>{_esc(run.get('suite') or '—')}</td>"
                f"<td class='num'>{int(run.get('row_count') or 0)}</td>"
                "</tr>"
            )
        body.append("</tbody></table>")

    if selected_suites_list:
        body.append(
            f"<p class='meta'>Suites in this report: "
            f"{', '.join(f'<code>{_esc(s)}</code>' for s in selected_suites_list)}.</p>"
        )

    body.append("<footer>")
    body.append(
        "Generated by <code>spark-benchmark</code> · "
        "no scripts, no CDN — open the file anywhere."
    )
    body.append("</footer>")
    body.append("</div>")

    return _wrap_document(title=title, body_html="".join(body))


def _render_suite_block_html(suite: dict[str, Any]) -> str:
    name = str(suite.get("suite") or "")
    models = suite.get("models") or []
    parts: list[str] = []
    parts.append(f"<h2>Suite: <code>{_esc(name)}</code></h2>")
    commentary = _suite_commentary(name, suite) if models else ""
    if commentary:
        parts.append(f'<p class="commentary">{_esc(commentary)}</p>')
    if not models:
        parts.append('<p class="meta">No models scored in this suite.</p>')
        return "".join(parts)
    parts.append('<table aria-label="Per-model results">')
    parts.append(
        "<thead><tr>"
        "<th>Model</th>"
        "<th class='num'>Passes</th>"
        "<th class='num'>Total</th>"
        "<th class='num'>Pass rate</th>"
        "<th class='num'>Runs</th>"
        "<th class='num'>Avg TTFT (ms)</th>"
        "<th class='num'>Avg tok/s</th>"
        "</tr></thead><tbody>"
    )
    for model in models:
        ttft = model.get("avg_ttft_ms")
        tok_s = model.get("avg_tokens_per_s")
        parts.append(
            "<tr>"
            f"<td><code>{_esc(model.get('model'))}</code></td>"
            f"<td class='num'>{int(model.get('passes') or 0)}</td>"
            f"<td class='num'>{int(model.get('total') or 0)}</td>"
            f"<td class='num'>{_fmt_pct(model.get('pass_rate'))}</td>"
            f"<td class='num'>{int(model.get('runs') or 0)}</td>"
            f"<td class='num'>{_fmt_num(ttft) if ttft is not None else '—'}</td>"
            f"<td class='num'>{_fmt_num(tok_s) if tok_s is not None else '—'}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")

    bar_rows = [(str(m.get("model", "?")), float(m.get("pass_rate") or 0.0)) for m in models]
    parts.append('<div class="card">')
    parts.append("<h3>Pass rate</h3>")
    parts.append(_svg_bars(bar_rows, max_value=1.0))
    parts.append("</div>")
    return "".join(parts)


def _model_names_from_suites(suites: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for suite in suites:
        for model in suite.get("models") or []:
            name = str(model.get("model") or "")
            if name and name not in seen_set:
                seen.append(name)
                seen_set.add(name)
    return seen


# --------------------------------------------------------------------- #
# Custom (BYOT) summary                                                 #
# --------------------------------------------------------------------- #


def render_custom_summary_html(summary: dict[str, Any]) -> str:
    """Render a custom suite ``summary.json`` payload as HTML.

    Counterpart to ``custom_suites.render_custom_summary_markdown``.
    Layout: header card (suite name, version, mode, backend, task
    count) → per-model telemetry table + bar chart of mean tok/s →
    one collapsible ``<details>`` per task with each model's reply
    side-by-side, errors highlighted in red.
    """
    suite_name = str(summary.get("suite") or "")
    title = f"Custom suite: {suite_name}"
    per_model = list(summary.get("per_model") or [])
    rows = list(summary.get("rows") or [])

    body: list[str] = []
    body.append('<div class="container">')

    # Header
    body.append("<header>")
    body.append(f"<h1>{_esc(title)}</h1>")
    body.append('<div class="meta-row">')
    body.append(
        f"<div><strong>Version:</strong> {_esc(summary.get('suite_version', '—'))}</div>"
    )
    body.append(
        f"<div><strong>Mode:</strong> {_esc(summary.get('mode', '—'))} (no scoring)</div>"
    )
    body.append(
        f"<div><strong>Backend:</strong> {_esc(summary.get('backend', '—'))}</div>"
    )
    body.append(
        f"<div><strong>Tasks:</strong> {int(summary.get('task_count') or 0)}</div>"
    )
    body.append(
        f"<div><strong>Models:</strong> {len(per_model)}</div>"
    )
    body.append(f"<div><strong>Generated:</strong> {_esc(_now_utc_iso())}</div>")
    body.append("</div>")
    if summary.get("description"):
        body.append(f"<p class='commentary'>{_esc(summary['description'])}</p>")
    body.append("</header>")

    # Telemetry table
    body.append("<h2>Per-model telemetry</h2>")
    if not per_model:
        body.append('<p class="meta">No models recorded in this run.</p>')
    else:
        body.append('<table aria-label="Per-model telemetry">')
        body.append(
            "<thead><tr>"
            "<th>Model</th>"
            "<th class='num'>Completed</th>"
            "<th class='num'>Errored</th>"
            "<th class='num'>Mean TTFT (ms)</th>"
            "<th class='num'>Mean decode tps</th>"
            "<th class='num'>Decode tokens</th>"
            "<th class='num'>Wall (s)</th>"
            "</tr></thead><tbody>"
        )
        for bucket in per_model:
            err_count = int(bucket.get("tasks_errored") or 0)
            err_class = "bad" if err_count else ""
            body.append(
                "<tr>"
                f"<td><code>{_esc(bucket.get('model'))}</code>"
                + (f" <span class='badge bad'>{err_count} err</span>" if err_class else "")
                + "</td>"
                f"<td class='num'>{int(bucket.get('tasks_completed') or 0)}</td>"
                f"<td class='num'>{err_count}</td>"
                f"<td class='num'>"
                f"{'—' if bucket.get('mean_ttft_ms') is None else _fmt_num(bucket['mean_ttft_ms'])}"
                f"</td>"
                f"<td class='num'>"
                f"{'—' if bucket.get('mean_decode_tps') is None else _fmt_num(bucket['mean_decode_tps'], places=2)}"
                f"</td>"
                f"<td class='num'>{int(bucket.get('total_decode_tokens') or 0)}</td>"
                f"<td class='num'>{_fmt_num(bucket.get('wall_time_s'), places=2)}</td>"
                "</tr>"
            )
        body.append("</tbody></table>")

        tps_rows = [
            (str(b.get("model", "?")), float(b["mean_decode_tps"]))
            for b in per_model
            if b.get("mean_decode_tps") is not None
        ]
        if tps_rows:
            body.append('<div class="card">')
            body.append("<h3>Mean decode tps</h3>")
            body.append(_svg_bars(tps_rows))
            body.append("</div>")

    # Per-task collapsible blocks
    by_task: dict[str, list[dict[str, Any]]] = {}
    task_order: list[str] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        if task_id not in by_task:
            task_order.append(task_id)
            by_task[task_id] = []
        by_task[task_id].append(row)

    if task_order:
        body.append("<h2>Side-by-side outputs</h2>")
        body.append(
            "<p class='meta'>One collapsible block per task. Click a task header to expand.</p>"
        )
        for task_id in task_order:
            body.append(_render_custom_task_block_html(task_id, by_task[task_id]))

    body.append("<footer>")
    body.append(
        "Generated by <code>spark-benchmark</code> · "
        "no scripts, no CDN — open the file anywhere."
    )
    body.append("</footer>")
    body.append("</div>")

    return _wrap_document(title=title, body_html="".join(body))


def _render_custom_task_block_html(task_id: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    prompt = str(rows[0].get("prompt") or "")
    has_error = any(row.get("error") for row in rows)
    badge = " <span class='badge bad'>contains errors</span>" if has_error else ""
    parts: list[str] = []
    parts.append("<details>")
    parts.append(
        f"<summary>Task <code>{_esc(task_id)}</code>{badge}</summary>"
    )
    parts.append('<div class="task-block">')
    parts.append('<div class="prompt">')
    parts.append("<strong>Prompt</strong>")
    parts.append(f"<pre>{_esc(prompt) or '(empty)'}</pre>")
    parts.append("</div>")

    for row in rows:
        model = _esc(row.get("model") or "?")
        if row.get("error"):
            err_type = _esc(row["error"].get("type", ""))
            err_msg = _esc(str(row["error"].get("message", ""))[:1500])
            parts.append('<div class="model-reply">')
            parts.append(
                f'<h4 class="error-line">{model} '
                f'<span class="telemetry">ERROR · {err_type}</span></h4>'
            )
            parts.append(f"<pre>{err_msg}</pre>")
            parts.append("</div>")
            continue
        gen = row.get("generation") or {}
        metrics = (gen.get("metrics") or {}) if isinstance(gen, dict) else {}
        ttft = metrics.get("ttft_ms")
        decode_time = metrics.get("decode_time_s") or 0.0
        decode_tokens = metrics.get("decode_tokens") or 0
        tps = (decode_tokens / decode_time) if decode_time > 0 and decode_tokens > 0 else None
        finish = gen.get("finish_reason", "?") if isinstance(gen, dict) else "?"
        bits: list[str] = []
        if isinstance(ttft, (int, float)):
            bits.append(f"TTFT {ttft:.0f} ms")
        if tps is not None:
            bits.append(f"{tps:.1f} tps")
        bits.append(f"finish: {finish}")
        telem = ", ".join(bits)
        output = (gen.get("output") or "") if isinstance(gen, dict) else ""
        parts.append('<div class="model-reply">')
        parts.append(
            f'<h4>{model} <span class="telemetry">{_esc(telem)}</span></h4>'
        )
        parts.append(f"<pre>{_esc(output) or '(empty)'}</pre>")
        parts.append("</div>")
    parts.append("</div>")
    parts.append("</details>")
    return "".join(parts)


# --------------------------------------------------------------------- #
# Document wrapper                                                      #
# --------------------------------------------------------------------- #


def _wrap_document(*, title: str, body_html: str) -> str:
    """Wrap the body fragment in a complete, valid HTML5 document."""
    return (
        "<!doctype html>"
        '<html lang="en">'
        "<head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title>"
        f"<style>{_CSS}</style>"
        "</head>"
        f"<body>{body_html}</body>"
        "</html>\n"
    )
