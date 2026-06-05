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
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from spark_benchmark.reporting import (
    _find_suite,
    _overall_rank_rows,
    _suite_commentary,
    _verdict_paragraph,
)


# Canonical suite-name detection. The aggregate carries versioned names
# (e.g. ``code_generation_v1``) while the narrative helpers in
# ``reporting.py`` key by the unversioned slug. Centralising the test
# here keeps the suite renderers in sync with the rest of the codebase.
_CANONICAL_SUITES = (
    "openclaw_speed",
    "hallucination_grounding",
    "practical_structured_output",
    "code_generation",
    "sustained_throughput",
    "long_context_retrieval",
)


def _canonical_suite_key(name: str) -> str | None:
    for canonical in _CANONICAL_SUITES:
        if name == canonical or name.startswith(f"{canonical}_"):
            return canonical
    return None


# --------------------------------------------------------------------- #
# Shared CSS                                                            #
# --------------------------------------------------------------------- #


_CSS = """
:root {
  --fg: #e2e8f0;
  --fg-muted: #94a3b8;
  --fg-faint: #475569;
  --fg-invert: #060d18;
  --bg: #060d18;
  --bg-card: #0d1726;
  --bg-row-alt: #0f1a2e;
  --bg-soft: #111d2e;
  --border: rgba(255,255,255,0.08);
  --border-strong: rgba(255,255,255,0.16);
  --accent: #76b900;
  --accent-strong: #5a9000;
  --accent-cyan: #22d3ee;
  --accent-violet: #a78bfa;
  --good: #4ade80;
  --good-bg: rgba(74,222,128,0.14);
  --warn: #fb923c;
  --warn-bg: rgba(251,146,60,0.14);
  --bad: #f87171;
  --bad-bg: rgba(248,113,113,0.14);
  --gold: #fbbf24;
  --silver: #94a3b8;
  --bronze: #d97706;
  --code-bg: #111d2e;
  --code-fg: #e2e8f0;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.5);
  --shadow-md: 0 4px 16px rgba(0,0,0,0.6);
  --shadow-lg: 0 20px 60px rgba(0,0,0,0.7);
  --nvidia-green: #76b900;
}
* { box-sizing: border-box; }
html, body { background: var(--bg); }
body {
  margin: 0;
  padding: 0 0 64px;
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
               Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.container { max-width: 1180px; margin: 0 auto; padding: 0 24px; }
h1 { margin: 0 0 8px; font-size: 44px; letter-spacing: -0.025em; line-height: 1.05; font-weight: 700; }
h2 { margin: 40px 0 14px; font-size: 24px; letter-spacing: -0.015em; font-weight: 700; }
h3 { margin: 22px 0 8px; font-size: 17px; letter-spacing: -0.01em; font-weight: 600; }
h4 { margin: 14px 0 6px; font-size: 14px; font-weight: 600; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.04em; }
p, li { color: var(--fg); }
small, .meta { color: var(--fg-muted); font-size: 13px; }

/* ------------ Hero ------------ */
.hero {
  position: relative;
  margin-bottom: 28px;
  padding: 48px 24px 52px;
  color: var(--fg);
  background: linear-gradient(135deg, #060d18 0%, #0a1628 40%, #071a0e 100%);
  border-bottom: 1px solid rgba(118,185,0,0.3);
  overflow: hidden;
}
.hero::before {
  content: "";
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 80% 50%, rgba(118,185,0,0.07) 0%, transparent 60%);
  pointer-events: none;
}
.hero .hero-brand {
  position: absolute; top: 18px; right: 28px;
  font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
  color: var(--fg-muted);
  z-index: 2;
}
.hero .hero-brand .nv { color: var(--nvidia-green); }
.hero .hero-inner {
  position: relative; z-index: 1;
  max-width: 1180px; margin: 0 auto; padding: 0 24px;
  display: flex; flex-wrap: wrap; gap: 32px; align-items: flex-start;
}
.hero .hero-main { flex: 1 1 460px; min-width: 0; }
.hero .hero-breadcrumb {
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--nvidia-green);
  margin-bottom: 10px;
}
.hero h1 { color: #e2e8f0; font-size: 52px; }
.hero .tagline { color: rgba(226,232,240,0.78); font-size: 17px; margin-top: 4px; max-width: 720px; }
.hero .hero-meta {
  display: flex; flex-wrap: wrap; gap: 8px 16px;
  margin-top: 16px;
  font-size: 13px;
  color: rgba(226,232,240,0.6);
}
.hero .hero-meta code {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14);
  padding: 1px 8px; border-radius: 6px; color: var(--fg);
}
.hero .winner-card {
  flex: 0 1 300px;
  min-width: 220px;
  align-self: flex-start;
  margin-top: 4px;
  display: flex; flex-direction: column;
  padding: 18px 22px;
  background: rgba(251,191,36,0.06);
  border: 1px solid rgba(251,191,36,0.4);
  border-radius: 14px;
  box-shadow: 0 0 28px rgba(251,191,36,0.10), var(--shadow-md);
}
.hero .winner-card .label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--gold); opacity: 0.9;
}
.hero .winner-card .name {
  font-size: 22px; font-weight: 700; margin: 6px 0 4px; letter-spacing: -0.01em; color: #e2e8f0;
}
.hero .winner-card .name code { background: transparent; border: none; padding: 0; color: #e2e8f0; font: inherit; }
.hero .winner-card .reason { color: rgba(226,232,240,0.72); font-size: 13px; }
.hero .winner-card .kpi-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.hero .winner-card .kpi-chip {
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 6px;
  padding: 2px 8px;
  font-size: 11px; font-weight: 600;
  color: var(--fg-muted);
}

/* ------------ Stat tiles ------------ */
.stat-tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin: -28px auto 32px;
  max-width: 1180px;
  padding: 0 24px;
  position: relative;
  z-index: 2;
}
.stat-tile {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
  box-shadow: var(--shadow-md);
}
.stat-tile .label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--fg-muted); font-weight: 600; }
.stat-tile .value { font-size: 28px; font-weight: 700; color: var(--fg); letter-spacing: -0.01em; margin-top: 4px; line-height: 1.1; }
.stat-tile .sub { font-size: 12px; color: var(--fg-muted); margin-top: 2px; }

/* ------------ Executive Summary ------------ */
.exec-summary { margin: 0 0 28px; }
.exec-summary h2 { color: var(--fg); }
.exec-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 14px;
  margin-top: 14px;
}
.exec-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
}
.exec-card .ec-icon { font-size: 20px; margin-bottom: 8px; line-height: 1; }
.exec-card .ec-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--fg-muted); margin-bottom: 6px;
}
.exec-card .ec-value { font-size: 16px; font-weight: 700; color: var(--fg); margin-bottom: 4px; word-break: break-all; }
.exec-card .ec-detail { font-size: 12px; color: var(--fg-muted); line-height: 1.45; }

/* ------------ Overall Rankings cards ------------ */
.rankings-section { margin: 28px 0; }
.rankings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px;
  margin-top: 14px;
}
.ranking-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
}
.ranking-card.rank-1 { border-color: rgba(251,191,36,0.45); background: rgba(251,191,36,0.04); }
.ranking-card.rank-2 { border-color: rgba(148,163,184,0.4); }
.ranking-card.rank-3 { border-color: rgba(217,119,6,0.4); }
.ranking-card .rc-medal {
  font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 6px;
}
.ranking-card.rank-1 .rc-medal { color: var(--gold); }
.ranking-card.rank-2 .rc-medal { color: var(--silver); }
.ranking-card.rank-3 .rc-medal { color: var(--bronze); }
.ranking-card .rc-name { font-size: 16px; font-weight: 700; color: var(--fg); margin-bottom: 5px; word-break: break-all; }
.ranking-card .rc-score { font-size: 12px; font-weight: 600; color: var(--accent); margin-bottom: 10px; }
.ranking-card .rc-bullets { list-style: none; margin: 0; padding: 0; }
.ranking-card .rc-bullets li { font-size: 12px; color: var(--fg-muted); padding: 2px 0; }
.ranking-card .rc-bullets li::before { content: "·"; margin-right: 6px; color: var(--fg-faint); }

/* ------------ Cards / tables ------------ */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 22px 24px;
  margin: 16px 0 28px;
  box-shadow: var(--shadow-sm);
}
.card.ok    { border-top: 4px solid var(--good); }
.card.warn  { border-top: 4px solid var(--warn); }
.card.bad   { border-top: 4px solid var(--bad); }
.card.accent{ border-top: 4px solid var(--accent); }
.card h2:first-child, .card h3:first-child { margin-top: 0; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}
th, td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: middle;
}
thead th { background: var(--bg-soft); font-weight: 600; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.04em; font-size: 12px; position: sticky; top: 0; z-index: 1; }
tbody tr:nth-child(even) td { background: var(--bg-row-alt); }
tbody tr:hover td { background: var(--bg-soft); }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td code { background: var(--bg-soft); padding: 1px 6px; border-radius: 5px; font-size: 12.5px; color: var(--fg); }

/* ------------ Pass-rate cells (color-graded fill behind value) ------------ */
td.cell-pct {
  position: relative;
  font-variant-numeric: tabular-nums;
  text-align: right;
  font-weight: 600;
  background-image: linear-gradient(90deg, var(--cell-fill) var(--cell-pct), transparent var(--cell-pct));
  background-repeat: no-repeat;
}
td.cell-pct[data-band="good"] { --cell-fill: var(--good-bg); color: var(--good); }
td.cell-pct[data-band="warn"] { --cell-fill: var(--warn-bg); color: var(--warn); }
td.cell-pct[data-band="bad"]  { --cell-fill: var(--bad-bg);  color: var(--bad); }
td.cell-pct[data-band="na"]   { color: var(--fg-faint); font-weight: 400; }

/* ------------ Badges ------------ */
.badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  background: var(--bg-soft);
  color: var(--fg-muted);
  margin-left: 6px;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.badge.ok   { background: var(--good-bg); color: var(--good); }
.badge.warn { background: var(--warn-bg); color: var(--warn); }
.badge.bad  { background: var(--bad-bg);  color: var(--bad);  }
.badge.accent { background: rgba(118,185,0,0.15); color: var(--accent); }

/* ------------ Commentary / prose ------------ */
.commentary { font-style: italic; color: var(--fg-muted); margin: 6px 0 12px; line-height: 1.5; }

/* ------------ Code / pre ------------ */
pre, code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo,
               Consolas, "Liberation Mono", monospace;
  font-size: 13px;
}
pre {
  background: var(--code-bg);
  color: var(--code-fg);
  padding: 14px 16px;
  border-radius: 10px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 8px 0 16px;
}

/* ------------ Details / collapsibles ------------ */
details {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-card);
  padding: 4px 16px;
  margin: 10px 0;
  box-shadow: var(--shadow-sm);
  transition: border-color 0.15s ease;
}
details[open] { padding-bottom: 16px; border-color: var(--border-strong); }
details[open] > summary { border-bottom: 1px solid var(--border); margin-bottom: 6px; }
details > summary {
  cursor: pointer;
  font-weight: 600;
  padding: 10px 0;
  outline: none;
  list-style: none;
  display: flex; align-items: center; gap: 8px;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before {
  content: "▸";
  color: var(--fg-faint);
  transition: transform 0.15s ease;
  font-size: 12px;
}
details[open] > summary::before { transform: rotate(90deg); color: var(--accent); }
.task-block { padding-top: 4px; }
.task-block .prompt { margin: 4px 0 12px; }
.model-reply { margin: 6px 0 14px; }
.model-reply h4 {
  margin: 12px 0 4px;
  font-size: 15px;
  text-transform: none;
  letter-spacing: normal;
  color: var(--fg);
  display: flex; align-items: baseline; gap: 8px;
}
.model-reply h4 .telemetry {
  color: var(--fg-muted);
  font-size: 12px;
  font-weight: 400;
}
.error-line { color: var(--bad); font-weight: 600; }

/* ------------ Bar charts (horizontal) ------------ */
.bar-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
.bar-row .label {
  flex: 0 0 180px; font-size: 13px; color: var(--fg);
  text-overflow: ellipsis; overflow: hidden; white-space: nowrap;
}
.bar-row .value {
  flex: 0 0 84px; text-align: right;
  font-variant-numeric: tabular-nums; font-size: 13px;
  color: var(--fg-muted); font-weight: 600;
}
.bar-row svg { flex: 1 1 auto; min-width: 0; height: 14px; display: block; }
.bar-row .dual-bars-track { min-width: 0; }
.bar-row .dual-bars-track svg { width: 100%; height: auto; display: block; }
svg.lines { width: 100%; height: auto; display: block; }

/* ------------ Per-suite 3-up dashboard grid ------------ */
.suite-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
  margin: 16px 0;
}
.suite-grid .panel {
  background: var(--bg-soft);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}
.suite-grid .panel h4 {
  margin: 0 0 10px;
  color: var(--fg-muted);
}
.suite-wide { margin: 16px 0; }

/* ------------ Pass-fail strips ------------ */
.pf-strip { display: flex; gap: 2px; flex-wrap: wrap; padding: 6px 0; }
.pf-strip .cell {
  width: 14px; height: 14px;
  border-radius: 3px;
  background: var(--bg-row-alt);
}
.pf-strip .cell.pass { background: var(--good); }
.pf-strip .cell.fail { background: var(--bad); }
.pf-strip .cell.na   { background: var(--fg-faint); opacity: 0.4; }
.pf-row { display: grid; grid-template-columns: 180px 1fr; gap: 12px; align-items: center; padding: 4px 0; }
.pf-row .label { font-size: 13px; color: var(--fg); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ------------ Section verdict / recommendation ------------ */
.verdict-card {
  background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-soft) 100%);
  border: 1px solid var(--border);
  border-left: 5px solid var(--accent);
  border-radius: 14px;
  padding: 22px 26px;
  margin: 0 0 32px;
  box-shadow: var(--shadow-md);
}
.verdict-card h2 { margin-top: 0; }
.verdict-card .recommendation { font-size: 15px; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }
.verdict-card .recommendation strong { color: var(--accent); }

/* ------------ Footer ------------ */
footer {
  margin-top: 56px;
  padding: 18px 24px;
  border-top: 1px solid var(--border);
  color: var(--fg-faint); font-size: 12px;
  text-align: center;
}
footer code { background: transparent; padding: 0; color: var(--fg-muted); }

/* ------------ Print styles ------------ */
@media print {
  body { background: #060d18; }
  .hero { background: #071a0e; color: #e2e8f0; padding: 24px; }
  .hero::before { display: none; }
  .stat-tile, .card, table, details { box-shadow: none; break-inside: avoid; }
  details { border: 1px solid var(--border-strong); }
  details > summary { background: var(--bg-soft); }
  details:not([open]) > summary { padding: 4px 8px; }
  pre { white-space: pre-wrap; }
  thead th { position: static; }
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
    accent: str = "#6366f1",
    value_format: str = "auto",
    invert_color: bool = False,
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
        # Pick a per-row accent: when ``invert_color`` is set (used for
        # "lower is better" metrics like TTFT), the *smallest* values
        # get the strongest accent and the largest get a faded one.
        # Otherwise everyone shares the supplied accent.
        if invert_color:
            row_accent = _gradient_color_for_ratio(1.0 - ratio)
        else:
            row_accent = accent
        # Use viewBox so the bar resizes with the flex container.
        # Note: avoid percentage widths here — tests assert "%" not in html
        # for num-format bars. Use viewBox units instead.
        svg = (
            f'<svg viewBox="0 0 100 1" preserveAspectRatio="none" aria-hidden="true">'
            f'<rect x="0" y="0" width="100" height="1" fill="#0d1726" rx="1"/>'
            f'<rect x="0" y="0" width="100" height="1" fill="rgba(255,255,255,0.08)" />'
            f'<rect x="0" y="0" width="{ratio * 100:.2f}" height="1" fill="{row_accent}" />'
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


def _gradient_color_for_ratio(ratio: float) -> str:
    """Return a hex colour interpolated through bad → warn → good.

    Used by inverted bar charts (lower = better) so the fastest model
    is solid green and the slowest is red, with no lookup table to
    maintain. ``ratio`` is clamped to ``[0, 1]``.
    """
    ratio = max(0.0, min(1.0, ratio))
    # 3-stop gradient: red (#dc2626) → amber (#d97706) → green (#16a34a).
    stops = ((0xDC, 0x26, 0x26), (0xD9, 0x77, 0x06), (0x16, 0xA3, 0x4A))
    if ratio < 0.5:
        a, b = stops[0], stops[1]
        t = ratio * 2.0
    else:
        a, b = stops[1], stops[2]
        t = (ratio - 0.5) * 2.0
    r = round(a[0] + (b[0] - a[0]) * t)
    g = round(a[1] + (b[1] - a[1]) * t)
    bl = round(a[2] + (b[2] - a[2]) * t)
    return f"#{r:02x}{g:02x}{bl:02x}"


def _band_for_pass_rate(value: float | None) -> str:
    """Map a 0..1 pass-rate to a CSS color band: good / warn / bad / na."""
    if value is None:
        return "na"
    v = float(value)
    if v >= 0.95:
        return "good"
    if v >= 0.80:
        return "warn"
    return "bad"


def _cell_pct_html(value: float | None) -> str:
    """Render a ``<td class="cell-pct">`` for a 0..1 pass-rate value.

    The cell carries a CSS ``--cell-pct`` custom property so the
    proportional fill (good_bg / warn_bg / bad_bg) lines up with the
    numeric value. Empty / missing values render as a dimmed em-dash.
    """
    band = _band_for_pass_rate(value)
    if value is None:
        return f'<td class="cell-pct" data-band="{band}" style="--cell-pct: 0%;">—</td>'
    pct = max(0.0, min(1.0, float(value))) * 100.0
    return (
        f'<td class="cell-pct" data-band="{band}" '
        f'style="--cell-pct: {pct:.1f}%;">{_fmt_pct(value)}</td>'
    )


def _svg_heatmap(
    row_labels: list[str],
    col_labels: list[str],
    values: dict[tuple[str, str], float | None],
    *,
    cell_w: int = 62,
    cell_h: int = 34,
    left_pad: int = 92,
) -> str:
    """Render a pass-rate heatmap (rows × cols) as scalable inline SVG.

    ``values`` is keyed by ``(row_label, col_label)`` and holds a 0..1
    pass-rate, or ``None`` for cells that didn't run (e.g. a context
    length the model can't load — rendered as a dimmed N/A tile). Colour
    runs red → amber → green via :func:`_gradient_color_for_ratio`.
    ``left_pad`` widens the row-label gutter for long labels (e.g. model
    names).
    """
    if not row_labels or not col_labels:
        return ""
    top_pad = 24
    pad = 6
    width = left_pad + len(col_labels) * cell_w + pad
    height = top_pad + len(row_labels) * cell_h + pad
    parts = [
        f'<svg class="heatmap" viewBox="0 0 {width} {height}" '
        f'style="width:100%;height:auto;display:block;max-width:{width}px;" '
        f'role="img" preserveAspectRatio="xMinYMin meet">'
        f'<rect width="100%" height="100%" fill="#0d1726" rx="6"/>'
    ]
    # Column (depth) headers.
    for ci, col in enumerate(col_labels):
        cx = left_pad + ci * cell_w + cell_w / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{top_pad - 9}" text-anchor="middle" '
            f'font-size="11" fill="var(--fg-muted)" font-weight="600">{_esc(col)}</text>'
        )
    # Rows.
    for ri, row in enumerate(row_labels):
        ry = top_pad + ri * cell_h
        parts.append(
            f'<text x="{left_pad - 8}" y="{ry + cell_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="var(--fg-muted)" font-family="ui-monospace,monospace">{_esc(row)}</text>'
        )
        for ci, col in enumerate(col_labels):
            cx = left_pad + ci * cell_w
            value = values.get((row, col))
            if value is None:
                fill = "var(--surface-2, #1f2733)"
                label = "N/A"
                text_fill = "var(--fg-muted)"
            else:
                fill = _gradient_color_for_ratio(float(value))
                label = _fmt_pct(value)
                text_fill = "#ffffff"
            parts.append(
                f'<rect x="{cx + 2:.1f}" y="{ry + 2:.1f}" width="{cell_w - 4}" '
                f'height="{cell_h - 4}" rx="5" fill="{fill}" '
                f'stroke="rgba(0,0,0,0.25)" stroke-width="1"></rect>'
            )
            parts.append(
                f'<text x="{cx + cell_w / 2:.1f}" y="{ry + cell_h / 2 + 4:.1f}" '
                f'text-anchor="middle" font-size="11" font-weight="600" '
                f'fill="{text_fill}">{_esc(label)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------- #
# Inline SVG: line chart, gauge, dual bars, stacked bars, etc.          #
# --------------------------------------------------------------------- #


def _svg_line_chart(
    series: list[tuple[str, list[tuple[float, float]]]],
    *,
    height: int = 120,
    palette: tuple[str, ...] = ("#6366f1", "#06b6d4", "#dc2626", "#16a34a", "#d97706"),
    y_label: str = "",
    x_label: str = "",
    secondary: list[tuple[str, list[tuple[float, float]]]] | None = None,
) -> str:
    """Render one or more ``(label, [(x, y), ...])`` series as an inline-SVG line chart.

    Width is responsive (``viewBox`` + ``preserveAspectRatio="none"``),
    height is the given number of pixels. ``secondary`` is rendered on
    the right axis with dashed lines, useful for overlaying GPU temp on
    top of a tps-over-time chart. Empty input → empty string.

    The chart is intentionally plain: no axes ticks (we put units in the
    surrounding caption instead), no legend (labels appear next to
    series colour swatches in HTML right above the SVG). The point is
    "shape of the curve, at a glance" — for exact numbers the user has
    the JSON.
    """
    series = [(label, points) for label, points in series if points]
    secondary = [(label, points) for label, points in (secondary or []) if points]
    if not series and not secondary:
        return ""

    all_xs = [x for _, pts in series + secondary for x, _ in pts]
    all_primary = [y for _, pts in series for _, y in pts]
    all_secondary = [y for _, pts in secondary for _, y in pts]
    if not all_xs:
        return ""
    x_min = min(all_xs)
    x_max = max(all_xs)
    if x_max <= x_min:
        x_max = x_min + 1.0
    y_min_p = min(all_primary) if all_primary else 0.0
    y_max_p = max(all_primary) if all_primary else 1.0
    if y_max_p <= y_min_p:
        y_max_p = y_min_p + 1.0
    y_min_s = min(all_secondary) if all_secondary else 0.0
    y_max_s = max(all_secondary) if all_secondary else 1.0
    if y_max_s <= y_min_s:
        y_max_s = y_min_s + 1.0
    pad_l, pad_r, pad_t, pad_b = 4, 4, 8, 18
    width = 800  # viewBox units; SVG scales to container width
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def _project(x: float, y: float, *, y_lo: float, y_hi: float) -> tuple[float, float]:
        sx = pad_l + (x - x_min) / (x_max - x_min) * plot_w
        sy = pad_t + plot_h - (y - y_lo) / (y_hi - y_lo) * plot_h
        return sx, sy

    parts: list[str] = []
    parts.append(
        f'<svg class="lines" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" aria-hidden="true">'
        f'<rect width="100%" height="100%" fill="#0d1726" rx="6"/>'
    )
    # Light gridlines at 25/50/75 %.
    for frac in (0.25, 0.5, 0.75):
        gy = pad_t + plot_h * (1 - frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            f'stroke="rgba(255,255,255,0.08)" stroke-width="0.6" />'
        )
    # Primary series.
    for i, (_, points) in enumerate(series):
        accent = palette[i % len(palette)]
        coords = " ".join(
            f"{px:.1f},{py:.1f}"
            for px, py in (_project(x, y, y_lo=y_min_p, y_hi=y_max_p) for x, y in points)
        )
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{accent}" '
            f'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" />'
        )
    # Secondary series, dashed, separate y-axis.
    for i, (_, points) in enumerate(secondary):
        accent = palette[(i + len(series)) % len(palette)]
        coords = " ".join(
            f"{px:.1f},{py:.1f}"
            for px, py in (_project(x, y, y_lo=y_min_s, y_hi=y_max_s) for x, y in points)
        )
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{accent}" '
            f'stroke-width="1.4" stroke-dasharray="3 3" stroke-linejoin="round" />'
        )
    # X axis caption (range only — keeps it small).
    parts.append(
        f'<text x="{pad_l}" y="{height - 4}" font-size="10" fill="#94a3b8">'
        f"{_fmt_num(x_min, places=0)}{_esc(x_label)}</text>"
    )
    parts.append(
        f'<text x="{pad_l + plot_w}" y="{height - 4}" font-size="10" fill="#94a3b8" '
        f'text-anchor="end">{_fmt_num(x_max, places=0)}{_esc(x_label)}</text>'
    )
    if y_label:
        parts.append(
            f'<text x="{pad_l + 2}" y="{pad_t + 10}" font-size="10" fill="#94a3b8">'
            f"{_esc(y_label)}</text>"
        )
    parts.append("</svg>")

    # Legend HTML (sits above the SVG, color swatches → labels).
    legend_parts = ['<div class="line-legend" style="display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--fg-muted);margin-bottom:6px;">']
    for i, (label, _) in enumerate(series):
        accent = palette[i % len(palette)]
        legend_parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;">'
            f'<span style="display:inline-block;width:12px;height:3px;background:{accent};border-radius:2px;"></span>'
            f"{_esc(label)}</span>"
        )
    for i, (label, _) in enumerate(secondary):
        accent = palette[(i + len(series)) % len(palette)]
        legend_parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;">'
            f'<span style="display:inline-block;width:12px;height:0;border-top:2px dashed {accent};"></span>'
            f"{_esc(label)} <small>(2nd axis)</small></span>"
        )
    legend_parts.append("</div>")
    return "".join(legend_parts) + "".join(parts)


def _svg_gauge(
    value: float,
    *,
    max_value: float = 1.0,
    label: str = "",
    suffix: str = "",
    invert: bool = False,
) -> str:
    """Render a 180-degree semicircle gauge as inline SVG.

    ``value`` is clamped to ``[0, max_value]``. The fill is colour-graded
    along bad → warn → good. ``invert=True`` flips the colour mapping
    (useful for "lower is better" metrics, e.g. throttle ratio).
    The label and the value are stacked centre-aligned underneath the
    arc.
    """
    if max_value <= 0:
        max_value = 1.0
    ratio = max(0.0, min(1.0, value / max_value))
    sweep = ratio * math.pi  # 0..π radians
    # Semicircle: cx=100, cy=80, r=70, starts at (30, 80) → (170, 80)
    # Filled arc: same circle, end angle = π * ratio measured from left.
    end_x = 100 - 70 * math.cos(sweep)
    end_y = 80 - 70 * math.sin(sweep)
    color_ratio = (1.0 - ratio) if invert else ratio
    color = _gradient_color_for_ratio(color_ratio)
    return (
        f'<svg viewBox="0 0 200 110" preserveAspectRatio="xMidYMid meet" '
        f'class="gauge" aria-hidden="true" style="width:100%;height:auto;">'
        f'<rect width="100%" height="100%" fill="#0d1726" rx="6"/>'
        f'<path d="M 30 80 A 70 70 0 0 1 170 80" stroke="#1e293b" stroke-width="14" fill="none" stroke-linecap="round" />'
        f'<path d="M 30 80 A 70 70 0 0 1 {end_x:.2f} {end_y:.2f}" '
        f'stroke="{color}" stroke-width="14" fill="none" stroke-linecap="round" />'
        f'<text x="100" y="76" text-anchor="middle" font-size="22" font-weight="700" fill="#e2e8f0">'
        f"{_fmt_num(value * 100 if max_value == 1.0 else value, places=0)}{_esc(suffix)}</text>"
        f'<text x="100" y="100" text-anchor="middle" font-size="11" fill="#94a3b8">'
        f"{_esc(label)}</text>"
        f"</svg>"
    )


def _svg_dual_bars(
    rows: list[tuple[str, float, float]],
    *,
    label_a: str = "A",
    label_b: str = "B",
    accent_a: str = "#6366f1",
    accent_b: str = "#06b6d4",
    suffix: str = "",
) -> str:
    """Render rows as paired horizontal bars (e.g. initial vs sustained).

    Each row gets two stacked thin bars: ``value_a`` on top in
    ``accent_a``, ``value_b`` underneath in ``accent_b``. Values are
    normalised to the global maximum across both series so the visual
    delta between A and B is honest.
    """
    rows = [(str(name), float(va or 0.0), float(vb or 0.0)) for name, va, vb in rows]
    if not rows:
        return ""
    cap = max(max(va, vb) for _, va, vb in rows) or 1.0
    parts: list[str] = []
    parts.append(
        '<div class="line-legend" style="display:flex;gap:14px;font-size:12px;'
        'color:var(--fg-muted);margin-bottom:6px;">'
        f'<span style="display:inline-flex;align-items:center;gap:6px;">'
        f'<span style="width:12px;height:6px;background:{accent_a};border-radius:2px;"></span>'
        f"{_esc(label_a)}</span>"
        f'<span style="display:inline-flex;align-items:center;gap:6px;">'
        f'<span style="width:12px;height:6px;background:{accent_b};border-radius:2px;"></span>'
        f"{_esc(label_b)}</span>"
        "</div>"
    )
    for name, va, vb in rows:
        ra = va / cap
        rb = vb / cap
        parts.append(
            f'<div class="bar-row dual-bars" style="align-items:flex-start;">'
            f'<span class="label" title="{_esc(name)}">{_esc(name)}</span>'
            f'<div class="dual-bars-track" style="flex:1 1 0;min-width:0;display:flex;'
            f'flex-direction:column;gap:2px;overflow:hidden;">'
            f'<svg viewBox="0 0 100 1" preserveAspectRatio="none" '
            f'style="width:100%;height:7px;display:block;">'
            f'<rect x="0" y="0" width="100" height="1" fill="#0d1726" rx="1"/>'
            f'<rect x="0" y="0" width="100" height="1" fill="rgba(255,255,255,0.08)"/>'
            f'<rect x="0" y="0" width="{ra * 100:.2f}" height="1" fill="{accent_a}"/></svg>'
            f'<svg viewBox="0 0 100 1" preserveAspectRatio="none" '
            f'style="width:100%;height:7px;display:block;">'
            f'<rect x="0" y="0" width="100" height="1" fill="#0d1726" rx="1"/>'
            f'<rect x="0" y="0" width="100" height="1" fill="rgba(255,255,255,0.08)"/>'
            f'<rect x="0" y="0" width="{rb * 100:.2f}" height="1" fill="{accent_b}"/></svg>'
            f"</div>"
            f'<span class="value">{_fmt_num(va)}{_esc(suffix)} / {_fmt_num(vb)}{_esc(suffix)}</span>'
            f"</div>"
        )
    return "".join(parts)


def _svg_stacked_bars(
    rows: list[tuple[str, list[tuple[str, float]]]],
    *,
    palette: tuple[str, ...] = (
        "#16a34a",  # passed
        "#dc2626",  # failed (catch-all)
        "#d97706",  # timeout / warn
        "#0ea5e9",  # other 1
        "#a855f7",  # other 2
    ),
) -> str:
    """Render rows as stacked horizontal bars.

    Each row is ``(label, [(segment_label, value), ...])``. Segments
    are rendered left-to-right in input order. Total per row is
    normalised to the row's own sum (so each row's bar fills the same
    visual width but the *colour split* shows the proportion). This is
    deliberate: the goal is "what fraction passed vs. timed out vs.
    failed" per model, not absolute counts.
    """
    rows = [(name, list(segments)) for name, segments in rows if segments]
    if not rows:
        return ""
    parts: list[str] = []
    seg_labels = [seg for _, segs in rows for seg, _ in segs]
    seen: dict[str, str] = {}
    for i, lab in enumerate(seg_labels):
        if lab not in seen:
            seen[lab] = palette[len(seen) % len(palette)]
    parts.append(
        '<div class="line-legend" style="display:flex;flex-wrap:wrap;gap:6px 14px;'
        'font-size:12px;color:var(--fg-muted);margin-bottom:6px;">'
    )
    for label, color in seen.items():
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;">'
            f'<span style="width:12px;height:8px;background:{color};border-radius:2px;"></span>'
            f"{_esc(label)}</span>"
        )
    parts.append("</div>")
    for name, segments in rows:
        total = sum(max(0.0, float(v)) for _, v in segments) or 1.0
        offset = 0.0
        chunks: list[str] = [
            '<svg viewBox="0 0 100 1" preserveAspectRatio="none" '
            'aria-hidden="true">'
            '<rect x="0" y="0" width="100" height="1" fill="#0d1726" rx="1"/>'
            '<rect x="0" y="0" width="100" height="1" fill="rgba(255,255,255,0.08)"/>'
        ]
        for seg_label, value in segments:
            v = max(0.0, float(value))
            if v <= 0:
                continue
            w = v / total * 100.0
            color = seen[seg_label]
            chunks.append(
                f'<rect x="{offset:.2f}" y="0" width="{w:.2f}" height="1" fill="{color}" />'
            )
            offset += w
        chunks.append("</svg>")
        # Build a "53/12/3" hint to right of the bar.
        hint = " / ".join(f"{int(v)}" for _, v in segments)
        parts.append(
            f'<div class="bar-row">'
            f'<span class="label" title="{_esc(name)}">{_esc(name)}</span>'
            f"{''.join(chunks)}"
            f'<span class="value">{_esc(hint)}</span>'
            f"</div>"
        )
    return "".join(parts)


def _svg_thermometer(
    value: float | None,
    *,
    min_t: float = 30.0,
    max_t: float = 90.0,
    label: str = "",
    suffix: str = " °C",
) -> str:
    """Vertical thermometer-styled bar for a single GPU temp value."""
    if value is None:
        return ""
    v = float(value)
    ratio = max(0.0, min(1.0, (v - min_t) / max((max_t - min_t), 1e-9)))
    color = _gradient_color_for_ratio(ratio)  # hotter = redder
    fill_h = ratio * 80.0  # bulb is at y=90; column starts y=10
    # Note: test asserts exactly 2 <rect> elements, so no background rect here.
    return (
        '<svg viewBox="0 0 60 110" preserveAspectRatio="xMidYMid meet" '
        'class="thermo" aria-hidden="true" style="width:100%;height:auto;max-height:140px;">'
        f'<rect x="22" y="10" width="16" height="80" rx="8" fill="rgba(255,255,255,0.12)"/>'
        f'<rect x="22" y="{10 + (80 - fill_h):.1f}" width="16" height="{fill_h:.1f}" fill="{color}"/>'
        f'<circle cx="30" cy="92" r="12" fill="{color}"/>'
        f'<text x="30" y="96" text-anchor="middle" font-size="9" fill="white" font-weight="700">{_fmt_num(v, places=0)}</text>'
        f'<text x="30" y="108" text-anchor="middle" font-size="9" fill="#94a3b8">{_esc(label)}{_esc(suffix)}</text>'
        "</svg>"
    )


def _pass_fail_strip_html(
    per_model: list[tuple[str, list[bool | None]]],
) -> str:
    """Render per-task pass/fail strips, one row per model.

    ``per_model`` is a list of ``(model_name, [pass1, pass2, ...])``
    tuples where each entry is ``True`` (pass), ``False`` (fail), or
    ``None`` (no data — rendered as a faint grey cell). Empty input
    returns an empty string.
    """
    per_model = [(name, list(results)) for name, results in per_model if results]
    if not per_model:
        return ""
    parts: list[str] = []
    for name, results in per_model:
        cells = "".join(
            f'<span class="cell {"pass" if r is True else "fail" if r is False else "na"}"></span>'
            for r in results
        )
        parts.append(
            f'<div class="pf-row">'
            f'<span class="label" title="{_esc(name)}">{_esc(name)}</span>'
            f'<div class="pf-strip">{cells}</div>'
            f"</div>"
        )
    return "".join(parts)


# --------------------------------------------------------------------- #
# Lazy loaders for per-suite raw data (results.jsonl, telemetry-*.jsonl)
# --------------------------------------------------------------------- #


def _load_results_rows(run_dir: str | Path | None) -> list[dict[str, Any]]:
    """Read ``results.jsonl`` from a suite run-dir, best-effort.

    Returns an empty list if the file is missing or malformed — the
    HTML renderer treats absence of raw rows as "no per-task data, use
    aggregate values only" rather than raising.
    """
    if not run_dir:
        return []
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def _load_telemetry_samples(
    run_dir: str | Path | None,
    model_name: str,
    *,
    max_points: int = 240,
) -> list[dict[str, Any]]:
    """Read ``telemetry-<model>.jsonl`` and downsample uniformly.

    Sustained-throughput soaks at default settings produce ~10 samples
    per second over multiple minutes — a 30-minute soak yields ~18 000
    points. Downsampling to ``max_points`` keeps the resulting SVG
    polylines under a few KB while preserving the shape of the curve.
    """
    if not run_dir:
        return []
    path = Path(run_dir) / f"telemetry-{model_name}.jsonl"
    if not path.exists():
        return []
    samples: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    if len(samples) <= max_points:
        return samples
    step = len(samples) / max_points
    return [samples[int(i * step)] for i in range(max_points)]


def _per_task_pass_fail(rows: list[dict[str, Any]], model: str) -> list[bool | None]:
    """For a reliability suite, extract per-task pass/fail for one model.

    Order is preserved by ``task_id`` first-seen across the rows. Rows
    missing ``evaluation.passed`` are treated as ``None`` (no signal).
    """
    seen: dict[str, bool | None] = {}
    for row in rows:
        if str(row.get("model")) != model:
            continue
        tid = str(row.get("task_id") or "")
        if not tid:
            continue
        evaluation = row.get("evaluation") or {}
        passed = evaluation.get("passed") if isinstance(evaluation, dict) else None
        if tid not in seen:
            seen[tid] = bool(passed) if isinstance(passed, bool) else None
    return list(seen.values())


def _code_gen_status_breakdown(
    rows: list[dict[str, Any]], model: str
) -> list[tuple[str, float]]:
    """For ``code_generation`` rows, count sandbox status occurrences.

    Returns ``[(status, count), ...]`` for the model, ordered with the
    "passed" bucket first (so the stacked-bar legend reads naturally).
    Sandbox statuses come from ``row.samples[*].sandbox.status`` and
    are one of: ``passed``, ``failed``, ``timeout``, ``oom``,
    ``compile_error``, ``runtime_error``.
    """
    counts: dict[str, int] = {}
    for row in rows:
        if str(row.get("model")) != model:
            continue
        for sample in row.get("samples") or []:
            sandbox = (sample or {}).get("sandbox") or {}
            status = str(sandbox.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
    if not counts:
        return []
    ordered = ["passed", "failed", "timeout", "oom", "compile_error", "runtime_error"]
    sorted_keys = [k for k in ordered if k in counts] + sorted(
        k for k in counts if k not in ordered
    )
    return [(k, float(counts[k])) for k in sorted_keys]


# --------------------------------------------------------------------- #
# Executive summary + rankings helpers                                  #
# --------------------------------------------------------------------- #


def _exec_summary_html(
    ranking: list[dict[str, Any]],
    suites: list[dict[str, Any]],
) -> str:
    """Render a 4-card executive-summary section derived from ranking data."""
    if not ranking:
        return ""
    cards: list[str] = []

    # Card 1: Best Overall Balance
    top = ranking[0]
    reason_bits: list[str] = []
    if top.get("grounding_rate") is not None:
        reason_bits.append(f"Grounding {_fmt_pct(top['grounding_rate'])}")
    if top.get("structured_rate") is not None:
        reason_bits.append(f"Structured {_fmt_pct(top['structured_rate'])}")
    if top.get("avg_tokens_per_s") is not None:
        reason_bits.append(f"{_fmt_num(top['avg_tokens_per_s'])} tok/s")
    detail1 = " · ".join(reason_bits) if reason_bits else "Top combined score across all suites"
    cards.append(
        '<div class="exec-card">'
        '<div class="ec-icon">🏆</div>'
        '<div class="ec-title">Best Overall Balance</div>'
        f'<div class="ec-value">{_esc(top["model"])}</div>'
        f'<div class="ec-detail">{_esc(detail1)}</div>'
        '</div>'
    )

    # Card 2: Strongest single-suite pass rate
    best_suite_model: str | None = None
    best_suite_name: str | None = None
    best_suite_rate: float = -1.0
    for suite in suites:
        suite_name = str(suite.get("suite") or "")
        for model in suite.get("models") or []:
            rate = model.get("pass_rate")
            if rate is not None and float(rate) > best_suite_rate:
                best_suite_rate = float(rate)
                best_suite_model = str(model.get("model") or "?")
                best_suite_name = suite_name
    if best_suite_model:
        cards.append(
            '<div class="exec-card">'
            '<div class="ec-icon">🎯</div>'
            f'<div class="ec-title">Strongest — {_esc(best_suite_name or "suite")}</div>'
            f'<div class="ec-value">{_esc(best_suite_model)}</div>'
            f'<div class="ec-detail">{_fmt_pct(best_suite_rate)} pass rate</div>'
            '</div>'
        )

    # Card 3: Highest throughput
    best_tps_model: str | None = None
    best_tps: float = -1.0
    for r in ranking:
        tps = r.get("avg_tokens_per_s")
        if tps is not None and float(tps) > best_tps:
            best_tps = float(tps)
            best_tps_model = str(r["model"])
    if best_tps_model:
        cards.append(
            '<div class="exec-card">'
            '<div class="ec-icon">⚡</div>'
            '<div class="ec-title">Highest Throughput</div>'
            f'<div class="ec-value">{_esc(best_tps_model)}</div>'
            f'<div class="ec-detail">{_fmt_num(best_tps)} tok/s average decode</div>'
            '</div>'
        )

    # Card 4: Key Finding — context degradation if available, otherwise score gap
    lc_finding: str | None = None
    for suite in suites:
        if _canonical_suite_key(str(suite.get("suite") or "")) == "long_context_retrieval":
            for model in suite.get("models") or []:
                ffl = model.get("first_failure_length")
                if ffl:
                    lc_finding = (
                        f"Context Degradation — {_esc(str(model.get('model') or '?'))} "
                        f"degrades at {int(ffl):,} tokens"
                    )
                    break
            if lc_finding:
                break
    if not lc_finding:
        if len(ranking) >= 2:
            score_gap = float(ranking[0]["overall_score"]) - float(ranking[-1]["overall_score"])
            lc_finding = (
                f"{_esc(ranking[0]['model'])} leads by "
                f"{score_gap * 100:.1f} score points over "
                f"{_esc(ranking[-1]['model'])}"
            )
        else:
            lc_finding = "Single model evaluated — no comparison baseline"
    cards.append(
        '<div class="exec-card">'
        '<div class="ec-icon">⚠️</div>'
        '<div class="ec-title">Key Finding</div>'
        f'<div class="ec-value ec-detail">{lc_finding}</div>'
        '</div>'
    )

    return (
        '<section class="exec-summary">'
        '<h2>Executive Summary</h2>'
        '<div class="exec-grid">' + "".join(cards) + '</div>'
        '</section>'
    )


def _overall_rankings_cards_html(ranking: list[dict[str, Any]]) -> str:
    """Render numbered ranking cards with score and stat bullets."""
    if not ranking:
        return ""
    rank_class = {1: "rank-1", 2: "rank-2", 3: "rank-3"}
    rank_label = {1: "🥇 #1 — Gold", 2: "🥈 #2 — Silver", 3: "🥉 #3 — Bronze"}
    parts: list[str] = []
    parts.append('<section class="rankings-section">')
    parts.append('<h2>Overall Rankings</h2>')
    parts.append('<div class="rankings-grid">')
    for i, row in enumerate(ranking, start=1):
        score = round(float(row["overall_score"]) * 100)
        bullets: list[str] = []
        if row.get("grounding_rate") is not None:
            bullets.append(f"Grounding: {_fmt_pct(row['grounding_rate'])}")
        if row.get("structured_rate") is not None:
            bullets.append(f"Structured JSON: {_fmt_pct(row['structured_rate'])}")
        if row.get("avg_ttft_ms") is not None:
            bullets.append(f"TTFT: {_fmt_num(row['avg_ttft_ms'])} ms")
        if row.get("avg_tokens_per_s") is not None:
            bullets.append(f"Throughput: {_fmt_num(row['avg_tokens_per_s'])} tok/s")
        bullets = bullets[:3]
        if not bullets:
            bullets = ["No detailed metrics available"]
        li_items = "".join(f'<li>{_esc(b)}</li>' for b in bullets)
        medal = rank_label.get(i, f"#{i}")
        css_class = rank_class.get(i, "")
        parts.append(
            f'<div class="ranking-card {css_class}">'
            f'<div class="rc-medal">{medal}</div>'
            f'<div class="rc-name"><code>{_esc(row["model"])}</code></div>'
            f'<div class="rc-score">OVERALL SCORE {score}/100</div>'
            f'<ul class="rc-bullets">{li_items}</ul>'
            '</div>'
        )
    parts.append('</div>')
    parts.append('</section>')
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

    Layout: hero banner → stat tiles → verdict card → overall ranking
    table + chart → per-suite dashboard cards (each with a 3-up grid
    of charts tailored to that suite type) → recent runs tail.
    """
    suites = aggregate.get("suites") or []
    runs = aggregate.get("runs") or []
    selected_models_list = list(selected_models) if selected_models else _model_names_from_suites(suites)
    selected_suites_list = list(selected_suites) if selected_suites else [s["suite"] for s in suites]
    ranking = _overall_rank_rows(aggregate, selected_models_list)
    suite_map = {
        canonical: _find_suite(aggregate, canonical) for canonical in _CANONICAL_SUITES
    }
    winner_name = ranking[0]["model"] if ranking else None

    body: list[str] = []

    # ------------------------------------------------------------- #
    # Hero banner
    # ------------------------------------------------------------- #
    body.append('<section class="hero">')
    body.append('<div class="hero-brand"><span class="nv">NVIDIA</span> | OpenClaw</div>')
    body.append('<div class="hero-inner">')
    body.append('<div class="hero-main">')
    body.append('<div class="hero-breadcrumb">DGX SPARK BENCHMARK</div>')
    body.append(f"<h1>{_esc(title)}</h1>")
    if request:
        body.append(f'<p class="tagline">{_esc(request)}</p>')
    else:
        body.append(
            '<p class="tagline">Reproducible local-LLM benchmark on '
            "NVIDIA DGX Spark — reliability, latency, throughput, and code "
            "generation, with every number reproducible from "
            "<code>results/</code>.</p>"
        )
    body.append('<div class="hero-meta">')
    body.append(
        f"<span><strong>Generated:</strong> {_esc(_now_utc_iso())}</span>"
    )
    body.append(
        f"<span><strong>Runs root:</strong> "
        f"<code>{_esc(aggregate.get('runs_root', '—'))}</code></span>"
    )
    if selected_models_list:
        body.append(
            f"<span><strong>Models:</strong> "
            f"<code>{_esc(', '.join(selected_models_list))}</code></span>"
        )
    body.append("</div>")
    body.append("</div>")  # /.hero-main
    if winner_name and ranking:
        winner_row = ranking[0]
        reason_bits: list[str] = []
        if winner_row.get("grounding_rate") is not None and winner_row["grounding_rate"] >= 0.99:
            reason_bits.append("perfect grounding reliability")
        elif winner_row.get("grounding_rate") is not None and winner_row["grounding_rate"] >= 0.8:
            reason_bits.append("strong grounding reliability")
        if winner_row.get("structured_rate") is not None and winner_row["structured_rate"] >= 0.99:
            reason_bits.append("perfect structured-output adherence")
        if winner_row.get("avg_ttft_ms") is not None:
            reason_bits.append(
                f"TTFT {_fmt_num(winner_row['avg_ttft_ms'])} ms"
            )
        if winner_row.get("avg_tokens_per_s") is not None:
            reason_bits.append(
                f"{_fmt_num(winner_row['avg_tokens_per_s'])} tok/s"
            )
        reason = "; ".join(reason_bits) if reason_bits else "balanced result across all suites"
        kpi_chips: list[str] = []
        if winner_row.get("grounding_rate") is not None:
            kpi_chips.append(
                f'<span class="kpi-chip">Grounding {_fmt_pct(winner_row["grounding_rate"])}</span>'
            )
        if winner_row.get("avg_tokens_per_s") is not None:
            kpi_chips.append(
                f'<span class="kpi-chip">{_fmt_num(winner_row["avg_tokens_per_s"])} tok/s</span>'
            )
        if winner_row.get("avg_ttft_ms") is not None:
            kpi_chips.append(
                f'<span class="kpi-chip">TTFT {_fmt_num(winner_row["avg_ttft_ms"])} ms</span>'
            )
        kpi_html = (
            f'<div class="kpi-chips">{"".join(kpi_chips)}</div>' if kpi_chips else ""
        )
        body.append('<div class="winner-card">')
        body.append('<span class="label">Recommended pick</span>')
        body.append(f'<div class="name"><code>{_esc(winner_name)}</code></div>')
        body.append(f'<div class="reason">{_esc(reason)}</div>')
        if kpi_html:
            body.append(kpi_html)
        body.append("</div>")
    body.append("</div>")  # /.hero-inner
    body.append("</section>")

    # ------------------------------------------------------------- #
    # Stat tiles
    # ------------------------------------------------------------- #
    body.append(_stat_tiles_html(aggregate, suites, runs, ranking))

    body.append('<div class="container">')

    # ------------------------------------------------------------- #
    # Executive Summary (4-card insight strip)
    # ------------------------------------------------------------- #
    body.append(_exec_summary_html(ranking, suites))

    # ------------------------------------------------------------- #
    # Verdict + recommendation card
    # ------------------------------------------------------------- #
    if ranking:
        verdict = _verdict_paragraph(ranking, suite_map)
        body.append('<section class="verdict-card">')
        body.append("<h2>Verdict</h2>")
        if verdict:
            body.append(f"<p>{_esc(verdict)}</p>")
        else:
            body.append(
                "<p>Single-model run — no comparison available. See the "
                "per-suite breakdown below for details.</p>"
            )
        body.append(
            f'<div class="recommendation"><strong>Recommendation:</strong> '
            f"<code>{_esc(winner_name)}</code> is the current default pick — "
            "strongest combined result across reliability and speed.</div>"
        )
        body.append("</section>")

    # ------------------------------------------------------------- #
    # Overall ranking table + bar chart
    # ------------------------------------------------------------- #
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
            badge = ' <span class="badge accent">winner</span>' if i == 1 else ""
            body.append(
                "<tr>"
                f"<td class='num'>{i}</td>"
                f"<td><code>{_esc(row['model'])}</code>{badge}</td>"
                f"<td class='num'>{_fmt_num(row['overall_score'], places=3)}</td>"
                f"{_cell_pct_html(row.get('grounding_rate'))}"
                f"{_cell_pct_html(row.get('structured_rate'))}"
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
        body.append('<div class="card accent">')
        body.append("<h3>Overall score</h3>")
        body.append(_svg_bars(bar_rows, max_value=1.0))
        body.append("</div>")

        body.append(_overall_rankings_cards_html(ranking))

    # ------------------------------------------------------------- #
    # Per-suite dashboard cards (one per suite, suite-specific charts)
    # ------------------------------------------------------------- #
    for suite in suites:
        body.append(_render_suite_block_html(suite))

    # ------------------------------------------------------------- #
    # Recent runs tail
    # ------------------------------------------------------------- #
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

    body.append("</div>")  # /.container

    body.append("<footer>")
    body.append(
        "Generated by <code>spark-benchmark</code> · "
        "self-contained HTML — no scripts, no CDN, no external assets."
    )
    body.append("</footer>")

    return _wrap_document(title=title, body_html="".join(body))


def _stat_tiles_html(
    aggregate: dict[str, Any],
    suites: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
) -> str:
    """Render the four-tile summary strip directly under the hero.

    Tiles: model count, suite count, total tasks (sum of ``total``
    across every suite × model), and overall pass rate (passes /
    total). Edges gracefully — empty aggregate yields zero-tiles.
    """
    model_set: set[str] = set()
    suite_count = len(suites)
    total_tasks = 0
    total_passes = 0
    for suite in suites:
        for model in suite.get("models") or []:
            model_set.add(str(model.get("model") or ""))
            total_tasks += int(model.get("total") or 0)
            total_passes += int(model.get("passes") or 0)
    overall_rate = (total_passes / total_tasks) if total_tasks > 0 else None
    band = _band_for_pass_rate(overall_rate)
    rate_label = _fmt_pct(overall_rate) if overall_rate is not None else "—"
    parts: list[str] = []
    parts.append('<div class="stat-tiles">')
    parts.append(
        f'<div class="stat-tile"><div class="label">Models tested</div>'
        f'<div class="value">{len([m for m in model_set if m])}</div>'
        f'<div class="sub">{_esc(", ".join(sorted(m for m in model_set if m))[:60] or "—")}</div></div>'
    )
    parts.append(
        f'<div class="stat-tile"><div class="label">Suites run</div>'
        f'<div class="value">{suite_count}</div>'
        f'<div class="sub">{len(runs)} run dirs aggregated</div></div>'
    )
    parts.append(
        f'<div class="stat-tile"><div class="label">Total tasks</div>'
        f'<div class="value">{total_tasks}</div>'
        f'<div class="sub">{total_passes} passed</div></div>'
    )
    color = {"good": "var(--good)", "warn": "var(--warn)", "bad": "var(--bad)", "na": "var(--fg-faint)"}[band]
    parts.append(
        f'<div class="stat-tile"><div class="label">Overall pass rate</div>'
        f'<div class="value" style="color: {color};">{rate_label}</div>'
        f'<div class="sub">across all suites × models</div></div>'
    )
    if ranking:
        parts.append(
            f'<div class="stat-tile"><div class="label">Top model score</div>'
            f'<div class="value">{_fmt_num(ranking[0]["overall_score"], places=3)}</div>'
            f'<div class="sub"><code>{_esc(ranking[0]["model"])}</code></div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _suite_titles_and_subtitles() -> dict[str, tuple[str, str]]:
    """Human-readable suite headlines for the per-suite cards."""
    return {
        "openclaw_speed": (
            "Speed probe",
            "First-token latency and steady-state decode throughput.",
        ),
        "hallucination_grounding": (
            "Grounding / hallucination",
            "How often each model stays grounded in the provided context.",
        ),
        "practical_structured_output": (
            "Structured-output reliability",
            "Strict JSON / schema adherence on practical structured-output prompts.",
        ),
        "code_generation": (
            "Code generation (HumanEval-style)",
            "Sandboxed pass@1 — does the generated code actually run and produce the right answer?",
        ),
        "sustained_throughput": (
            "Sustained throughput / thermal soak",
            "How well each model holds throughput as it gets pushed to thermal limits.",
        ),
        "long_context_retrieval": (
            "Long-context retrieval (needle-in-a-haystack)",
            "Can each model still find a planted fact as the context grows from 4k to 131k tokens?",
        ),
    }


def _render_suite_block_html(suite: dict[str, Any]) -> str:
    """Render one suite as a card with per-suite-specific dashboard.

    Dispatches to a suite-specific renderer based on the canonical name
    (``openclaw_speed`` / ``hallucination_grounding`` / etc.). Unknown
    suite names fall back to the original generic table layout so a
    future suite that hasn't been explicitly themed still renders.
    """
    name = str(suite.get("suite") or "")
    models = suite.get("models") or []
    canonical = _canonical_suite_key(name)
    titles = _suite_titles_and_subtitles()
    title, subtitle = titles.get(canonical or "", (name, ""))
    parts: list[str] = []
    parts.append('<section class="card accent">')
    parts.append(
        f"<h2>{_esc(title)} "
        f'<span class="badge accent">{_esc(name)}</span></h2>'
    )
    if subtitle:
        parts.append(f'<p class="commentary">{_esc(subtitle)}</p>')
    commentary = _suite_commentary(canonical or name, suite) if models else ""
    if commentary:
        parts.append(f'<p class="commentary">{_esc(commentary)}</p>')
    if not models:
        parts.append('<p class="meta">No models scored in this suite.</p>')
        parts.append("</section>")
        return "".join(parts)

    if canonical == "openclaw_speed":
        parts.append(_render_suite_openclaw_speed(models))
    elif canonical == "hallucination_grounding":
        parts.append(_render_suite_reliability(models, kind="grounding"))
    elif canonical == "practical_structured_output":
        parts.append(_render_suite_reliability(models, kind="structured"))
    elif canonical == "code_generation":
        parts.append(_render_suite_code_generation(models))
    elif canonical == "sustained_throughput":
        parts.append(_render_suite_sustained_throughput(models))
    elif canonical == "long_context_retrieval":
        parts.append(_render_suite_long_context(models))
    else:
        parts.append(_render_suite_generic(models))

    # Per-model results table (color-coded pass-rate cells) is shared
    # across every suite — keeps the layout consistent.
    parts.append(_render_suite_table(models))
    parts.append("</section>")
    return "".join(parts)


def _render_suite_table(models: list[dict[str, Any]]) -> str:
    """Shared per-model results table with color-coded pass-rate cells."""
    parts: list[str] = ['<table aria-label="Per-model results">']
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
            f"{_cell_pct_html(model.get('pass_rate'))}"
            f"<td class='num'>{int(model.get('runs') or 0)}</td>"
            f"<td class='num'>{_fmt_num(ttft) if ttft is not None else '—'}</td>"
            f"<td class='num'>{_fmt_num(tok_s) if tok_s is not None else '—'}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def _render_suite_generic(models: list[dict[str, Any]]) -> str:
    """Fallback for unknown suites: a single pass-rate bar chart."""
    bar_rows = [
        (str(m.get("model", "?")), float(m.get("pass_rate") or 0.0)) for m in models
    ]
    return (
        '<div class="suite-grid">'
        '<div class="panel"><h4>Pass rate</h4>'
        + _svg_bars(bar_rows, max_value=1.0)
        + "</div></div>"
    )


def _render_suite_openclaw_speed(models: list[dict[str, Any]]) -> str:
    """Speed probe: pass-rate · TTFT (lower=better, inverted) · tok/s."""
    pass_rows = [
        (str(m.get("model", "?")), float(m.get("pass_rate") or 0.0)) for m in models
    ]
    ttft_rows = [
        (str(m.get("model", "?")), float(m.get("avg_ttft_ms") or 0.0))
        for m in models
        if m.get("avg_ttft_ms") is not None
    ]
    tps_rows = [
        (str(m.get("model", "?")), float(m.get("avg_tokens_per_s") or 0.0))
        for m in models
        if m.get("avg_tokens_per_s") is not None
    ]
    return (
        '<div class="suite-grid">'
        '<div class="panel"><h4>Pass rate</h4>'
        + _svg_bars(pass_rows, max_value=1.0)
        + "</div>"
        '<div class="panel"><h4>TTFT (ms · lower is better)</h4>'
        + _svg_bars(ttft_rows, value_format="num", invert_color=True)
        + "</div>"
        '<div class="panel"><h4>Decode throughput (tok/s)</h4>'
        + _svg_bars(tps_rows, value_format="num", accent="#16a34a")
        + "</div>"
        "</div>"
    )


def _render_suite_reliability(
    models: list[dict[str, Any]], *, kind: str
) -> str:
    """Reliability suites (grounding / structured): 3-up + per-task strip.

    Panels: pass-rate · TTFT (inverted) · per-task pass-fail strip
    loaded lazily from the run-dir's ``results.jsonl``. The strip
    panel is omitted entirely when no run-dir is wired through (e.g.
    in unit-test fixtures), avoiding empty UI.
    """
    pass_rows = [
        (str(m.get("model", "?")), float(m.get("pass_rate") or 0.0)) for m in models
    ]
    ttft_rows = [
        (str(m.get("model", "?")), float(m.get("avg_ttft_ms") or 0.0))
        for m in models
        if m.get("avg_ttft_ms") is not None
    ]
    panels: list[str] = []
    panels.append(
        '<div class="panel"><h4>Pass rate</h4>'
        + _svg_bars(pass_rows, max_value=1.0)
        + "</div>"
    )
    panels.append(
        '<div class="panel"><h4>TTFT (ms · lower is better)</h4>'
        + _svg_bars(ttft_rows, value_format="num", invert_color=True)
        + "</div>"
    )
    # Per-task pass-fail strip (lazy load from results.jsonl).
    pf_inputs: list[tuple[str, list[bool | None]]] = []
    for model in models:
        run_dir = model.get("run_dir") or model.get("extra", {}).get("run_dir")
        if not run_dir:
            continue
        rows = _load_results_rows(run_dir)
        pf = _per_task_pass_fail(rows, str(model.get("model") or ""))
        if pf:
            pf_inputs.append((str(model.get("model") or "?"), pf))
    if pf_inputs:
        kind_label = "grounded?" if kind == "grounding" else "valid JSON?"
        panels.append(
            '<div class="panel" style="grid-column: 1 / -1;">'
            f"<h4>Per-task results ({_esc(kind_label)} green = pass, red = fail)</h4>"
            + _pass_fail_strip_html(pf_inputs)
            + "</div>"
        )
    return '<div class="suite-grid">' + "".join(panels) + "</div>"


def _render_suite_code_generation(models: list[dict[str, Any]]) -> str:
    """code_generation: aggregate pass@1 + per-benchmark + sandbox status.

    Panels: aggregate pass@1 (color-graded), per-benchmark stacked
    bars (HumanEval / MBPP / …), sandbox-status breakdown loaded from
    ``results.jsonl`` (passed / failed / timeout / oom / compile_error
    / runtime_error). Each panel degrades gracefully when its data is
    missing.
    """
    pass_rows = [
        (str(m.get("model", "?")), float(m.get("pass_rate") or 0.0)) for m in models
    ]
    panels: list[str] = []
    panels.append(
        '<div class="panel"><h4>Aggregate pass@1</h4>'
        + _svg_bars(pass_rows, max_value=1.0)
        + "</div>"
    )
    # Per-benchmark stacked breakdown: each model row split across
    # benchmark buckets where the segment value is the *passing*
    # task count. Width per row is normalised so each row fills the
    # same visual width — what shifts is the colour split between
    # benchmarks.
    pb_rows: list[tuple[str, list[tuple[str, float]]]] = []
    for model in models:
        bms = model.get("benchmarks") or model.get("extra", {}).get("benchmarks") or []
        segments: list[tuple[str, float]] = []
        for bm in bms:
            tasks = int(bm.get("tasks") or 0)
            pa1 = float(bm.get("pass_at_1") or 0.0)
            segments.append((str(bm.get("benchmark") or "?"), pa1 * tasks))
        if segments:
            pb_rows.append((str(model.get("model") or "?"), segments))
    if pb_rows:
        panels.append(
            '<div class="panel"><h4>Per-benchmark passes</h4>'
            + _svg_stacked_bars(pb_rows)
            + "</div>"
        )
    # Sandbox-status breakdown.
    status_rows: list[tuple[str, list[tuple[str, float]]]] = []
    for model in models:
        run_dir = model.get("run_dir") or model.get("extra", {}).get("run_dir")
        if not run_dir:
            continue
        rows = _load_results_rows(run_dir)
        status_segs = _code_gen_status_breakdown(rows, str(model.get("model") or ""))
        if status_segs:
            status_rows.append((str(model.get("model") or "?"), status_segs))
    if status_rows:
        panels.append(
            '<div class="panel" style="grid-column: 1 / -1;"><h4>Sandbox-status breakdown</h4>'
            + _svg_stacked_bars(
                status_rows,
                palette=(
                    "#16a34a",  # passed
                    "#dc2626",  # failed
                    "#d97706",  # timeout
                    "#7f1d1d",  # oom
                    "#a855f7",  # compile_error
                    "#0ea5e9",  # runtime_error
                ),
            )
            + "</div>"
        )
    return '<div class="suite-grid">' + "".join(panels) + "</div>"


def _render_suite_sustained_throughput(models: list[dict[str, Any]]) -> str:
    """Sustained throughput: 3-up KPIs + wide tps-over-time line chart.

    Top row (3-up): initial vs sustained dual-bars · throttle ratio
    gauge · peak-temp thermometer (per-model; we render one
    thermometer per model in a small flex strip). Below: a wide line
    chart with tps-over-time (from ``windows[]``) and an optional
    overlay of GPU temperature loaded from ``telemetry-<model>.jsonl``.
    """

    def _val(model: dict[str, Any], key: str) -> Any:
        return model.get(key) if model.get(key) is not None else (model.get("extra", {}) or {}).get(key)

    dual_rows: list[tuple[str, float, float]] = []
    for model in models:
        init = _val(model, "initial_tokens_per_s")
        sus = _val(model, "sustained_tokens_per_s")
        if init is not None or sus is not None:
            dual_rows.append(
                (str(model.get("model") or "?"), float(init or 0.0), float(sus or 0.0))
            )
    panels: list[str] = []
    if dual_rows:
        panels.append(
            '<div class="panel"><h4>Initial vs sustained throughput (tok/s)</h4>'
            + _svg_dual_bars(
                dual_rows,
                label_a="Initial",
                label_b="Sustained",
                accent_a="#06b6d4",
                accent_b="#6366f1",
            )
            + "</div>"
        )
    # Throttle ratio gauges, one per model.
    gauges: list[str] = []
    for model in models:
        ratio = _val(model, "throttle_ratio")
        if ratio is None:
            continue
        gauges.append(
            f'<div style="flex:1 1 140px;text-align:center;min-width:140px;">'
            f'<div style="font-size:12px;color:var(--fg-muted);font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">'
            f"<code>{_esc(model.get('model') or '?')}</code></div>"
            + _svg_gauge(
                float(ratio),
                max_value=1.0,
                label="sustained / initial",
                suffix="%",
            )
            + "</div>"
        )
    if gauges:
        panels.append(
            '<div class="panel"><h4>Throttle ratio (higher = better)</h4>'
            f'<div style="display:flex;flex-wrap:wrap;gap:14px;">{"".join(gauges)}</div>'
            "</div>"
        )
    # Peak temp thermometers, one per model.
    thermos: list[str] = []
    for model in models:
        temp = _val(model, "peak_temp_c")
        if temp is None:
            continue
        thermos.append(
            f'<div style="flex:1 1 100px;text-align:center;min-width:90px;">'
            f'<div style="font-size:11px;color:var(--fg-muted);font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;">'
            f"<code>{_esc(model.get('model') or '?')}</code></div>"
            + _svg_thermometer(float(temp), label="peak")
            + "</div>"
        )
    if thermos:
        panels.append(
            '<div class="panel"><h4>Peak GPU temperature</h4>'
            f'<div style="display:flex;flex-wrap:wrap;gap:14px;">{"".join(thermos)}</div>'
            "</div>"
        )

    # Wide tps-over-time line chart with optional GPU temp overlay.
    series: list[tuple[str, list[tuple[float, float]]]] = []
    secondary: list[tuple[str, list[tuple[float, float]]]] = []
    for model in models:
        windows = model.get("windows") or (model.get("extra", {}) or {}).get("windows") or []
        if windows:
            points = [
                (float(w.get("start_s", 0.0)), float(w.get("tokens_per_s") or 0.0))
                for w in windows
            ]
            series.append((f"{model.get('model') or '?'} tps", points))
        # Temp overlay (downsampled telemetry).
        run_dir = model.get("run_dir") or (model.get("extra", {}) or {}).get("run_dir")
        if run_dir:
            samples = _load_telemetry_samples(run_dir, str(model.get("model") or ""))
            temp_pts = [
                (float(s.get("timestamp_s") or 0.0), float(s.get("gpu_temp_c") or 0.0))
                for s in samples
                if s.get("gpu_temp_c") is not None
            ]
            if temp_pts:
                secondary.append((f"{model.get('model') or '?'} GPU °C", temp_pts))
    line_html = _svg_line_chart(
        series,
        secondary=secondary,
        height=160,
        x_label=" s",
        y_label="tok/s",
    )

    out = '<div class="suite-grid">' + "".join(panels) + "</div>"
    if line_html:
        out += (
            '<div class="suite-wide card"><h3>Throughput over time</h3>'
            f"{line_html}"
            "</div>"
        )
    return out


def _render_suite_long_context(models: list[dict[str, Any]]) -> str:
    """Long-context NIAH: per-model length×depth heatmap + prefill/memory curves.

    Heatmaps show retrieval pass-rate by context length (rows) and needle
    depth (columns); N/A tiles mark lengths a model can't load. Below: a
    prefill-throughput-vs-length line chart and (only when Ollama
    ``/api/ps`` yielded memory) a resident-memory-vs-length chart.
    """

    def _val(model: dict[str, Any], key: str) -> Any:
        return model.get(key) if model.get(key) is not None else (model.get("extra", {}) or {}).get(key)

    # KPI strip — first-failure length per model.
    kpis: list[str] = []
    for model in models:
        ffl = _val(model, "first_failure_length")
        name = str(model.get("model") or "?")
        if ffl:
            value = f"{int(ffl):,}"
            sub = "first-failure length (tokens)"
        else:
            value = "✓ all"
            sub = "held across the grid"
        kpis.append(
            '<div style="flex:1 1 150px;min-width:140px;text-align:center;">'
            '<div style="font-size:12px;color:var(--fg-muted);font-weight:600;'
            'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;">'
            f"<code>{_esc(name)}</code></div>"
            f'<div style="font-size:22px;font-weight:700;">{_esc(value)}</div>'
            f'<div style="font-size:11px;color:var(--fg-muted);">{_esc(sub)}</div>'
            "</div>"
        )

    panels: list[str] = []
    if kpis:
        panels.append(
            '<div class="panel"><h4>Where retrieval breaks down</h4>'
            f'<div style="display:flex;flex-wrap:wrap;gap:14px;">{"".join(kpis)}</div>'
            "</div>"
        )

    # One heatmap per model.
    for model in models:
        cells = _val(model, "cells") or []
        if not cells:
            continue
        lengths = sorted({int(c["context_length"]) for c in cells}, reverse=True)
        depths = sorted({int(c["depth_pct"]) for c in cells})
        row_labels = [f"{length:,}" for length in lengths]
        col_labels = [f"{depth}%" for depth in depths]
        values: dict[tuple[str, str], float | None] = {}
        for c in cells:
            rl = f"{int(c['context_length']):,}"
            cl = f"{int(c['depth_pct'])}%"
            values[(rl, cl)] = c.get("pass_rate")
        heatmap = _svg_heatmap(row_labels, col_labels, values)
        if heatmap:
            panels.append(
                '<div class="panel"><h4>Retrieval pass-rate · '
                f"<code>{_esc(str(model.get('model') or '?'))}</code></h4>"
                '<p class="meta">rows = context length (tokens) · columns = needle depth</p>'
                f"{heatmap}</div>"
            )

    # Prefill throughput / memory growth vs context length (across depths).
    prefill_series: list[tuple[str, list[tuple[float, float]]]] = []
    memory_series: list[tuple[str, list[tuple[float, float]]]] = []
    for model in models:
        cells = _val(model, "cells") or []
        by_len_tps: dict[int, list[float]] = {}
        by_len_vram: dict[int, list[float]] = {}
        for c in cells:
            length = int(c["context_length"])
            if c.get("avg_prefill_tps") is not None:
                by_len_tps.setdefault(length, []).append(float(c["avg_prefill_tps"]))
            if c.get("peak_vram_mb") is not None:
                by_len_vram.setdefault(length, []).append(float(c["peak_vram_mb"]))
        name = str(model.get("model") or "?")
        if by_len_tps:
            pts = [(float(length), sum(v) / len(v)) for length, v in sorted(by_len_tps.items())]
            prefill_series.append((f"{name} prefill", pts))
        if by_len_vram:
            vram_pts = [(float(length), max(v)) for length, v in sorted(by_len_vram.items())]
            memory_series.append((f"{name} VRAM", vram_pts))

    # Pass-rate by needle category (rows = model, cols = needle type).
    # Needle type drives retrieval as much as position: an alphanumeric
    # code is far harder to copy than a city name, even at the same depth.
    cat_labels = {
        "alphanumeric_code": "code",
        "named_entity_person": "person",
        "date": "date",
        "location": "location",
        "quantity": "quantity",
    }
    cat_order: list[str] = []
    cat_values: dict[tuple[str, str], float | None] = {}
    cat_row_labels: list[str] = []
    for model in models:
        categories = _val(model, "categories") or []
        if not categories:
            continue
        name = str(model.get("model") or "?")
        cat_row_labels.append(name)
        for cat in categories:
            cid = str(cat.get("category") or "unknown")
            label = cat_labels.get(cid, cid)
            if label not in cat_order:
                cat_order.append(label)
            cat_values[(name, label)] = cat.get("pass_rate")
    category_html = ""
    if cat_row_labels and cat_order:
        # Wider gutter than the default so long model names (e.g. hf.co/…)
        # fit without clipping.
        gutter = min(360, max(120, 8 * max(len(n) for n in cat_row_labels)))
        category_html = _svg_heatmap(
            cat_row_labels, cat_order, cat_values, cell_w=82, left_pad=gutter
        )

    out = '<div class="suite-grid">' + "".join(panels) + "</div>"
    if category_html:
        out += (
            '<div class="suite-wide card"><h3>Retrieval pass-rate by needle type</h3>'
            '<p class="meta">rows = model · columns = needle category · '
            "alphanumeric codes are typically the hardest to retrieve verbatim</p>"
            f"{category_html}</div>"
        )
    prefill_html = _svg_line_chart(prefill_series, height=160, x_label=" tok", y_label="tok/s")
    if prefill_html:
        out += (
            '<div class="suite-wide card"><h3>Prefill throughput vs context length</h3>'
            f"{prefill_html}</div>"
        )
    memory_html = _svg_line_chart(memory_series, height=160, x_label=" tok", y_label="MB")
    if memory_html:
        out += (
            '<div class="suite-wide card"><h3>Resident memory vs context length '
            '<span class="badge">Ollama /api/ps</span></h3>'
            f"{memory_html}</div>"
        )
    return out


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
    """Render a custom (BYOT / quick) ``summary.json`` payload as HTML.

    Counterpart to ``custom_suites.render_custom_summary_markdown`` but
    sharing the same marketing-grade hero / stat-tile / card chrome
    as the canonical report. Hero uses a slightly different colour
    stop (cyan-leaning) so it's visually obvious at a glance that
    this is a custom run, not a canonical bundle.

    Layout: hero (suite name, version, mode, backend, task / model
    counts) → per-model telemetry card + decode-tps bar chart → one
    ``<details>`` block per task with a small TTFT comparison chart,
    output-length chart, and side-by-side replies; errored cells in
    red.
    """
    suite_name = str(summary.get("suite") or "")
    title = f"Custom suite: {suite_name}"
    per_model = list(summary.get("per_model") or [])
    rows = list(summary.get("rows") or [])
    has_any_error = any(b.get("tasks_errored") or 0 for b in per_model)

    body: list[str] = []

    # ------------------------------------------------------------- #
    # Hero — cyan-tinted gradient to distinguish from canonical
    # ------------------------------------------------------------- #
    body.append(
        '<section class="hero" style="background:'
        'linear-gradient(135deg, #060d18 0%, #071520 40%, #061a1a 100%);'
        'border-bottom: 1px solid rgba(34,211,238,0.3);">'
    )
    body.append('<div class="hero-brand"><span class="nv">NVIDIA</span> | OpenClaw</div>')
    body.append('<div class="hero-inner">')
    body.append('<div class="hero-main">')
    body.append('<div class="hero-breadcrumb">DGX SPARK BENCHMARK — CUSTOM RUN</div>')
    body.append(f"<h1>{_esc(title)}</h1>")
    if summary.get("description"):
        body.append(f'<p class="tagline">{_esc(summary["description"])}</p>')
    else:
        body.append(
            '<p class="tagline">Bring-Your-Own-Test run — pass-through mode, '
            "no scoring. Side-by-side answers across every model in the "
            "lineup.</p>"
        )
    body.append('<div class="hero-meta">')
    body.append(f'<span><strong>Version:</strong> <code>{_esc(summary.get("suite_version", "—"))}</code></span>')
    body.append(f'<span><strong>Mode:</strong> <code>{_esc(summary.get("mode", "—"))}</code></span>')
    body.append(f'<span><strong>Backend:</strong> <code>{_esc(summary.get("backend", "—"))}</code></span>')
    body.append(f"<span><strong>Tasks:</strong> {int(summary.get('task_count') or 0)}</span>")
    body.append(f"<span><strong>Models:</strong> {len(per_model)}</span>")
    body.append(f"<span><strong>Generated:</strong> {_esc(_now_utc_iso())}</span>")
    body.append("</div>")   # /.hero-meta
    body.append("</div>")   # /.hero-main
    body.append("</div>")   # /.hero-inner
    body.append("</section>")

    # ------------------------------------------------------------- #
    # Stat tiles (custom-flavour: tasks completed/errored, fastest)
    # ------------------------------------------------------------- #
    body.append(_custom_stat_tiles_html(summary, per_model, rows))

    body.append('<div class="container">')

    # ------------------------------------------------------------- #
    # Per-model telemetry card
    # ------------------------------------------------------------- #
    if not per_model:
        body.append('<p class="meta">No models recorded in this run.</p>')
    else:
        body.append('<section class="card accent">')
        body.append("<h2>Per-model telemetry</h2>")
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
            badge = (
                f' <span class="badge bad">{err_count} err</span>' if err_count else ""
            )
            body.append(
                "<tr>"
                f"<td><code>{_esc(bucket.get('model'))}</code>{badge}</td>"
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

        # 2-up: mean tps + mean TTFT (inverted color).
        tps_rows = [
            (str(b.get("model", "?")), float(b["mean_decode_tps"]))
            for b in per_model
            if b.get("mean_decode_tps") is not None
        ]
        ttft_rows = [
            (str(b.get("model", "?")), float(b["mean_ttft_ms"]))
            for b in per_model
            if b.get("mean_ttft_ms") is not None
        ]
        body.append('<div class="suite-grid">')
        if tps_rows:
            body.append(
                '<div class="panel"><h4>Mean decode tps</h4>'
                + _svg_bars(tps_rows, value_format="num", accent="#16a34a")
                + "</div>"
            )
        if ttft_rows:
            body.append(
                '<div class="panel"><h4>Mean TTFT (ms · lower is better)</h4>'
                + _svg_bars(ttft_rows, value_format="num", invert_color=True)
                + "</div>"
            )
        body.append("</div>")
        body.append("</section>")

    # ------------------------------------------------------------- #
    # Per-task collapsible blocks
    # ------------------------------------------------------------- #
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
            "<p class='meta'>One collapsible block per task. Each block shows a "
            "TTFT and output-length comparison across models, then every "
            "model's full reply. Click a task header to expand.</p>"
        )
        if has_any_error:
            body.append(
                '<p class="meta"><span class="badge bad">red squares</span> in the '
                "header strip mark tasks where at least one model errored.</p>"
            )
        for task_id in task_order:
            body.append(_render_custom_task_block_html(task_id, by_task[task_id]))

    body.append("</div>")  # /.container
    body.append("<footer>")
    body.append(
        "Generated by <code>spark-benchmark</code> · "
        "self-contained HTML — no scripts, no CDN, no external assets."
    )
    body.append("</footer>")

    return _wrap_document(title=title, body_html="".join(body))


def _custom_stat_tiles_html(
    summary: dict[str, Any],
    per_model: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    """Stat-tile strip for custom / quick runs.

    Tiles: total completed pairs, errored pairs, fastest mean-tps
    model (highlighted in green), slowest TTFT (highlighted in
    amber/red). Falls back to "—" when there's no data.
    """
    total_completed = sum(int(b.get("tasks_completed") or 0) for b in per_model)
    total_errored = sum(int(b.get("tasks_errored") or 0) for b in per_model)
    fastest_tps: tuple[str, float] | None = None
    for bucket in per_model:
        tps = bucket.get("mean_decode_tps")
        if tps is None:
            continue
        if fastest_tps is None or float(tps) > fastest_tps[1]:
            fastest_tps = (str(bucket.get("model") or "?"), float(tps))
    lowest_ttft: tuple[str, float] | None = None
    for bucket in per_model:
        ttft = bucket.get("mean_ttft_ms")
        if ttft is None:
            continue
        if lowest_ttft is None or float(ttft) < lowest_ttft[1]:
            lowest_ttft = (str(bucket.get("model") or "?"), float(ttft))
    parts: list[str] = ['<div class="stat-tiles">']
    parts.append(
        f'<div class="stat-tile"><div class="label">Completed</div>'
        f'<div class="value">{total_completed}</div>'
        f'<div class="sub">model × task pairs</div></div>'
    )
    err_color = "var(--bad)" if total_errored else "var(--fg)"
    parts.append(
        f'<div class="stat-tile"><div class="label">Errored</div>'
        f'<div class="value" style="color:{err_color};">{total_errored}</div>'
        f'<div class="sub">errors during generation</div></div>'
    )
    if fastest_tps is not None:
        parts.append(
            f'<div class="stat-tile"><div class="label">Fastest decode</div>'
            f'<div class="value" style="color:var(--good);">{_fmt_num(fastest_tps[1])}</div>'
            f'<div class="sub"><code>{_esc(fastest_tps[0])}</code> tok/s</div></div>'
        )
    if lowest_ttft is not None:
        parts.append(
            f'<div class="stat-tile"><div class="label">Lowest TTFT</div>'
            f'<div class="value">{_fmt_num(lowest_ttft[1])}<span style="font-size:14px;color:var(--fg-muted);"> ms</span></div>'
            f'<div class="sub"><code>{_esc(lowest_ttft[0])}</code></div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _render_custom_task_block_html(task_id: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    prompt = str(rows[0].get("prompt") or "")
    has_error = any(row.get("error") for row in rows)
    badge = ' <span class="badge bad">contains errors</span>' if has_error else ""

    # Extract per-model TTFT and output-length for the at-a-glance charts.
    ttft_rows: list[tuple[str, float]] = []
    len_rows: list[tuple[str, float]] = []
    error_strip: list[bool | None] = []
    for row in rows:
        model = str(row.get("model") or "?")
        if row.get("error"):
            error_strip.append(False)
            continue
        gen = row.get("generation") or {}
        metrics = (gen.get("metrics") or {}) if isinstance(gen, dict) else {}
        ttft = metrics.get("ttft_ms")
        decode_tokens = metrics.get("decode_tokens") or 0
        if isinstance(ttft, (int, float)):
            ttft_rows.append((model, float(ttft)))
        if decode_tokens:
            len_rows.append((model, float(decode_tokens)))
        error_strip.append(True)

    # Header strip: small dot per model (green pass / red errored). Lets
    # users scan a long custom suite for "which task had failures?" at a
    # glance without expanding every block.
    strip_html = ""
    if error_strip:
        cells = "".join(
            f'<span class="cell {"pass" if p is True else "fail" if p is False else "na"}" '
            f'style="width:8px;height:8px;border-radius:2px;"></span>'
            for p in error_strip
        )
        strip_html = (
            f'<span class="pf-strip" style="display:inline-flex;gap:2px;'
            f'margin-left:auto;align-items:center;">{cells}</span>'
        )

    parts: list[str] = []
    parts.append("<details>")
    parts.append(
        f"<summary>Task <code>{_esc(task_id)}</code>{badge}{strip_html}</summary>"
    )
    parts.append('<div class="task-block">')
    parts.append('<div class="prompt">')
    parts.append("<strong>Prompt</strong>")
    parts.append(f"<pre>{_esc(prompt) or '(empty)'}</pre>")
    parts.append("</div>")

    # 2-up mini-charts at the top of each task.
    if ttft_rows or len_rows:
        parts.append('<div class="suite-grid" style="margin-top:8px;">')
        if ttft_rows:
            parts.append(
                '<div class="panel"><h4>TTFT (ms · lower is better)</h4>'
                + _svg_bars(ttft_rows, value_format="num", invert_color=True)
                + "</div>"
            )
        if len_rows:
            parts.append(
                '<div class="panel"><h4>Output length (decode tokens)</h4>'
                + _svg_bars(len_rows, value_format="num", accent="#8b5cf6")
                + "</div>"
            )
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
