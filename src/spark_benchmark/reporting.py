from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def _passes_total_from_model(model: dict[str, Any]) -> tuple[int, int]:
    """Derive (passes, total) from a per-model summary entry across schemas.

    Reliability / speed suites store flat ``passes`` and ``total`` per model.
    The code_generation suite stores nested ``benchmarks`` with ``pass_at_1`` and
    ``tasks`` per benchmark (HumanEval, MBPP, ...). Normalise both shapes.
    """
    if "passes" in model and "total" in model:
        return int(model["passes"]), int(model["total"])
    passes = 0
    total = 0
    for bm in model.get("benchmarks") or []:
        tasks = int(bm.get("tasks") or 0)
        pa1 = float(bm.get("pass_at_1") or 0.0)
        passes += round(pa1 * tasks)
        total += tasks
    return passes, total


def aggregate_runs(runs_root: Path) -> dict[str, Any]:
    run_dirs = collect_run_dirs(runs_root)
    runs: list[dict[str, Any]] = []
    suite_model_totals: dict[str, dict[str, dict[str, Any]]] = {}

    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = _load_json(manifest_path)
        summary_path = run_dir / "summary.json"
        summary = _load_json(summary_path) if summary_path.exists() else None
        rows = _load_jsonl(run_dir / "results.jsonl")

        run_item = {
            "run_id": run_dir.name,
            "experiment": manifest["experiment"]["name"],
            "backend": manifest["backend"]["name"],
            "platform": manifest["platform"]["display_name"],
            "suite": summary["suite"] if summary else None,
            "suite_version": summary["suite_version"] if summary else None,
            "row_count": len(rows),
            "model_count": len(summary["models"]) if summary else 0,
            "models": summary["models"] if summary else [],
        }
        runs.append(run_item)

        if not summary:
            continue

        suite_bucket = suite_model_totals.setdefault(summary["suite"], {})
        metrics_by_model: dict[str, dict[str, float]] = {}
        for row in rows:
            model_name = row.get("model")
            if not model_name:
                continue
            # Reliability / speed rows expose metrics at row.generation.metrics.
            # Code-generation rows wrap multiple attempts in row.samples[*].generation.metrics,
            # so collect from whichever shape the suite produced.
            metrics_list: list[dict[str, Any]] = []
            row_generation = row.get("generation") or {}
            row_metrics = row_generation.get("metrics") if isinstance(row_generation, dict) else None
            if row_metrics:
                metrics_list.append(row_metrics)
            for sample in row.get("samples") or []:
                sample_gen = (sample or {}).get("generation") or {}
                sample_metrics = sample_gen.get("metrics") if isinstance(sample_gen, dict) else None
                if sample_metrics:
                    metrics_list.append(sample_metrics)
            if not metrics_list:
                continue
            metric_bucket = metrics_by_model.setdefault(
                model_name,
                {
                    "samples": 0.0,
                    "ttft_ms_sum": 0.0,
                    "decode_time_s_sum": 0.0,
                    "decode_tokens_sum": 0.0,
                },
            )
            for metrics in metrics_list:
                metric_bucket["samples"] += 1
                metric_bucket["ttft_ms_sum"] += float(metrics.get("ttft_ms") or 0.0)
                metric_bucket["decode_time_s_sum"] += float(metrics.get("decode_time_s") or 0.0)
                metric_bucket["decode_tokens_sum"] += float(metrics.get("decode_tokens") or 0.0)
        for model in summary["models"]:
            model_bucket = suite_bucket.setdefault(
                model["model"],
                {
                    "model": model["model"],
                    "passes": 0,
                    "total": 0,
                    "runs": 0,
                    "metric_samples": 0.0,
                    "ttft_ms_sum": 0.0,
                    "decode_time_s_sum": 0.0,
                    "decode_tokens_sum": 0.0,
                    "extra": {},
                },
            )
            passes, total = _passes_total_from_model(model)
            model_bucket["passes"] += passes
            model_bucket["total"] += total
            model_bucket["runs"] += 1
            # Forward any suite-specific scalars (e.g. sustained_throughput's
            # initial/sustained/throttle_ratio) so render_cli_benchmark_summary
            # can pick them up downstream. Last write wins, which is fine
            # because these are per-model facts, not summable across runs.
            for key in (
                "initial_tokens_per_s",
                "sustained_tokens_per_s",
                "peak_tokens_per_s",
                "throttle_ratio",
                "time_to_throttle_s",
                "avg_power_w",
                "peak_temp_c",
                "energy_j_per_token",
                "throttle_reasons_observed",
                "telemetry_source",
                # Per-window throughput series (sustained_throughput) — used
                # by the HTML renderer to draw a tps-over-time line chart.
                # Discarded by the markdown / CLI summaries since they only
                # surface scalar KPIs.
                "windows",
                # Per-benchmark breakdown (code_generation) — list of
                # {benchmark, tasks, pass_at_1, pass_at_k, failed_task_ids}.
                # The HTML renderer turns this into a stacked-bar chart;
                # the markdown report prints the pass_at_1 directly via
                # the row's existing benchmarks field.
                "benchmarks",
                # Per-model run directory inside this suite's run bundle.
                # The HTML renderer uses it to lazy-load telemetry-*.jsonl
                # and re-read results.jsonl when it needs per-task data
                # (pass/fail strips, sandbox-failure breakdown). Optional
                # — falls back gracefully when the file is absent.
                "run_dir",
                # Long-context (needle-in-a-haystack) per-model fields. The
                # HTML renderer turns `cells` into a length×depth heatmap plus
                # prefill-throughput and memory-growth charts; the scalar
                # `first_failure_length` headlines where retrieval breaks down.
                "cells",
                "first_failure_length",
                "skipped",
                "errors",
                # Per-needle-category retrieval breakdown (long-context):
                # list of {category, passes, n, pass_rate}. The HTML renderer
                # draws a category × model pass-rate panel since needle type
                # (e.g. alphanumeric codes vs. names) drives retrieval as much
                # as position does.
                "categories",
            ):
                if key == "run_dir":
                    model_bucket["extra"]["run_dir"] = str(run_dir)
                    continue
                if model.get(key) is not None:
                    model_bucket["extra"][key] = model[key]
            row_metrics = metrics_by_model.get(model["model"])
            if row_metrics:
                model_bucket["metric_samples"] += row_metrics["samples"]
                model_bucket["ttft_ms_sum"] += row_metrics["ttft_ms_sum"]
                model_bucket["decode_time_s_sum"] += row_metrics["decode_time_s_sum"]
                model_bucket["decode_tokens_sum"] += row_metrics["decode_tokens_sum"]

    suites: list[dict[str, Any]] = []
    for suite_name, model_buckets in sorted(suite_model_totals.items()):
        models = []
        for model in model_buckets.values():
            total = model["total"] or 1
            metric_samples = model["metric_samples"] or 0.0
            avg_ttft_ms = round(model["ttft_ms_sum"] / metric_samples, 2) if metric_samples else None
            avg_tokens_per_s = None
            if model["decode_time_s_sum"] > 0:
                avg_tokens_per_s = round(model["decode_tokens_sum"] / model["decode_time_s_sum"], 2)
            extra = model.pop("extra", {})
            models.append(
                {
                    **model,
                    "pass_rate": round(model["passes"] / total, 4),
                    "avg_ttft_ms": avg_ttft_ms,
                    "avg_tokens_per_s": avg_tokens_per_s,
                    **extra,
                }
            )
        models.sort(key=lambda item: (-item["pass_rate"], item["model"]))
        suites.append({"suite": suite_name, "models": models})

    return {
        "runs_root": str(runs_root),
        "total_runs": len(runs),
        "suites": suites,
        "runs": runs[-10:],
    }


def render_markdown_report(aggregate: dict[str, Any]) -> str:
    lines = [
        "# spark-benchmark report",
        "",
        f"- runs root: {aggregate['runs_root']}",
        f"- total runs: {aggregate['total_runs']}",
        "",
    ]

    for suite in aggregate["suites"]:
        lines.extend(
            [
                f"## {suite['suite']}",
                "",
                "| model | passes | total | pass_rate | runs | avg_ttft_ms | avg_tok_s |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model in suite["models"]:
            ttft = "-" if model["avg_ttft_ms"] is None else model["avg_ttft_ms"]
            tok_s = "-" if model["avg_tokens_per_s"] is None else model["avg_tokens_per_s"]
            lines.append(
                f"| {model['model']} | {model['passes']} | {model['total']} | {model['pass_rate']:.2%} | {model['runs']} | {ttft} | {tok_s} |"
            )
        lines.append("")

    if aggregate["runs"]:
        lines.extend(
            [
                "## Recent Runs",
                "",
                "| run_id | experiment | backend | suite | rows |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for run in reversed(aggregate["runs"]):
            lines.append(
                f"| {run['run_id']} | {run['experiment']} | {run['backend']} | {run['suite'] or '-'} | {run['row_count']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def render_html_report(aggregate: dict[str, Any]) -> str:
    """Render the canonical aggregate as a polished standalone HTML page.

    Implementation lives in :mod:`spark_benchmark.reporting_html` to keep
    the styling / SVG plumbing separate from the data-aggregation logic
    in this module. The function signature is preserved so existing
    callers (CLI ``report`` command, ad-hoc scripts) keep working.
    """
    # Local import avoids a top-level cycle: reporting_html imports
    # narrative helpers (_overall_rank_rows, _suite_commentary,
    # _verdict_paragraph, _find_suite) from this module.
    from spark_benchmark.reporting_html import render_canonical_report_html

    return render_canonical_report_html(aggregate)


def write_report(output: Path, report_format: str, aggregate: dict[str, Any]) -> None:
    """Write the aggregate to ``output`` in the requested format.

    ``report_format`` accepts ``"markdown"`` (default), ``"html"``, or
    ``"both"``. For ``"both"``, the path's suffix is rewritten so the
    Markdown copy ends in ``.md`` and the HTML copy in ``.html`` —
    callers that want fine-grained control should call the renderers
    directly.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if report_format == "both":
        md_path = output.with_suffix(".md")
        html_path = output.with_suffix(".html")
        md_path.write_text(render_markdown_report(aggregate), encoding="utf-8")
        html_path.write_text(render_html_report(aggregate), encoding="utf-8")
        return
    body = render_markdown_report(aggregate)
    if report_format == "html":
        body = render_html_report(aggregate)
    output.write_text(body, encoding="utf-8")


def _find_suite(aggregate: dict[str, Any], suite_name: str) -> dict[str, Any] | None:
    for suite in aggregate.get("suites", []):
        current = str(suite.get("suite") or "")
        if current == suite_name or current.startswith(f"{suite_name}_"):
            return suite
    return None


def _models_by_name(suite: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not suite:
        return {}
    return {str(model["model"]): model for model in suite.get("models", [])}


def _overall_rank_rows(aggregate: dict[str, Any], model_names: list[str]) -> list[dict[str, Any]]:
    speed_models = _models_by_name(_find_suite(aggregate, "openclaw_speed"))
    grounding_models = _models_by_name(_find_suite(aggregate, "hallucination_grounding"))
    structured_models = _models_by_name(_find_suite(aggregate, "practical_structured_output"))

    ttfts = [speed_models[name]["avg_ttft_ms"] for name in model_names if name in speed_models and speed_models[name]["avg_ttft_ms"] is not None]
    toks = [speed_models[name]["avg_tokens_per_s"] for name in model_names if name in speed_models and speed_models[name]["avg_tokens_per_s"] is not None]
    min_ttft = min(ttfts) if ttfts else None
    max_ttft = max(ttfts) if ttfts else None
    min_tok = min(toks) if toks else None
    max_tok = max(toks) if toks else None

    rows: list[dict[str, Any]] = []
    for name in model_names:
        grounding = grounding_models.get(name, {})
        structured = structured_models.get(name, {})
        speed = speed_models.get(name, {})
        grounding_rate = float(grounding.get("pass_rate") or 0.0)
        structured_rate = float(structured.get("pass_rate") or 0.0)
        ttft = speed.get("avg_ttft_ms")
        tok_s = speed.get("avg_tokens_per_s")

        speed_score = 0.0
        if ttft is not None and min_ttft is not None and max_ttft is not None:
            if max_ttft == min_ttft:
                speed_score += 0.5
            else:
                speed_score += (max_ttft - float(ttft)) / (max_ttft - min_ttft) * 0.5
        if tok_s is not None and min_tok is not None and max_tok is not None:
            if max_tok == min_tok:
                speed_score += 0.5
            else:
                speed_score += (float(tok_s) - min_tok) / (max_tok - min_tok) * 0.5

        overall_score = grounding_rate * 0.6 + structured_rate * 0.25 + speed_score * 0.15
        rows.append(
            {
                "model": name,
                "grounding_rate": grounding_rate,
                "structured_rate": structured_rate,
                "avg_ttft_ms": ttft,
                "avg_tokens_per_s": tok_s,
                "speed_score": round(speed_score, 4),
                "overall_score": round(overall_score, 4),
            }
        )
    rows.sort(key=lambda item: (-item["overall_score"], -item["grounding_rate"], -(item["structured_rate"]), item["avg_ttft_ms"] or 10**9, -(item["avg_tokens_per_s"] or 0.0), item["model"]))
    return rows


def render_cli_benchmark_summary(
    *,
    request: str,
    selected_models: list[str],
    selected_suites: list[str],
    aggregate: dict[str, Any],
    report_path: Path,
) -> str:
    suite_titles = {
        "openclaw_speed": "OpenClaw-like speed probe",
        "hallucination_grounding": "Grounding / hallucination reliability",
        "practical_structured_output": "Structured output reliability",
        "code_generation": "Code generation (HumanEval)",
        "code_generation_v1": "Code generation (HumanEval)",
        "sustained_throughput": "Sustained throughput (thermal / decode soak)",
        "sustained_throughput_v1": "Sustained throughput (thermal / decode soak)",
        "long_context_retrieval": "Long-context retrieval (needle-in-a-haystack)",
        "long_context_retrieval_v1": "Long-context retrieval (needle-in-a-haystack)",
    }
    suite_map = {
        "openclaw_speed": _find_suite(aggregate, "openclaw_speed"),
        "hallucination_grounding": _find_suite(aggregate, "hallucination_grounding"),
        "practical_structured_output": _find_suite(aggregate, "practical_structured_output"),
        "code_generation": _find_suite(aggregate, "code_generation"),
        "sustained_throughput": _find_suite(aggregate, "sustained_throughput"),
        "long_context_retrieval": _find_suite(aggregate, "long_context_retrieval"),
    }
    lines = [
        "BENCHMARK SUMMARY",
        "",
        f"Request: {request}",
        f"Models tested: {', '.join(selected_models)}",
        f"Tests run: {', '.join(selected_suites)}",
        "",
    ]

    for suite_name in selected_suites:
        suite = suite_map.get(suite_name)
        if not suite:
            continue
        lines.append(f"{suite_titles.get(suite_name, suite_name)}:")
        for model in suite.get("models", []):
            pass_rate = f"{float(model['pass_rate']) * 100:.2f}%"
            ttft = "-" if model.get("avg_ttft_ms") is None else f"{model['avg_ttft_ms']} ms"
            tok_s = "-" if model.get("avg_tokens_per_s") is None else f"{model['avg_tokens_per_s']} tok/s"
            lines.append(
                f"- {model['model']}: pass rate {pass_rate} ({model['passes']}/{model['total']}), avg TTFT {ttft}, avg speed {tok_s}"
            )
        commentary = _suite_commentary(suite_name, suite)
        if commentary:
            lines.append(f"  → {commentary}")
        lines.append("")

    ranking = _overall_rank_rows(aggregate, selected_models)
    lines.append("Overall ranking:")
    for index, row in enumerate(ranking, start=1):
        reason_bits = []
        if row["grounding_rate"] >= 0.99:
            reason_bits.append("best grounding reliability")
        elif row["grounding_rate"] >= 0.8:
            reason_bits.append("strong grounding reliability")
        if row["structured_rate"] >= 0.99:
            reason_bits.append("perfect structured output")
        if index == 1 and row["avg_ttft_ms"] is not None:
            reason_bits.append("best overall weighted score")
        if row["avg_ttft_ms"] is not None and row["avg_tokens_per_s"] is not None:
            reason_bits.append(f"TTFT {row['avg_ttft_ms']} ms, {row['avg_tokens_per_s']} tok/s")
        reason = "; ".join(reason_bits) if reason_bits else "balanced result"
        lines.append(f"{index}. {row['model']} — {reason}")
    lines.append("")

    verdict = _verdict_paragraph(ranking, suite_map)
    if verdict:
        lines.append("Verdict:")
        lines.append(verdict)
        lines.append("")

    if ranking:
        winner = ranking[0]
        lines.append(
            f"Recommendation: {winner['model']} is the current default pick because it achieved the strongest combined result across reliability and speed."
        )
    lines.append(f"Full report saved to: {report_path}")
    return "\n".join(lines)


def _suite_commentary(suite_name: str, suite: dict[str, Any]) -> str:
    """One-line English narrative for a single suite's outcome."""
    models = suite.get("models") or []
    if not models:
        return ""
    sorted_models = sorted(models, key=lambda m: -float(m.get("pass_rate") or 0.0))
    best, worst = sorted_models[0], sorted_models[-1]
    best_rate = float(best.get("pass_rate") or 0.0) * 100
    worst_rate = float(worst.get("pass_rate") or 0.0) * 100

    if suite_name == "openclaw_speed":
        with_ttft = [m for m in models if m.get("avg_ttft_ms") is not None]
        if not with_ttft:
            return "All models completed the speed probe."
        fastest_ttft = min(with_ttft, key=lambda m: m["avg_ttft_ms"])
        fastest_tok = max(
            (m for m in with_ttft if m.get("avg_tokens_per_s")),
            key=lambda m: m["avg_tokens_per_s"],
            default=None,
        )
        bits = [f"All {len(models)} models completed the speed probe."]
        bits.append(
            f"Fastest first-token: {fastest_ttft['model']} ({fastest_ttft['avg_ttft_ms']} ms)."
        )
        if fastest_tok is not None:
            bits.append(
                f"Fastest decode: {fastest_tok['model']} ({fastest_tok['avg_tokens_per_s']} tok/s)."
            )
        return " ".join(bits)

    if suite_name == "hallucination_grounding":
        if best_rate >= 99 and worst_rate >= 99:
            return "Every model produced grounded answers — no hallucinations detected."
        if worst_rate < 50:
            return (
                f"Wide spread: {best['model']} led at {best_rate:.0f}% while "
                f"{worst['model']} hallucinated on more than half the tasks ({worst_rate:.0f}%)."
            )
        return (
            f"{best['model']} was most reliable at {best_rate:.0f}%, "
            f"{worst['model']} the weakest at {worst_rate:.0f}%."
        )

    if suite_name == "practical_structured_output":
        if best_rate >= 99 and worst_rate >= 99:
            return "All models returned the expected JSON on every task."
        return (
            f"{best['model']} produced valid JSON {best_rate:.0f}% of the time, "
            f"{worst['model']} only {worst_rate:.0f}%."
        )

    if suite_name == "code_generation":
        if best_rate == 0:
            return (
                "No model solved any HumanEval task in this run — likely a "
                "regression in extraction or sandbox setup, not the models."
            )
        if best_rate >= 80:
            return (
                f"{best['model']} solved {best['passes']}/{best['total']} HumanEval tasks "
                f"({best_rate:.0f}%); {worst['model']} trailed at {worst_rate:.0f}%."
            )
        return (
            f"All models struggled. {best['model']} led with {best['passes']}/{best['total']} "
            f"({best_rate:.0f}%); {worst['model']} only {worst['passes']}/{worst['total']}."
        )

    if suite_name == "sustained_throughput":
        # Sustained_throughput model rows carry the soak-specific metrics under
        # whatever keys the runner wrote; locate the worst throttler (largest
        # drop from initial → sustained) and pick out peak temperature.
        scored: list[tuple[float, dict[str, Any]]] = []
        for m in models:
            init = m.get("initial_tokens_per_s")
            sus = m.get("sustained_tokens_per_s")
            if init and sus is not None and init > 0:
                scored.append(((sus / init), m))
        if not scored:
            return f"Sustained loop completed for {len(models)} model(s); no per-window throughput captured."
        scored.sort(key=lambda pair: pair[0])
        worst_ratio, worst_m = scored[0]
        best_ratio, best_m = scored[-1]
        peak_temp = max(
            (m.get("peak_temp_c") for m in models if m.get("peak_temp_c") is not None),
            default=None,
        )
        bits = [
            f"{best_m['model']} held {best_ratio * 100:.0f}% of its initial throughput "
            f"({best_m.get('initial_tokens_per_s')} → {best_m.get('sustained_tokens_per_s')} tok/s)."
        ]
        if worst_m is not best_m:
            bits.append(
                f"{worst_m['model']} throttled to {worst_ratio * 100:.0f}% "
                f"({worst_m.get('initial_tokens_per_s')} → {worst_m.get('sustained_tokens_per_s')} tok/s)."
            )
        if peak_temp is not None:
            bits.append(f"Peak GPU temp observed: {peak_temp:.0f} °C.")
        return " ".join(bits)

    if suite_name in ("long_context_retrieval", "long_context_retrieval_v1"):
        bits = [
            f"{best['model']} retrieved best overall at {best_rate:.0f}% across the grid."
        ]
        breakers = [m for m in models if m.get("first_failure_length")]
        if breakers:
            soonest = min(breakers, key=lambda m: m["first_failure_length"])
            bits.append(
                f"Retrieval first breaks down for {soonest['model']} at "
                f"{soonest['first_failure_length']} tokens."
            )
        else:
            bits.append("No model dropped below the failure threshold at any tested length.")
        return " ".join(bits)

    return ""


def _verdict_paragraph(
    ranking: list[dict[str, Any]],
    suite_map: dict[str, dict[str, Any] | None],
) -> str:
    """Free-form English verdict tying together the per-suite picture."""
    if not ranking:
        return ""
    leader = ranking[0]
    bits: list[str] = []
    bits.append(
        f"{leader['model']} is the strongest overall performer in this run."
    )
    # Identify per-suite specialists when they differ from the leader.
    def _winner(suite_key: str) -> str | None:
        suite = suite_map.get(suite_key)
        if not suite or not suite.get("models"):
            return None
        return max(suite["models"], key=lambda m: float(m.get("pass_rate") or 0.0))["model"]

    speed_winner = None
    speed_suite = suite_map.get("openclaw_speed")
    if speed_suite and speed_suite.get("models"):
        with_ttft = [m for m in speed_suite["models"] if m.get("avg_ttft_ms") is not None]
        if with_ttft:
            speed_winner = min(with_ttft, key=lambda m: m["avg_ttft_ms"])["model"]

    grounding_winner = _winner("hallucination_grounding")
    code_winner = _winner("code_generation")

    if speed_winner and speed_winner != leader["model"]:
        bits.append(f"{speed_winner} is the right pick if first-token latency matters most.")
    if grounding_winner and grounding_winner != leader["model"]:
        bits.append(f"{grounding_winner} is the safer choice when factual grounding is critical.")
    if code_winner and code_winner != leader["model"]:
        bits.append(f"{code_winner} is the strongest coder among the tested models.")
    return " ".join(bits)
