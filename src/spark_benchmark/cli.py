from __future__ import annotations

import curses
import json
import os
import sys
from pathlib import Path

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from spark_benchmark.code_generation import (
    default_reference_scores_path,
    run_code_generation_suite,
)
from spark_benchmark.sustained_throughput import (
    load_sustained_throughput_suite,
    run_sustained_throughput_suite,
)
from spark_benchmark.config import load_backend, load_experiment, load_model_config, load_platform
from spark_benchmark.custom_suites import (
    load_custom_suite,
    run_custom_suite_quick,
    slugify_suite_name,
    validate_custom_suite,
)
from spark_benchmark.quick import (
    QUICK_TASK_ID,
    build_quick_suite,
    default_save_dir,
    save_quick_suite_as_yaml,
)
from spark_benchmark.model_registry import (
    detect_ollama_models,
    find_config_by_name_or_tag,
    resolve_runnable_models,
)
from spark_benchmark.orchestration import BenchmarkPlan, parse_benchmark_request, run_benchmark_bundle
from spark_benchmark.reporting import aggregate_runs, render_cli_benchmark_summary, write_report
from spark_benchmark.reliability import (
    load_reliability_suite,
    run_hallucination_grounding_suite,
    run_practical_structured_output_suite,
)
from spark_benchmark.results_bundle import ensure_run_dir, make_run_id, write_json, write_manifest, write_result
from spark_benchmark.runners.registry import build_backend
from spark_benchmark.runtime import build_manifest

APP_HELP = """[bold cyan]
  ▄██████╗  ██████╗ ██╗  ██╗
  ██╔══██╗██╔════╝ ╚██╗██╔╝
  ██║  ██║██║  ███╗╚███╔╝
  ██║  ██║██║   ██║██╔██╗
  ██████╔╝╚██████╔╝██╔╝ ██╗
  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝
[/bold cyan]
[bold]spark-benchmark[/bold] // pixel CLI for DGX Spark testing
"""

app = typer.Typer(invoke_without_command=True, help=APP_HELP, rich_markup_mode="rich")
console = Console()


def maybe_print_banner() -> None:
    if not sys.stdout.isatty():
        return
    if os.environ.get("SPARK_BENCH_NO_BANNER") == "1":
        return

    art = Text(
        "\n".join(
            [
                "  ▄██████╗  ██████╗ ██╗  ██╗",
                "  ██╔══██╗██╔════╝ ╚██╗██╔╝",
                "  ██║  ██║██║  ███╗╚███╔╝ ",
                "  ██║  ██║██║   ██║██╔██╗ ",
                "  ██████╔╝╚██████╔╝██╔╝ ██╗",
                "  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝",
                "   spark-benchmark // local model trials",
            ]
        ),
        style="bold #7dd3fc",
    )
    subtitle = Text("pixel CLI for DGX Spark benchmarking", style="#f59e0b")
    console.print(
        Panel.fit(
            Text.assemble(art, "\n", subtitle),
            border_style="#38bdf8",
            padding=(1, 2),
            title="[bold white]DGX Spark[/bold white]",
            subtitle="[bold #a3e635]test • compare • break[/bold #a3e635]",
        )
    )


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_runtime_context(experiment: Path, platform: str) -> tuple[Path, object, object, list[object]]:
    repo_root = resolve_repo_root()
    experiment_file = load_experiment(experiment)
    experiment_spec = experiment_file.experiment
    platform_path = repo_root / "configs" / "platforms" / f"{platform}.yaml"
    backend_path = repo_root / "configs" / "backends" / f"{experiment_spec.backend.value}.yaml"
    if not platform_path.exists():
        raise typer.BadParameter(f"Unknown platform config: {platform_path}")
    if not backend_path.exists():
        raise typer.BadParameter(f"Unknown backend config: {backend_path}")

    platform_config = load_platform(platform_path)
    backend_config = load_backend(backend_path)
    missing_models = []
    model_configs = []
    for model_name in experiment_spec.models:
        model_path = repo_root / "configs" / "models" / f"{model_name}.yaml"
        if not model_path.exists():
            missing_models.append(model_name)
        else:
            model_configs.append(load_model_config(model_path))
    if missing_models:
        raise typer.BadParameter(f"Missing model configs: {', '.join(missing_models)}")
    return repo_root, experiment_spec, backend_config, model_configs


def detect_ollama_model_tags(backend_config: object) -> set[str]:
    """Lightweight wrapper kept for callers that only need the set of tags.

    Delegates to :func:`spark_benchmark.model_registry.detect_ollama_models`
    so the curses TUI and the Typer commands share one detection path.
    """
    return {item.tag for item in detect_ollama_models(backend_config)}  # type: ignore[arg-type]


def parse_index_selection(raw_value: str, total: int) -> list[int]:
    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not tokens:
        raise typer.BadParameter("Selection must not be empty.")
    indexes: list[int] = []
    for token in tokens:
        if token.lower() == "all":
            return list(range(1, total + 1))
        try:
            index = int(token)
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid selection token: {token}") from exc
        if index < 1 or index > total:
            raise typer.BadParameter(f"Selection {index} is outside the allowed range 1-{total}.")
        if index not in indexes:
            indexes.append(index)
    return indexes


WIZARD_BANNER = [
    "  ▄██████╗  ██████╗ ██╗  ██╗",
    "  ██╔══██╗██╔════╝ ╚██╗██╔╝",
    "  ██║  ██║██║  ███╗╚███╔╝",
    "  ██║  ██║██║   ██║██╔██╗",
    "  ██████╔╝╚██████╔╝██╔╝ ██╗",
    "  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝",
    "  spark-benchmark // benchmark wizard",
]


def _run_curses_multiselect(
    title: str,
    options: list[str],
    preselected: set[int] | None = None,
    header_lines: list[str] | None = None,
    disabled: set[int] | None = None,
) -> list[int]:
    preselected = set() if preselected is None else set(preselected)
    header_lines = [] if header_lines is None else list(header_lines)
    disabled = set() if disabled is None else set(disabled)
    enabled_indexes = [idx for idx in range(len(options)) if idx not in disabled]
    if not enabled_indexes:
        return []

    def _inner(stdscr: object) -> list[int]:
        window = stdscr
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        window.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, 8, -1)
        current = enabled_indexes[0]
        selected = {idx for idx in preselected if idx not in disabled}

        while True:
            window.erase()
            height, width = window.getmaxyx()
            header = [*header_lines, "", title, "Up/Down move • Space toggle • Enter confirm • greyed entries are unavailable."]
            row = 0
            for line in header:
                if row < height:
                    window.addnstr(row, 0, line, max(0, width - 1))
                row += 1
            row += 1

            visible_height = max(1, height - row)
            start = 0
            if current >= visible_height:
                start = current - visible_height + 1

            for idx in range(start, min(len(options), start + visible_height)):
                is_disabled = idx in disabled
                if is_disabled:
                    marker = "·"
                else:
                    marker = "●" if idx in selected else "○"
                line = f"{marker} {options[idx]}"
                attr = curses.A_REVERSE if idx == current else curses.A_NORMAL
                if is_disabled and curses.has_colors():
                    attr |= curses.color_pair(2) | curses.A_DIM
                elif idx in selected and curses.has_colors():
                    attr |= curses.color_pair(1)
                window.addnstr(row + idx - start, 0, line, max(0, width - 1), attr)

            window.refresh()
            key = window.getch()
            if key in (curses.KEY_UP, ord("k")):
                current = (current - 1) % len(options)
            elif key in (curses.KEY_DOWN, ord("j")):
                current = (current + 1) % len(options)
            elif key == ord(" "):
                if current in disabled:
                    continue
                if current in selected:
                    selected.remove(current)
                else:
                    selected.add(current)
            elif key in (10, 13, curses.KEY_ENTER):
                if selected:
                    return sorted(selected)
            elif key in (27, ord("q")):
                return []

    return curses.wrapper(_inner)


def prompt_multiselect(
    title: str,
    options: list[str],
    default_indexes: list[int] | None = None,
    header_lines: list[str] | None = None,
    disabled_indexes: list[int] | None = None,
) -> list[int]:
    default_indexes = [] if default_indexes is None else default_indexes
    disabled_zero_based = {index - 1 for index in (disabled_indexes or [])}
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _run_curses_multiselect(
            title,
            options,
            {index - 1 for index in default_indexes},
            header_lines=header_lines,
            disabled=disabled_zero_based,
        )

    console.print(f"[yellow]{title}[/yellow]")
    for index, option in enumerate(options, start=1):
        if (index - 1) in disabled_zero_based:
            console.print(f"[dim]{index}. {option} (unavailable)[/dim]")
        else:
            console.print(f"{index}. {option}")
    enabled_defaults = [i for i in default_indexes if (i - 1) not in disabled_zero_based]
    default = ",".join(str(index) for index in enabled_defaults) if enabled_defaults else "all"
    answer = typer.prompt("Select by number", default=default)
    selection = parse_index_selection(answer, len(options))
    blocked = [index for index in selection if (index - 1) in disabled_zero_based]
    if blocked:
        raise typer.BadParameter(f"Entries {blocked} are unavailable.")
    return [index - 1 for index in selection]


@app.callback()
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from spark_benchmark.shell import run_shell

        run_shell()


@app.command("shell")
def shell_command() -> None:
    """Launch the interactive spark-benchmark shell."""
    from spark_benchmark.shell import run_shell

    run_shell()


def shell_entrypoint() -> None:
    """Console-script entry point for `spark_benchmark`."""
    from spark_benchmark.shell import run_shell

    run_shell()


@app.command()
def run(
    experiment: Path = typer.Option(..., exists=True, dir_okay=False),
    platform: str = typer.Option(...),
    dry_run: bool = typer.Option(False, help="Only validate config and print resolved manifest."),
    smoke_prompt: str | None = typer.Option(None, help="Run a single generation smoke test against the first configured model."),
    run_suite: str | None = typer.Option(None, help="Run a built-in suite such as hallucination_grounding."),
    allow_auto_detected: bool = typer.Option(
        False,
        "--allow-auto-detected",
        help=(
            "Also run against any chat model present in Ollama that is not in the "
            "experiment YAML. Auto-detected entries use Ollama defaults and are "
            "labeled in the manifest. Off by default for reproducibility."
        ),
    ),
) -> None:
    maybe_print_banner()
    repo_root, experiment_spec, backend_config, experiment_model_configs = load_runtime_context(
        experiment, platform
    )
    platform_config = load_platform(repo_root / "configs" / "platforms" / f"{platform}.yaml")

    resolved = resolve_runnable_models(
        backend_config=backend_config,
        experiment_model_configs=experiment_model_configs,
        allow_auto_detected=allow_auto_detected,
    )
    model_configs = resolved.configs

    runs_root = repo_root / "results" / "runs"
    run_id = make_run_id()
    run_dir = ensure_run_dir(runs_root, run_id)

    manifest = build_manifest(
        experiment=experiment_spec,
        platform_config=platform_config,
        backend_config=backend_config,
        model_names=[m.name for m in model_configs],
        results_dir=run_dir,
    )
    write_manifest(run_dir, manifest)
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if dry_run:
        return
    if smoke_prompt:
        backend = build_backend(backend_config)
        backend.load_model(model_configs[0])
        result = backend.generate(smoke_prompt, experiment_spec.sampling)
        write_result(run_dir, result)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        backend.unload()
        return
    if run_suite:
        backend = build_backend(backend_config)
        if run_suite in {
            "long_context_retrieval",
            "long_context_retrieval_v1",
            "long_context_retrieval_fast",
        }:
            from spark_benchmark.long_context import (
                load_haystack_texts,
                load_long_context_fixture,
                profile_for_suite_name,
                resolve_profile_matrix,
                run_long_context_suite,
            )

            fixture = load_long_context_fixture(
                repo_root / "data" / "long_context" / "long_context_retrieval_v1.json"
            )
            haystack_texts = load_haystack_texts(fixture, repo_root)
            matrix = resolve_profile_matrix(fixture, profile_for_suite_name(run_suite))
            summary = run_long_context_suite(
                run_dir=run_dir,
                fixture=fixture,
                haystack_texts=haystack_texts,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment_spec.sampling,
                matrix=matrix,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        suite = load_reliability_suite(repo_root, run_suite)
        if run_suite in {"hallucination_grounding", "hallucination_grounding_v1"}:
            summary = run_hallucination_grounding_suite(
                run_dir=run_dir,
                suite=suite,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment_spec.sampling,
            )
        elif run_suite in {"practical_structured_output", "practical_structured_output_v1"}:
            summary = run_practical_structured_output_suite(
                run_dir=run_dir,
                suite=suite,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment_spec.sampling,
            )
        elif run_suite in {"code_generation", "code_generation_v1"}:
            summary = run_code_generation_suite(
                run_dir=run_dir,
                suite=suite,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment_spec.sampling,
                reference_scores_path=default_reference_scores_path(repo_root),
            )
        elif run_suite in {"sustained_throughput", "sustained_throughput_v1"}:
            suite = load_sustained_throughput_suite(repo_root)
            summary = run_sustained_throughput_suite(
                run_dir=run_dir,
                suite=suite,
                backend=backend,
                backend_config=backend_config,
                model_configs=model_configs,
                sampling=experiment_spec.sampling,
            )
        else:
            raise typer.BadParameter(f"Unsupported built-in suite: {run_suite}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print("[yellow]full run execution not implemented yet; use --smoke-prompt for a single-generation backend check[/yellow]")


@app.command("console")
def console_run(
    experiment: Path = typer.Option(..., "--experiment", exists=True, dir_okay=False),
    platform: str = typer.Option(..., "--platform"),
    model: str | None = typer.Option(
        None,
        "--model",
        help=(
            "Model to chat with. Accepts an experiment model name "
            "(e.g. ``qwen-3.6``), an Ollama tag (``phi4:14b``), or its "
            "slugified form (``phi4-14b``). Without --allow-auto-detected, "
            "only experiment models resolve."
        ),
    ),
    allow_auto_detected: bool = typer.Option(
        False,
        "--allow-auto-detected",
        help=(
            "Allow chatting with any chat model Ollama reports, even when it "
            "is not in the experiment YAML. The auto-detected config uses "
            "Ollama defaults (no curated sampling, ctx, or notes)."
        ),
    ),
) -> None:
    maybe_print_banner()
    _, experiment_spec, backend_config, experiment_model_configs = load_runtime_context(
        experiment, platform
    )
    resolved = resolve_runnable_models(
        backend_config=backend_config,
        experiment_model_configs=experiment_model_configs,
        allow_auto_detected=allow_auto_detected,
    )
    model_configs = resolved.configs

    if not model_configs:
        raise typer.BadParameter("No models available. Pass --allow-auto-detected to use Ollama tags directly.")

    if model is None:
        selected_model = model_configs[0]
    else:
        selected_model = find_config_by_name_or_tag(
            model, configs=model_configs, classified=resolved.classified
        )
    if selected_model is None:
        available = ", ".join(item.name for item in model_configs)
        raise typer.BadParameter(f"Unknown model '{model}'. Available: {available}")

    backend = build_backend(backend_config)
    backend.load_model(selected_model)
    console.print(
        Panel.fit(
            f"model: {selected_model.name}\nbackend: {backend_config.name.value}\n\nWrite a prompt and press Enter.\nCommands: /exit, /quit",
            title="[bold white]Interactive Console[/bold white]",
            border_style="#f59e0b",
        )
    )
    try:
        while True:
            prompt = typer.prompt("prompt", prompt_suffix=" > ")
            if prompt.strip().lower() in {"/exit", "/quit"}:
                break
            if not prompt.strip():
                continue
            result = backend.generate(prompt, experiment_spec.sampling)
            console.print(Panel(result.output, title=f"[bold #7dd3fc]{selected_model.name}[/bold #7dd3fc]", border_style="#38bdf8"))
    finally:
        backend.unload()


@app.command()
def benchmark(
    request: list[str] = typer.Argument(..., help="Natural-language benchmark request."),
    experiment: Path = typer.Option(..., "--experiment", exists=True, dir_okay=False),
    platform: str = typer.Option(..., "--platform"),
    allow_auto_detected: bool = typer.Option(
        False,
        "--allow-auto-detected",
        help=(
            "Expand the candidate model pool to include any chat model "
            "Ollama reports, even if it is not in the experiment YAML. "
            "Recognised by slugified name (``phi4:14b`` -> ``phi4-14b``)."
        ),
    ),
) -> None:
    maybe_print_banner()
    repo_root, experiment_spec, backend_config, experiment_model_configs = load_runtime_context(
        experiment, platform
    )
    platform_config = load_platform(repo_root / "configs" / "platforms" / f"{platform}.yaml")
    resolved = resolve_runnable_models(
        backend_config=backend_config,
        experiment_model_configs=experiment_model_configs,
        allow_auto_detected=allow_auto_detected,
    )
    all_model_configs = resolved.configs
    request_text = " ".join(request).strip()
    available_models = [model.name for model in all_model_configs]
    plan = parse_benchmark_request(request_text, available_models)
    selected_configs = [model for model in all_model_configs if model.name in plan.selected_models]
    if not selected_configs:
        raise typer.BadParameter("No models selected after parsing the request.")

    bundle_dir = repo_root / "results" / "benchmarks" / make_run_id()
    backend = build_backend(backend_config)
    result = run_benchmark_bundle(
        bundle_dir=bundle_dir,
        repo_root=repo_root,
        experiment=experiment_spec,
        platform_config=platform_config,
        backend_config=backend_config,
        model_configs=selected_configs,
        backend=backend,
        plan=plan,
    )
    aggregate = aggregate_runs(bundle_dir)
    report_path = bundle_dir / "report.md"
    report_html_path = bundle_dir / "report.html"
    write_report(report_path, "both", aggregate)
    summary = render_cli_benchmark_summary(
        request=request_text,
        selected_models=plan.selected_models,
        selected_suites=plan.selected_suites,
        aggregate=aggregate,
        report_path=report_path,
    )
    print(summary)
    print(f"HTML report: {report_html_path}")


@app.command()
def wizard(
    experiment: Path = typer.Option(..., "--experiment", exists=True, dir_okay=False),
    platform: str = typer.Option(..., "--platform"),
    allow_auto_detected: bool = typer.Option(
        False,
        "--allow-auto-detected",
        help=(
            "Also offer any chat model Ollama reports, not just the curated "
            "experiment lineup. Auto-detected entries are flagged in the "
            "model picker."
        ),
    ),
) -> None:
    repo_root, experiment_spec, backend_config, experiment_model_configs = load_runtime_context(
        experiment, platform
    )
    platform_config = load_platform(repo_root / "configs" / "platforms" / f"{platform}.yaml")

    detected = detect_ollama_models(backend_config)
    detected_tags = {item.tag for item in detected}
    curated_available = [
        model
        for model in experiment_model_configs
        if (model.artifact_path or model.revision) in detected_tags
    ]
    available_configs: list = list(curated_available)
    auto_names: set[str] = set()
    if allow_auto_detected:
        resolved = resolve_runnable_models(
            backend_config=backend_config,
            experiment_model_configs=experiment_model_configs,
            allow_auto_detected=True,
        )
        for cfg in resolved.auto_detected_configs:
            if cfg.name not in {m.name for m in available_configs}:
                available_configs.append(cfg)
                auto_names.add(cfg.name)
    if not available_configs:
        if allow_auto_detected:
            raise typer.BadParameter(
                "Ollama returned no chat-capable models. Pull at least one with `ollama pull <tag>` and retry."
            )
        raise typer.BadParameter(
            "No configured experiment models were detected in Ollama. "
            "Pass --allow-auto-detected to also offer any non-vision tag from `ollama list`."
        )

    wizard_intro = [
        *WIZARD_BANNER,
        "Choose models first, then choose test suites.",
        "Detected Ollama models mapped to this experiment:"
        + (" (curated + auto-detected)" if auto_names else ""),
    ]
    model_options = [
        (
            f"{model.name}  [{model.artifact_path or model.revision}]"
            + ("  (auto-detected)" if model.name in auto_names else "")
        )
        for model in available_configs
    ]
    selected_model_indexes = prompt_multiselect(
        "Select models",
        model_options,
        default_indexes=list(range(1, len(available_configs) + 1)),
        header_lines=wizard_intro,
    )
    selected_configs = [available_configs[index] for index in selected_model_indexes]

    suite_options = [
        ("openclaw_speed", "OpenClaw-like speed probe"),
        ("hallucination_grounding", "Grounding / hallucination reliability"),
        ("practical_structured_output", "Structured output reliability"),
        ("code_generation", "Code generation (HumanEval starter)"),
    ]
    selected_suite_indexes = prompt_multiselect(
        "Select test suites",
        [label for _, label in suite_options],
        default_indexes=list(range(1, len(suite_options) + 1)),
        header_lines=[*WIZARD_BANNER, "Choose which benchmark suites to run."],
    )
    selected_suites = [suite_options[index][0] for index in selected_suite_indexes]

    plan = BenchmarkPlan(
        request="interactive wizard selection",
        selected_models=[model.name for model in selected_configs],
        selected_suites=selected_suites,
        rationale=["Interactive wizard selection."],
    )

    maybe_print_banner()
    console.print(
        Panel.fit(
            f"Models: {', '.join(plan.selected_models)}\nSuites: {', '.join(plan.selected_suites)}\n\nRunning benchmark now. This can take a while depending on the selected models.",
            title="[bold white]Benchmark Run[/bold white]",
            border_style="#f59e0b",
        )
    )

    bundle_dir = repo_root / "results" / "benchmarks" / make_run_id()
    backend = build_backend(backend_config)
    run_benchmark_bundle(
        bundle_dir=bundle_dir,
        repo_root=repo_root,
        experiment=experiment_spec,
        platform_config=platform_config,
        backend_config=backend_config,
        model_configs=selected_configs,
        backend=backend,
        plan=plan,
        progress_callback=lambda message: console.print(f"[cyan]{message}[/cyan]"),
    )
    aggregate = aggregate_runs(bundle_dir)
    report_path = bundle_dir / "report.md"
    report_html_path = bundle_dir / "report.html"
    write_report(report_path, "both", aggregate)
    summary = render_cli_benchmark_summary(
        request="interactive wizard selection",
        selected_models=plan.selected_models,
        selected_suites=plan.selected_suites,
        aggregate=aggregate,
        report_path=report_path,
    )
    print(summary)
    print(f"HTML report: {report_html_path}")


@app.command()
def aggregate(runs: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True)) -> None:
    maybe_print_banner()
    summary = aggregate_runs(runs)
    json_path = write_json(runs / "aggregate.json", summary)
    md_path = runs / "aggregate.md"
    html_path = runs / "aggregate.html"
    write_report(md_path, "both", summary)
    print(
        json.dumps(
            {
                "status": "ok",
                "runs_dir": str(runs),
                "aggregate_json": str(json_path),
                "aggregate_markdown": str(md_path),
                "aggregate_html": str(html_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("validate-custom")
def validate_custom_command(
    suite: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a custom suite YAML or JSON.",
    ),
    experiment: Path | None = typer.Option(
        None,
        "--experiment",
        exists=True,
        dir_okay=False,
        help="Optional experiment file. When provided, also checks suite.models against the curated lineup.",
    ),
    platform: str | None = typer.Option(
        None, "--platform", help="Required only when --experiment is set."
    ),
) -> None:
    """Validate a custom suite without running it.

    Catches schema errors (duplicate task IDs, empty prompts, mode not yet
    supported), warns about long prompts, and verifies any
    ``suite.models`` references against the experiment lineup when one is
    provided. Exits non-zero on any error-severity issue.
    """
    maybe_print_banner()
    try:
        loaded = load_custom_suite(suite)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]ERROR loading {suite}: {exc}[/red]")
        raise typer.Exit(code=2)

    available_models: list[str] | None = None
    if experiment is not None:
        if platform is None:
            raise typer.BadParameter("--platform is required when --experiment is provided")
        _, _, _, experiment_model_configs = load_runtime_context(experiment, platform)
        available_models = [m.name for m in experiment_model_configs]

    issues = validate_custom_suite(loaded, available_models=available_models)

    payload = {
        "suite": loaded.name,
        "version": loaded.version,
        "mode": loaded.mode,
        "task_count": len(loaded.tasks),
        "issues": [{"severity": i.severity, "message": i.message} for i in issues],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise typer.Exit(code=1)


@app.command("run-custom")
def run_custom_command(
    suite: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a custom suite YAML or JSON.",
    ),
    experiment: Path = typer.Option(
        ...,
        "--experiment",
        exists=True,
        dir_okay=False,
        help="Experiment YAML — provides the backend and the curated model pool.",
    ),
    platform: str = typer.Option(..., "--platform"),
    models: str | None = typer.Option(
        None,
        "--models",
        help=(
            "Comma-separated model names overriding the suite's `models:` list. "
            "Each entry is a curated experiment name or a slugified Ollama tag "
            "(e.g. `phi4-14b`)."
        ),
    ),
    allow_auto_detected: bool = typer.Option(
        True,
        "--allow-auto-detected/--no-allow-auto-detected",
        help=(
            "Defaults to True for custom suites — the user explicitly opted "
            "in to a non-canonical workload. Pass --no-allow-auto-detected "
            "to restrict the model pool to the experiment YAML."
        ),
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help="Always start fresh; do not skip (model, task) pairs already in results.jsonl.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Where to write the run bundle. Defaults to results/custom/<slug>/<run-id>/.",
    ),
) -> None:
    """Execute a Mode A custom suite end-to-end.

    No scoring is performed in v0.2.0 — this just generates each model's
    response to each prompt, captures telemetry, and writes a side-by-side
    Markdown summary. See `docs/custom-tests-spec.md` for the v0.3+
    roadmap that adds scoring, judges, and sharing.
    """
    maybe_print_banner()
    try:
        loaded = load_custom_suite(suite)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"could not load {suite}: {exc}") from exc

    repo_root, experiment_spec, backend_config, experiment_model_configs = load_runtime_context(
        experiment, platform
    )
    resolved = resolve_runnable_models(
        backend_config=backend_config,
        experiment_model_configs=experiment_model_configs,
        allow_auto_detected=allow_auto_detected,
    )
    available_models = [m.name for m in resolved.configs]

    issues = validate_custom_suite(loaded, available_models=available_models)
    error_issues = [i for i in issues if i.severity == "error"]
    if error_issues:
        for issue in issues:
            console.print(f"[red]{issue.render()}[/red]" if issue.severity == "error" else f"[yellow]{issue.render()}[/yellow]")
        raise typer.Exit(code=1)
    for issue in issues:
        console.print(f"[yellow]{issue.render()}[/yellow]")

    if models:
        wanted = [name.strip() for name in models.split(",") if name.strip()]
    elif loaded.models:
        wanted = list(loaded.models)
    else:
        wanted = available_models

    selected_configs = []
    for name in wanted:
        cfg = find_config_by_name_or_tag(
            name, configs=resolved.configs, classified=resolved.classified
        )
        if cfg is None:
            raise typer.BadParameter(
                f"model {name!r} not found. Available: {', '.join(available_models) or '(none)'}"
            )
        if cfg not in selected_configs:
            selected_configs.append(cfg)

    if not selected_configs:
        raise typer.BadParameter("no models selected for the run")

    run_id = make_run_id()
    if output_dir is not None:
        run_dir = output_dir
    else:
        run_dir = (
            repo_root
            / "results"
            / "custom"
            / slugify_suite_name(loaded.name)
            / run_id
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "manifest.json",
        {
            "kind": "custom",
            "run_id": run_id,
            "suite": loaded.name,
            "suite_version": loaded.version,
            "suite_path": str(suite.resolve()),
            "experiment": experiment_spec.name,
            "platform": platform,
            "backend": backend_config.name.value,
            "models": [cfg.name for cfg in selected_configs],
            "task_count": len(loaded.tasks),
            "mode": loaded.mode,
            "auto_detected_models": [
                cfg.name for cfg in resolved.auto_detected_configs if cfg in selected_configs
            ],
        },
    )

    backend = build_backend(backend_config)
    summary = run_custom_suite_quick(
        suite=loaded,
        backend=backend,
        backend_config=backend_config,
        model_configs=selected_configs,
        run_dir=run_dir,
        default_sampling=experiment_spec.sampling,
        progress_callback=lambda message: console.print(f"[cyan]{message}[/cyan]"),
        resume=not no_resume,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "run_dir": str(run_dir),
                "results": str(run_dir / "results.jsonl"),
                "summary_md": str(run_dir / "summary.md"),
                "summary_html": str(run_dir / "summary.html"),
                "summary_json": str(run_dir / "summary.json"),
                "task_count": len(loaded.tasks),
                "models": [cfg.name for cfg in selected_configs],
                "per_model": [
                    {
                        "model": bucket["model"],
                        "completed": bucket["tasks_completed"],
                        "errored": bucket["tasks_errored"],
                        "mean_ttft_ms": bucket["mean_ttft_ms"],
                        "mean_decode_tps": bucket["mean_decode_tps"],
                    }
                    for bucket in summary["per_model"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("quick")
def quick_command(
    prompt: str = typer.Argument(
        ...,
        help=(
            "The prompt to fan out to every selected model. Use shell quoting "
            "for multi-line input (e.g. bash $'...' or a quoted heredoc)."
        ),
    ),
    experiment: Path = typer.Option(
        ...,
        "--experiment",
        exists=True,
        dir_okay=False,
        help="Experiment YAML — provides the backend and the curated model pool.",
    ),
    platform: str = typer.Option(..., "--platform"),
    models: str | None = typer.Option(
        None,
        "--models",
        help=(
            "Comma-separated model names to fan the prompt out to. Defaults "
            "to every chat-capable model the resolver returns "
            "(curated + auto-detected, vision/embedding tags filtered)."
        ),
    ),
    allow_auto_detected: bool = typer.Option(
        True,
        "--allow-auto-detected/--no-allow-auto-detected",
        help=(
            "Defaults to True. The whole point of `quick` is to compare "
            "whatever you happen to have pulled in Ollama; pass "
            "--no-allow-auto-detected to restrict the lineup to the "
            "experiment YAML."
        ),
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help=(
            "Human-readable suite name (also used as the run-bundle slug). "
            "Defaults to a slug derived from the first ~40 chars of the "
            "prompt, prefixed with `quick-`."
        ),
    ),
    save: bool = typer.Option(
        False,
        "--save/--no-save",
        help=(
            "Persist the prompt as a reusable custom suite YAML before the "
            "run starts. Saved under examples/custom-tests/quick-saved/<slug>/ "
            "(git-ignored) so the TUI's Custom menu surfaces it next time."
        ),
    ),
    save_path: Path | None = typer.Option(
        None,
        "--save-path",
        help=(
            "Override the save location entirely. Implies --save. "
            "Must be an absolute or relative directory; suite.yaml is "
            "written inside it."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Allow saving over an existing suite.yaml at the target path.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Where to write the run bundle. Defaults to results/custom/<slug>/<run-id>/.",
    ),
) -> None:
    """Fan out a single ad-hoc prompt to every selected model.

    This is BYOT Mode A's lightest entry point — no YAML required.
    The prompt is built into a one-task ``CustomSuiteDefinition`` in
    memory and run through the same pipeline as ``run-custom``, so the
    output (``results/custom/<slug>/<run-id>/`` with ``manifest.json``,
    ``results.jsonl``, ``summary.md``, ``summary.json``) is identical
    to what a full custom suite produces.
    """
    maybe_print_banner()

    repo_root, experiment_spec, backend_config, experiment_model_configs = load_runtime_context(
        experiment, platform
    )
    resolved = resolve_runnable_models(
        backend_config=backend_config,
        experiment_model_configs=experiment_model_configs,
        allow_auto_detected=allow_auto_detected,
    )
    available_models = [m.name for m in resolved.configs]
    if not available_models:
        raise typer.BadParameter(
            "no runnable models found. Pull at least one chat-capable model "
            "into Ollama (or pass --no-allow-auto-detected with an experiment "
            "whose curated lineup is on disk)."
        )

    if models:
        wanted = [name.strip() for name in models.split(",") if name.strip()]
    else:
        wanted = available_models

    selected_configs = []
    for needle in wanted:
        cfg = find_config_by_name_or_tag(
            needle, configs=resolved.configs, classified=resolved.classified
        )
        if cfg is None:
            raise typer.BadParameter(
                f"model {needle!r} not found. Available: {', '.join(available_models)}"
            )
        if cfg not in selected_configs:
            selected_configs.append(cfg)
    if not selected_configs:
        raise typer.BadParameter("no models selected for the quick run")

    suite = build_quick_suite(
        prompt,
        name=name,
        sampling=experiment_spec.sampling,
    )

    saved_path: Path | None = None
    if save or save_path is not None:
        save_root = save_path if save_path is not None else default_save_dir(repo_root)
        try:
            saved_path = save_quick_suite_as_yaml(
                suite, save_root=save_root, overwrite=overwrite
            )
        except FileExistsError as exc:
            raise typer.BadParameter(
                f"a suite.yaml already exists at {exc}; pass --overwrite to replace it"
            ) from exc

    run_id = make_run_id()
    if output_dir is not None:
        run_dir = output_dir
    else:
        run_dir = (
            repo_root
            / "results"
            / "custom"
            / slugify_suite_name(suite.name)
            / run_id
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "manifest.json",
        {
            "kind": "custom",
            "run_id": run_id,
            "suite": suite.name,
            "suite_version": suite.version,
            "suite_path": str(saved_path) if saved_path is not None else None,
            "experiment": experiment_spec.name,
            "platform": platform,
            "backend": backend_config.name.value,
            "models": [cfg.name for cfg in selected_configs],
            "task_count": len(suite.tasks),
            "mode": suite.mode,
            "auto_detected_models": [
                cfg.name for cfg in resolved.auto_detected_configs if cfg in selected_configs
            ],
            "source": "cli-quick",
            "ad_hoc_prompt": True,
        },
    )

    backend = build_backend(backend_config)
    summary = run_custom_suite_quick(
        suite=suite,
        backend=backend,
        backend_config=backend_config,
        model_configs=selected_configs,
        run_dir=run_dir,
        default_sampling=experiment_spec.sampling,
        progress_callback=lambda message: console.print(f"[cyan]{message}[/cyan]"),
        resume=False,  # quick runs are single-task; resume would just re-skip the row.
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "run_dir": str(run_dir),
                "results": str(run_dir / "results.jsonl"),
                "summary_md": str(run_dir / "summary.md"),
                "summary_html": str(run_dir / "summary.html"),
                "summary_json": str(run_dir / "summary.json"),
                "saved_suite_yaml": str(saved_path) if saved_path is not None else None,
                "task_id": QUICK_TASK_ID,
                "models": [cfg.name for cfg in selected_configs],
                "per_model": [
                    {
                        "model": bucket["model"],
                        "completed": bucket["tasks_completed"],
                        "errored": bucket["tasks_errored"],
                        "mean_ttft_ms": bucket["mean_ttft_ms"],
                        "mean_decode_tps": bucket["mean_decode_tps"],
                    }
                    for bucket in summary["per_model"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def report(
    runs: Path = typer.Option(Path("results/runs"), exists=True, file_okay=False, dir_okay=True),
    format: str = typer.Option("markdown", "--format"),
    output: Path = typer.Option(...),
) -> None:
    maybe_print_banner()
    aggregate = aggregate_runs(runs)
    write_report(output, format, aggregate)
    print(json.dumps({"status": "ok", "output": str(output), "format": format}, ensure_ascii=False, indent=2))


@app.command()
def dashboard() -> None:
    maybe_print_banner()
    print(json.dumps({"status": "stub", "message": "Streamlit dashboard not implemented yet"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
