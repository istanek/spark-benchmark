from __future__ import annotations

import curses
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import typer
from rich.console import Console
from rich.panel import Panel

from spark_benchmark.config import load_backend, load_experiment, load_model_config, load_platform
from spark_benchmark.model_registry import (
    DetectedOllamaModel,
    OllamaModelInfo,
    classify_detected,
    detect_ollama_models,
    is_embedding_model,
    is_vision_model,
)
from spark_benchmark.models import BackendConfig, ExperimentSpec, ModelConfig, PlatformConfig
from spark_benchmark.orchestration import BenchmarkPlan, run_benchmark_bundle
from spark_benchmark.reporting import aggregate_runs, render_cli_benchmark_summary, write_report
from spark_benchmark.results_bundle import make_run_id
from spark_benchmark.runners.registry import build_backend


DEFAULT_EXPERIMENT = "spark-ollama-baseline.yaml"
DEFAULT_PLATFORM = "spark"

SUITE_REGISTRY: dict[str, dict[str, str]] = {
    "openclaw_speed": {
        "label": "OpenClaw-like speed probe (TTFT + decode)",
        "data_path": "data/performance/openclaw_speed_v1.json",
    },
    "hallucination_grounding": {
        "label": "Grounding / hallucination reliability",
        "data_path": "data/reliability/hallucination_grounding_v1.json",
    },
    "practical_structured_output": {
        "label": "Structured output reliability (JSON exact match)",
        "data_path": "data/practical/practical_structured_output_v1.json",
    },
    "code_generation": {
        "label": "Code generation (HumanEval starter)",
        "data_path": "data/code/code_generation_v1.json",
    },
    "sustained_throughput": {
        "label": "Sustained throughput (5-min thermal / decode soak)",
        "data_path": "data/performance/sustained_throughput_v1.json",
    },
}

MENU_ITEMS: list[tuple[str, str]] = [
    ("run", "Run"),
    ("models", "Models"),
    ("suites", "Suites"),
    ("info", "Info"),
    ("chat", "Chat"),
    ("refresh", "Refresh"),
    ("quit", "Quit"),
]

BANNER_LINES = [
    "",
    "  ██████╗  ██████╗ ██╗  ██╗    ███████╗██████╗  █████╗ ██████╗ ██╗  ██╗",
    "  ██╔══██╗██╔════╝ ╚██╗██╔╝    ██╔════╝██╔══██╗██╔══██╗██╔══██╗██║ ██╔╝",
    "  ██║  ██║██║  ███╗ ╚███╔╝     ███████╗██████╔╝███████║██████╔╝█████╔╝ ",
    "  ██║  ██║██║   ██║ ██╔██╗     ╚════██║██╔═══╝ ██╔══██║██╔══██╗██╔═██╗ ",
    "  ██████╔╝╚██████╔╝██╔╝ ██╗    ███████║██║     ██║  ██║██║  ██║██║  ██╗",
    "  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝",
    "                       spark-benchmark // local model trials",
    "",
]

console = Console()


@dataclass
class ShellContext:
    repo_root: Path
    experiment: ExperimentSpec
    platform: PlatformConfig
    backend_config: BackendConfig
    model_configs: list[ModelConfig]


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_default_context() -> ShellContext:
    repo_root = _resolve_repo_root()
    experiment_path = repo_root / "configs" / "experiments" / DEFAULT_EXPERIMENT
    platform_path = repo_root / "configs" / "platforms" / f"{DEFAULT_PLATFORM}.yaml"
    experiment_file = load_experiment(experiment_path)
    experiment_spec = experiment_file.experiment
    platform_config = load_platform(platform_path)
    backend_path = repo_root / "configs" / "backends" / f"{experiment_spec.backend.value}.yaml"
    backend_config = load_backend(backend_path)

    model_configs: list[ModelConfig] = []
    for model_name in experiment_spec.models:
        model_path = repo_root / "configs" / "models" / f"{model_name}.yaml"
        if model_path.exists():
            model_configs.append(load_model_config(model_path))

    return ShellContext(
        repo_root=repo_root,
        experiment=experiment_spec,
        platform=platform_config,
        backend_config=backend_config,
        model_configs=model_configs,
    )


def classify_models(
    ctx: ShellContext, detected: list[DetectedOllamaModel]
) -> list[OllamaModelInfo]:
    """Backwards-compatible wrapper around ``classify_detected``.

    Existing tests and callers pass a ``ShellContext``. Internally we just
    delegate to the shared registry so the curses TUI shares one
    classification path with ``cli.py``.
    """
    return classify_detected(ctx.model_configs, detected)


def load_suite_metadata(repo_root: Path, suite_name: str) -> dict[str, Any] | None:
    entry = SUITE_REGISTRY.get(suite_name)
    if not entry:
        return None
    path = repo_root / entry["data_path"]
    if not path.exists():
        return None
    return json.loads(path.read_text())


def chat_command(ctx: ShellContext, arg: str) -> None:
    """Readline-style chat with a single model. Runs outside the curses TUI."""
    arg = arg.strip()
    detected = detect_ollama_models(ctx.backend_config)
    classified = [item for item in classify_models(ctx, detected) if item.has_config]
    if not classified:
        console.print("[red]No ready models to chat with.[/red]")
        input("Press Enter to return...")
        return

    selected: ModelConfig | None = None
    if arg:
        for item in classified:
            if item.display_name == arg or item.tag == arg:
                selected = item.config
                break
        if selected is None:
            console.print(f"[red]Unknown model '{arg}'.[/red]")
            input("Press Enter to return...")
            return
    else:
        names = [f"{item.display_name}  [{item.tag}]" for item in classified]
        for index, name in enumerate(names, start=1):
            console.print(f"{index}. {name}")
        try:
            answer = typer.prompt("Pick a model (number)", default="1")
        except (EOFError, KeyboardInterrupt):
            return
        try:
            pick = int(answer)
        except ValueError:
            console.print("[red]Invalid choice.[/red]")
            input("Press Enter to return...")
            return
        if pick < 1 or pick > len(classified):
            console.print("[red]Out of range.[/red]")
            input("Press Enter to return...")
            return
        selected = classified[pick - 1].config

    backend = build_backend(ctx.backend_config)
    backend.load_model(selected)
    console.print(
        Panel.fit(
            f"model: {selected.name}\nbackend: {ctx.backend_config.name.value}\n\n"
            "Write a prompt and press Enter. Use /exit to return to the menu.",
            title="[bold white]Chat[/bold white]",
            border_style="#f59e0b",
        )
    )
    try:
        while True:
            try:
                prompt = typer.prompt("you", prompt_suffix=" > ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in {"/exit", "/quit"}:
                break
            if not prompt.strip():
                continue
            result = backend.generate(prompt, ctx.experiment.sampling)
            console.print(
                Panel(
                    result.output,
                    title=f"[bold #7dd3fc]{selected.name}[/bold #7dd3fc]",
                    border_style="#38bdf8",
                )
            )
    finally:
        backend.unload()


# --- curses TUI --------------------------------------------------------------


def _init_colors() -> None:
    if not curses.has_colors():
        return
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)     # banner / accents
        curses.init_pair(2, curses.COLOR_YELLOW, -1)   # menu cursor
        curses.init_pair(3, curses.COLOR_GREEN, -1)    # success / selected
        curses.init_pair(4, 8, -1)                     # dim / disabled
        curses.init_pair(5, curses.COLOR_RED, -1)      # errors
        curses.init_pair(6, curses.COLOR_BLUE, -1)     # divider
    except curses.error:
        pass


def _safe_addnstr(window: Any, y: int, x: int, text: str, max_w: int, attr: int = 0) -> None:
    if max_w <= 0:
        return
    try:
        window.addnstr(y, x, text, max_w, attr)
    except curses.error:
        # writing into the last cell raises on some terminals; ignore
        pass


def _curses_multiselect(
    stdscr: Any,
    title: str,
    options: list[str],
    preselected: set[int] | None = None,
    disabled: set[int] | None = None,
    header_lines: list[str] | None = None,
) -> list[int] | None:
    """Multiselect overlay using an active curses screen. Returns None on cancel."""
    preselected = set() if preselected is None else set(preselected)
    disabled = set() if disabled is None else set(disabled)
    header_lines = [] if header_lines is None else list(header_lines)
    enabled = [idx for idx in range(len(options)) if idx not in disabled]
    if not enabled:
        return None
    cursor = enabled[0]
    selected = {idx for idx in preselected if idx not in disabled}

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        max_w = max(1, width - 1)
        row = 0
        for line in header_lines:
            if row >= height:
                break
            _safe_addnstr(stdscr, row, 0, line, max_w, curses.color_pair(1) | curses.A_BOLD)
            row += 1
        if row < height:
            row += 1
        if row < height:
            _safe_addnstr(stdscr, row, 0, title, max_w, curses.A_BOLD)
            row += 1
        if row < height:
            _safe_addnstr(
                stdscr, row, 0,
                "↑/↓ move • Space toggle • Enter confirm • Esc/q cancel",
                max_w, curses.A_DIM,
            )
            row += 1
        if row < height:
            row += 1

        visible_height = max(1, height - row - 1)
        start = 0
        if cursor >= start + visible_height:
            start = cursor - visible_height + 1

        for idx in range(start, min(len(options), start + visible_height)):
            is_disabled = idx in disabled
            is_selected = idx in selected
            if is_disabled:
                marker = "·"
            else:
                marker = "●" if is_selected else "○"
            line = f"  {marker} {options[idx]}"
            attr = curses.A_REVERSE if idx == cursor else curses.A_NORMAL
            if is_disabled:
                attr |= curses.color_pair(4) | curses.A_DIM
            elif is_selected:
                attr |= curses.color_pair(3)
            _safe_addnstr(stdscr, row + idx - start, 0, line, max_w, attr)

        stdscr.refresh()
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % len(options)
        elif key == ord(" "):
            if cursor in disabled:
                continue
            if cursor in selected:
                selected.remove(cursor)
            else:
                selected.add(cursor)
        elif key in (10, 13, curses.KEY_ENTER):
            return sorted(selected)
        elif key in (27, ord("q")):
            return None


def _curses_singleselect(
    stdscr: Any,
    title: str,
    options: list[str],
    header_lines: list[str] | None = None,
) -> int | None:
    if not options:
        return None
    header_lines = [] if header_lines is None else list(header_lines)
    cursor = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        max_w = max(1, width - 1)
        row = 0
        for line in header_lines:
            if row >= height:
                break
            _safe_addnstr(stdscr, row, 0, line, max_w, curses.color_pair(1) | curses.A_BOLD)
            row += 1
        if row < height:
            row += 1
        if row < height:
            _safe_addnstr(stdscr, row, 0, title, max_w, curses.A_BOLD)
            row += 1
        if row < height:
            _safe_addnstr(
                stdscr, row, 0, "↑/↓ move • Enter select • Esc/q cancel",
                max_w, curses.A_DIM,
            )
            row += 1
        if row < height:
            row += 1
        visible_height = max(1, height - row - 1)
        start = 0
        if cursor >= start + visible_height:
            start = cursor - visible_height + 1
        for idx in range(start, min(len(options), start + visible_height)):
            attr = curses.A_REVERSE if idx == cursor else curses.A_NORMAL
            _safe_addnstr(stdscr, row + idx - start, 0, f"  {options[idx]}", max_w, attr)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % len(options)
        elif key in (10, 13, curses.KEY_ENTER):
            return cursor
        elif key in (27, ord("q")):
            return None


@dataclass
class TUIApp:
    ctx: ShellContext
    cursor: int = 0
    log_lines: list[str] = field(default_factory=list)
    scroll_offset: int = 0  # 0 = follow bottom; >0 = N lines back
    running: bool = True
    status_mode: str = "menu"  # "menu" or "ack"

    LOG_CAP: int = 2000

    def log(self, message: Any) -> None:
        text = message if isinstance(message, str) else str(message)
        for line in (text.splitlines() or [""]):
            self.log_lines.append(line)
        if len(self.log_lines) > self.LOG_CAP:
            self.log_lines = self.log_lines[-self.LOG_CAP :]

    def log_blank(self) -> None:
        self.log_lines.append("")

    def clear_log(self) -> None:
        self.log_lines = []
        self.scroll_offset = 0

    def log_renderable(self, renderable: Any) -> None:
        if isinstance(renderable, str):
            self.log(renderable)
            return
        buf = io.StringIO()
        Console(file=buf, force_terminal=False, width=140, color_system=None).print(renderable)
        self.log(buf.getvalue().rstrip("\n"))

    # --- drawing -----------------------------------------------------------

    def draw(self, stdscr: Any) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        max_w = max(1, width - 1)
        row = 0
        for line in BANNER_LINES:
            if row >= height:
                stdscr.refresh()
                return
            _safe_addnstr(stdscr, row, 0, line, max_w, curses.color_pair(1) | curses.A_BOLD)
            row += 1
        # menu row
        if row < height:
            x = 0
            for idx, (_key, label) in enumerate(MENU_ITEMS):
                cell = f" {label} "
                if x + len(cell) >= max_w:
                    break
                if idx == self.cursor:
                    attr = curses.A_REVERSE | curses.A_BOLD | curses.color_pair(2)
                else:
                    attr = curses.A_NORMAL
                _safe_addnstr(stdscr, row, x, cell, max_w - x, attr)
                x += len(cell) + 1
            row += 1
        # divider
        if row < height:
            _safe_addnstr(stdscr, row, 0, "─" * max_w, max_w, curses.color_pair(6))
            row += 1
        log_top = row
        # status footer eats last row
        log_bottom = max(log_top + 1, height - 1)
        log_height = log_bottom - log_top
        total = len(self.log_lines)
        end = max(0, total - self.scroll_offset)
        start = max(0, end - log_height)
        visible = self.log_lines[start:end]
        for i, line in enumerate(visible):
            y = log_top + i
            if y >= log_bottom:
                break
            _safe_addnstr(stdscr, y, 0, line, max_w)
        # status line
        if height >= 1:
            scroll = "" if self.scroll_offset == 0 else f"  [scrolled +{self.scroll_offset}]"
            if self.status_mode == "ack":
                hint = f"Press Enter to return to menu • ↑/↓ scroll • PgUp/PgDn • End jump{scroll}"
                attr = curses.color_pair(2) | curses.A_BOLD
            else:
                tag = MENU_ITEMS[self.cursor][1]
                hint = f"← → menu • Enter ‹{tag}› • ↑/↓ scroll • PgUp/PgDn • End jump • q quit{scroll}"
                attr = curses.A_DIM
            _safe_addnstr(stdscr, height - 1, 0, hint, max_w, attr)
        stdscr.refresh()

    # --- main loop ---------------------------------------------------------

    def _wait_for_ack(self, stdscr: Any) -> None:
        """Wait for Enter/Esc/q so user can read output, then return to clean menu."""
        self.status_mode = "ack"
        try:
            while True:
                self.draw(stdscr)
                try:
                    key = stdscr.getch()
                except KeyboardInterrupt:
                    return
                if key in (10, 13, curses.KEY_ENTER, 27, ord("q")):
                    return
                if key == curses.KEY_UP:
                    self.scroll_offset = min(
                        self.scroll_offset + 1, max(0, len(self.log_lines) - 1)
                    )
                elif key == curses.KEY_DOWN:
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                elif key == curses.KEY_PPAGE:
                    self.scroll_offset = min(
                        self.scroll_offset + 10, max(0, len(self.log_lines) - 1)
                    )
                elif key == curses.KEY_NPAGE:
                    self.scroll_offset = max(0, self.scroll_offset - 10)
                elif key == curses.KEY_END:
                    self.scroll_offset = 0
        finally:
            self.status_mode = "menu"

    def run(self, stdscr: Any) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)
        _init_colors()
        while self.running:
            self.draw(stdscr)
            try:
                key = stdscr.getch()
            except KeyboardInterrupt:
                self.running = False
                break
            if key in (curses.KEY_LEFT, ord("h")):
                self.cursor = (self.cursor - 1) % len(MENU_ITEMS)
            elif key in (curses.KEY_RIGHT, ord("l"), 9):  # 9 = Tab
                self.cursor = (self.cursor + 1) % len(MENU_ITEMS)
            elif key == curses.KEY_UP:
                self.scroll_offset = min(self.scroll_offset + 1, max(0, len(self.log_lines) - 1))
            elif key == curses.KEY_DOWN:
                self.scroll_offset = max(0, self.scroll_offset - 1)
            elif key == curses.KEY_PPAGE:
                self.scroll_offset = min(self.scroll_offset + 10, max(0, len(self.log_lines) - 1))
            elif key == curses.KEY_NPAGE:
                self.scroll_offset = max(0, self.scroll_offset - 10)
            elif key == curses.KEY_END:
                self.scroll_offset = 0
            elif key in (10, 13, curses.KEY_ENTER):
                action = MENU_ITEMS[self.cursor][0]
                self.dispatch(stdscr, action)
            elif key == ord("q"):
                self.running = False

    # --- dispatch ----------------------------------------------------------

    def dispatch(self, stdscr: Any, action: str) -> None:
        self.clear_log()
        if action == "quit":
            self.running = False
            return
        try:
            if action == "refresh":
                self.ctx = load_default_context()
                self.log("Context reloaded.")
            elif action == "models":
                self.show_models()
            elif action == "suites":
                self.show_suites()
            elif action == "info":
                self.show_info(stdscr)
            elif action == "run":
                self.do_run(stdscr)
            elif action == "chat":
                self.do_chat(stdscr)
        except Exception as exc:  # noqa: BLE001
            self.log_blank()
            self.log(f"ERROR: {exc}")

        # Let the user read the output, then snap back to a clean main screen.
        if self.log_lines and self.running:
            self._wait_for_ack(stdscr)
        self.clear_log()

    # --- actions -----------------------------------------------------------

    def show_models(self) -> None:
        detected = detect_ollama_models(self.ctx.backend_config)
        classified = classify_models(self.ctx, detected)
        detected_tags = {item.tag for item in detected}
        self.log_blank()
        self.log("── Models ──")
        if not classified:
            self.log("  (no models detected — is Ollama running?)")
            return
        for item in classified:
            if item.has_config and item.auto_detected:
                status = "ready (auto)"
            elif item.has_config and item.tag in detected_tags:
                status = "ready"
            elif item.has_config:
                status = "configured, not pulled"
            else:
                status = f"disabled — {item.disable_reason or 'no config'}"
            self.log(f"  {item.display_name:<28} {item.tag:<28} {status}")

    def show_suites(self) -> None:
        self.log_blank()
        self.log("── Suites ──")
        for name, entry in SUITE_REGISTRY.items():
            meta = load_suite_metadata(self.ctx.repo_root, name)
            task_count = "?" if meta is None else str(len(meta.get("tasks") or []))
            description = (meta.get("description") if meta else "") or entry["label"]
            self.log(f"  {name:<32} tasks={task_count:>3}  {description}")

    def show_info(self, stdscr: Any) -> None:
        suite_keys = list(SUITE_REGISTRY)
        labels = [f"{k}  —  {SUITE_REGISTRY[k]['label']}" for k in suite_keys]
        choice = _curses_singleselect(
            stdscr, "Pick a suite to inspect:", labels, header_lines=BANNER_LINES,
        )
        if choice is None:
            return
        canonical = suite_keys[choice]
        meta = load_suite_metadata(self.ctx.repo_root, canonical)
        self.log_blank()
        self.log(f"── info: {canonical} ──")
        if meta is None:
            self.log("  (suite data missing)")
            return
        self.log(f"  Name:        {meta.get('name', canonical)}")
        self.log(f"  Category:    {meta.get('category', '?')}")
        self.log(f"  Version:     {meta.get('version', '?')}")
        self.log(f"  Description: {meta.get('description', '')}")
        for note in (meta.get("notes") or []):
            self.log(f"    • {note}")
        tasks = meta.get("tasks") or []
        self.log(f"  Tasks: {len(tasks)}")
        for task in tasks[:3]:
            tid = task.get("task_id", "?")
            prompt = (task.get("prompt") or "").strip().replace("\n", " ")
            if len(prompt) > 100:
                prompt = prompt[:97] + "..."
            self.log(f"    {tid}: {prompt}")
        if len(tasks) > 3:
            self.log(f"    ...and {len(tasks) - 3} more")

    def do_run(self, stdscr: Any) -> None:
        detected = detect_ollama_models(self.ctx.backend_config)
        if not detected:
            self.log_blank()
            self.log("No models detected via Ollama. Is the daemon running?")
            return
        classified = classify_models(self.ctx, detected)

        model_labels: list[str] = []
        disabled: set[int] = set()
        defaults: set[int] = set()
        for idx, item in enumerate(classified):
            if item.has_config and item.auto_detected:
                model_labels.append(f"{item.tag}  (auto)")
                defaults.add(idx)
            elif item.has_config:
                model_labels.append(f"{item.display_name}  [{item.tag}]")
                defaults.add(idx)
            else:
                reason = item.disable_reason or "no YAML config"
                model_labels.append(f"{item.tag}  ({reason})")
                disabled.add(idx)

        picked_models = _curses_multiselect(
            stdscr,
            "Select models to benchmark:",
            model_labels,
            preselected=defaults,
            disabled=disabled,
            header_lines=BANNER_LINES,
        )
        if not picked_models:
            self.log_blank()
            self.log("(no models selected)")
            return
        selected_configs = [
            classified[i].config for i in picked_models if classified[i].has_config
        ]
        if not selected_configs:
            self.log_blank()
            self.log("(no usable models selected)")
            return

        suite_keys = list(SUITE_REGISTRY)
        suite_labels = [f"{k}  —  {SUITE_REGISTRY[k]['label']}" for k in suite_keys]
        picked_suites = _curses_multiselect(
            stdscr,
            "Select test suites to run:",
            suite_labels,
            preselected=set(range(len(suite_keys))),
            header_lines=BANNER_LINES,
        )
        if not picked_suites:
            self.log_blank()
            self.log("(no suites selected)")
            return
        selected_suites = [suite_keys[i] for i in picked_suites]

        self.log_blank()
        self.log("=== Benchmark run ===")
        self.log(f"Models: {', '.join(m.name for m in selected_configs)}")
        self.log(f"Suites: {', '.join(selected_suites)}")
        self.log_blank()
        self.draw(stdscr)

        plan = BenchmarkPlan(
            request="interactive shell selection",
            selected_models=[m.name for m in selected_configs],
            selected_suites=selected_suites,
            rationale=["Interactive shell selection."],
        )
        bundle_dir = self.ctx.repo_root / "results" / "benchmarks" / make_run_id()
        backend = build_backend(self.ctx.backend_config)

        def progress(message: str) -> None:
            self.log(message)
            self.draw(stdscr)

        run_benchmark_bundle(
            bundle_dir=bundle_dir,
            repo_root=self.ctx.repo_root,
            experiment=self.ctx.experiment,
            platform_config=self.ctx.platform,
            backend_config=self.ctx.backend_config,
            model_configs=selected_configs,
            backend=backend,
            plan=plan,
            progress_callback=progress,
        )
        aggregate = aggregate_runs(bundle_dir)
        report_path = bundle_dir / "report.md"
        write_report(report_path, "markdown", aggregate)
        summary = render_cli_benchmark_summary(
            request="interactive shell selection",
            selected_models=plan.selected_models,
            selected_suites=plan.selected_suites,
            aggregate=aggregate,
            report_path=report_path,
        )
        self.log_blank()
        self.log_renderable(summary)
        self.log_blank()
        self.log(f"Done. Results in {bundle_dir}")

    def do_chat(self, stdscr: Any) -> None:
        # Chat runs outside the TUI; the user exits it explicitly with /exit,
        # so we skip the post-action ack prompt by leaving the log empty.
        curses.endwin()
        try:
            chat_command(self.ctx, "")
        except KeyboardInterrupt:
            pass
        stdscr.clear()
        stdscr.refresh()
        self.clear_log()


def run_shell() -> None:
    try:
        ctx = load_default_context()
    except Exception as exc:
        console.print(f"[red]Failed to load default context:[/red] {exc}")
        sys.exit(1)

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print("[red]The interactive shell needs a real terminal.[/red]")
        sys.exit(1)

    app = TUIApp(ctx=ctx)
    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        pass
    console.print("[#a3e635]bye.[/#a3e635]")
