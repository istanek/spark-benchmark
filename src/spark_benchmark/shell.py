from __future__ import annotations

import curses
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import typer
from rich.console import Console
from rich.panel import Panel

from spark_benchmark.config import load_backend, load_experiment, load_model_config, load_platform
from spark_benchmark.custom_suites import (
    CustomSuiteDefinition,
    load_custom_suite,
    run_custom_suite_quick,
    slugify_suite_name,
    validate_custom_suite,
)
from spark_benchmark.quick import (
    build_quick_suite,
    default_save_dir,
    save_quick_suite_as_yaml,
)
from spark_benchmark.model_registry import (
    DetectedOllamaModel,
    OllamaModelInfo,
    classify_detected,
    detect_ollama_models,
    find_config_by_name_or_tag,
    is_embedding_model,
    is_vision_model,
    resolve_runnable_models,
)
from spark_benchmark.models import BackendConfig, ExperimentSpec, ModelConfig, PlatformConfig
from spark_benchmark.orchestration import BenchmarkPlan, run_benchmark_bundle
from spark_benchmark.quant_sweep import enrich_with_quant_sweep
from spark_benchmark.reporting import aggregate_runs, render_cli_benchmark_summary, write_report
from spark_benchmark.results_bundle import make_run_id, write_json
from spark_benchmark.runners.registry import build_backend


DEFAULT_EXPERIMENT = "spark-ollama-baseline.yaml"
DEFAULT_PLATFORM = "spark"


def _model_label(item: "OllamaModelInfo") -> str:
    """Return a human-readable label for a model picker entry."""
    if not item.has_config:
        reason = item.disable_reason or "no config"
        return f"{item.tag}  ({reason})"
    if item.auto_detected:
        suffix = "cloud" if item.is_cloud else "auto"
        return f"{item.tag}  ({suffix})"
    return f"{item.display_name}  [{item.tag}]"

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
    "long_context_retrieval": {
        "label": "Long-context retrieval — full grid (needle-in-a-haystack, 4k–131k; ~1h/model)",
        "data_path": "data/long_context/long_context_retrieval_v1.json",
        # Needs public-domain corpora fetched first; see preflight below.
        "needs_haystacks": "data/long_context/haystacks",
    },
    "long_context_retrieval_fast": {
        "label": "Long-context retrieval — fast preview (4k/32k/131k, fewer cells; ~10min/model)",
        "data_path": "data/long_context/long_context_retrieval_v1.json",
        "needs_haystacks": "data/long_context/haystacks",
        "profile": "fast",
    },
}

MENU_ITEMS: list[tuple[str, str]] = [
    ("run", "Run"),
    ("custom", "Custom"),
    ("quick", "Quick"),
    ("models", "Models"),
    ("suites", "Suites"),
    ("info", "Info"),
    ("chat", "Chat"),
    ("cloud", "Cloud"),
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


def load_default_context(
    experiment_path: Path | None = None,
    platform_name: str | None = None,
) -> ShellContext:
    repo_root = _resolve_repo_root()
    resolved_experiment = experiment_path or (
        repo_root / "configs" / "experiments" / DEFAULT_EXPERIMENT
    )
    resolved_platform = platform_name or DEFAULT_PLATFORM
    platform_path = repo_root / "configs" / "platforms" / f"{resolved_platform}.yaml"
    experiment_file = load_experiment(resolved_experiment)
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
    meta = json.loads(path.read_text())
    # When a registry entry pins a named profile (e.g. the fast preview),
    # surface that profile's grid so task counts / Info reflect reality.
    profile = entry.get("profile")
    profiles = meta.get("profiles")
    if profile and isinstance(profiles, dict) and profile in profiles:
        meta = {**meta, "test_matrix": profiles[profile]}
    return meta


def missing_haystacks(repo_root: Path, suite_name: str) -> list[str]:
    """Repo-relative haystack texts a suite needs but that aren't fetched yet.

    Empty for suites that don't declare ``needs_haystacks``. The
    long_context suite deliberately git-ignores its (large) corpora, so a
    fresh checkout must run ``scripts/fetch_haystacks.sh`` first.
    """
    entry = SUITE_REGISTRY.get(suite_name) or {}
    if "needs_haystacks" not in entry:
        return []
    meta = load_suite_metadata(repo_root, suite_name)
    if not meta:
        return []
    missing: list[str] = []
    for spec in (meta.get("haystacks") or {}).values():
        text_file = spec.get("text_file")
        if text_file and not (repo_root / text_file).exists():
            missing.append(text_file)
    return missing


def _suite_task_count(meta: dict[str, Any]) -> int:
    """Tasks-per-model for a suite's fixture.

    Task-list suites report ``len(tasks)``; the grid-based
    ``long_context_retrieval`` reports ``lengths × depths × needles_per_cell``.
    """
    matrix = meta.get("test_matrix")
    if isinstance(matrix, dict):
        lengths = matrix.get("context_lengths_tokens") or []
        depths = matrix.get("depth_percentages") or []
        per_cell = matrix.get("needles_per_cell") or 0
        return len(lengths) * len(depths) * int(per_cell)
    return len(meta.get("tasks") or [])


@dataclass
class CustomSuiteCandidate:
    """A custom-suite YAML/JSON that the TUI is willing to surface."""

    path: Path
    origin: str  # "example" | "recent"
    last_run: str | None = None  # run-id from manifest.json, only for origin="recent"

    def label(self) -> str:
        try:
            display = self.path.relative_to(Path.cwd())
        except ValueError:
            display = self.path
        if self.origin == "recent" and self.last_run:
            return f"{self.path.name}  (last run {self.last_run})  [{display}]"
        return f"{self.path.name}  ({self.origin})  [{display}]"


def discover_custom_suites(repo_root: Path) -> list[CustomSuiteCandidate]:
    """Find custom-suite YAMLs the user is likely to want to re-run.

    Looks in two well-known places:

    - ``examples/custom-tests/**/suite.yaml`` — shipped templates,
    - ``results/custom/<slug>/<run-id>/manifest.json`` — pulls
      ``suite_path`` from each manifest and dedupes by absolute path,
      keeping the newest ``run-id`` per path.

    Returns the list ordered with examples first, then recent-runs
    (newest run-id first within that group).
    """
    candidates: list[CustomSuiteCandidate] = []

    examples_root = repo_root / "examples" / "custom-tests"
    if examples_root.is_dir():
        for path in sorted(examples_root.glob("**/suite.yaml")):
            candidates.append(CustomSuiteCandidate(path=path.resolve(), origin="example"))
        for path in sorted(examples_root.glob("**/suite.json")):
            candidates.append(CustomSuiteCandidate(path=path.resolve(), origin="example"))

    custom_runs_root = repo_root / "results" / "custom"
    recent: dict[Path, CustomSuiteCandidate] = {}
    if custom_runs_root.is_dir():
        manifests: list[tuple[str, Path]] = []
        for manifest_path in custom_runs_root.glob("*/*/manifest.json"):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            run_id = str(payload.get("run_id") or manifest_path.parent.name)
            suite_path_raw = payload.get("suite_path")
            if not suite_path_raw:
                continue
            suite_path = Path(suite_path_raw)
            if not suite_path.exists():
                continue
            manifests.append((run_id, suite_path.resolve()))
        # Sort newest run-id last so the dict keeps the newest one per suite.
        for run_id, suite_path in sorted(manifests, key=lambda x: x[0]):
            recent[suite_path] = CustomSuiteCandidate(
                path=suite_path, origin="recent", last_run=run_id
            )

    seen = {c.path for c in candidates}
    for path, candidate in sorted(
        recent.items(), key=lambda x: (x[1].last_run or ""), reverse=True
    ):
        if path in seen:
            continue
        candidates.append(candidate)

    return candidates


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
    _experiment_path: Path | None = field(default=None, repr=False)
    _platform_name: str | None = field(default=None, repr=False)
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
        # Without this, ncurses waits up to ESCDELAY (default ~1000 ms)
        # after a bare ESC to disambiguate it from the start of an escape
        # sequence (arrow keys, F-keys). Users hit ESC, see nothing react,
        # hit it again — looks like "ESC needs two presses to leave a
        # submenu". 25 ms is what vim/htop use; safe for keypad() input.
        try:
            curses.set_escdelay(25)
        except (AttributeError, curses.error):
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
                self.ctx = load_default_context(
                    experiment_path=self._experiment_path,
                    platform_name=self._platform_name,
                )
                self.log("Context reloaded.")
            elif action == "models":
                self.show_models()
            elif action == "suites":
                self.show_suites()
            elif action == "info":
                self.show_info(stdscr)
            elif action == "run":
                self.do_run(stdscr)
            elif action == "custom":
                self.do_custom(stdscr)
            elif action == "quick":
                self.do_quick(stdscr)
            elif action == "chat":
                self.do_chat(stdscr)
            elif action == "cloud":
                self.do_cloud(stdscr)
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
                status = "ready (cloud)" if item.is_cloud else "ready (auto)"
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
            task_count = "?" if meta is None else str(_suite_task_count(meta))
            description = (meta.get("description") if meta else "") or entry["label"]
            self.log(f"  {name:<32} tasks={task_count:>4}  {description}")

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
        matrix = meta.get("test_matrix")
        if isinstance(matrix, dict):
            self.log(
                f"  Grid:        {matrix.get('context_lengths_tokens')} × depths "
                f"{matrix.get('depth_percentages')} × {matrix.get('needles_per_cell')} needles/cell"
            )
            self.log(f"  Needles:     {len(meta.get('needles') or [])}")
            self.log(f"  Tasks/model: {_suite_task_count(meta)}")
            return
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
            model_labels.append(_model_label(item))
            if item.has_config:
                defaults.add(idx)
            else:
                disabled.add(idx)

        picked_models = _curses_multiselect(
            stdscr,
            "Select models to benchmark:",
            model_labels,
            preselected=defaults,
            disabled=disabled,
            header_lines=BANNER_LINES,
        )
        if picked_models is None:
            return  # Esc/q — clean cancel back to the menu, no notice
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
        # Long-context profiles are slow and mutually redundant, so they're
        # opt-in: everything else is preselected, the user ticks the one
        # long-context profile they want (if any).
        default_suites = {
            i for i, k in enumerate(suite_keys) if not k.startswith("long_context_retrieval")
        }
        picked_suites = _curses_multiselect(
            stdscr,
            "Select test suites to run:",
            suite_labels,
            preselected=default_suites,
            header_lines=BANNER_LINES,
        )
        if picked_suites is None:
            return  # Esc/q — clean cancel back to the menu, no notice
        if not picked_suites:
            self.log_blank()
            self.log("(no suites selected)")
            return
        selected_suites = [suite_keys[i] for i in picked_suites]

        # Preflight: drop suites whose corpora aren't fetched yet rather than
        # crashing mid-run with a FileNotFoundError.
        runnable_suites: list[str] = []
        for suite_name in selected_suites:
            missing = missing_haystacks(self.ctx.repo_root, suite_name)
            if missing:
                self.log_blank()
                self.log(f"⚠ Skipping '{suite_name}': missing haystack corpora.")
                self.log("  Run scripts/fetch_haystacks.sh once, then re-run.")
                for text_file in missing:
                    self.log(f"    missing: {text_file}")
                continue
            runnable_suites.append(suite_name)
        if not runnable_suites:
            self.log_blank()
            self.log("(no runnable suites — nothing to do)")
            return
        selected_suites = runnable_suites

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
        enrich_with_quant_sweep(aggregate, selected_configs, self.ctx.repo_root)
        report_path = bundle_dir / "report.md"
        report_html_path = bundle_dir / "report.html"
        write_report(report_path, "both", aggregate)
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
        self.log(f"HTML report: {report_html_path}")

    def do_custom(self, stdscr: Any) -> None:
        """Pick a custom (BYOT) suite YAML/JSON, validate it, and run Mode A."""
        candidates = discover_custom_suites(self.ctx.repo_root)
        if not candidates:
            self.log_blank()
            self.log("── Custom (BYOT) ──")
            self.log("  No custom suites found.")
            self.log("")
            self.log("  Drop a YAML at one of these locations and come back:")
            self.log("    examples/custom-tests/<name>/suite.yaml   (shipped layout)")
            self.log("    results/custom/<slug>/<run-id>/           (auto, after a run)")
            self.log("")
            self.log("  Template:   examples/custom-tests/quick/suite.yaml")
            self.log("  Spec:       docs/custom-tests-spec.md")
            return

        suite_choice = _curses_singleselect(
            stdscr,
            "Pick a custom suite to run:",
            [c.label() for c in candidates],
            header_lines=BANNER_LINES,
        )
        if suite_choice is None:
            return
        candidate = candidates[suite_choice]

        self.log_blank()
        self.log("── Custom (BYOT) ──")
        self.log(f"  Suite file: {candidate.path}")

        try:
            loaded = load_custom_suite(candidate.path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"  Failed to load suite: {exc}")
            return
        self.log(f"  Suite:      {loaded.name} (v{loaded.version}, mode={loaded.mode})")
        self.log(f"  Tasks:      {len(loaded.tasks)}")

        # Use the same model resolver run-custom uses on the CLI:
        # default to allow_auto_detected=True for custom suites.
        resolved = resolve_runnable_models(
            backend_config=self.ctx.backend_config,
            experiment_model_configs=self.ctx.model_configs,
            allow_auto_detected=True,
        )
        if not resolved.classified:
            self.log("  No models detected via Ollama. Is the daemon running?")
            return

        available_names = [m.name for m in resolved.configs]
        issues = validate_custom_suite(loaded, available_models=available_names)
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity != "error"]
        for issue in warnings:
            self.log(f"  WARN: {issue.render()}")
        if errors:
            for issue in errors:
                self.log(f"  ERROR: {issue.render()}")
            self.log("  → fix the suite file and try again.")
            return

        suite_defaults = self._suite_default_model_indices(loaded, resolved.classified)
        model_labels: list[str] = []
        disabled: set[int] = set()
        defaults: set[int] = set()
        for idx, item in enumerate(resolved.classified):
            model_labels.append(_model_label(item))
            if not item.has_config:
                disabled.add(idx)
            elif suite_defaults is None or idx in suite_defaults:
                defaults.add(idx)

        picked = _curses_multiselect(
            stdscr,
            "Select models for this custom run:",
            model_labels,
            preselected=defaults,
            disabled=disabled,
            header_lines=BANNER_LINES,
        )
        if picked is None:
            return  # Esc/q — clean cancel back to the menu, no notice
        if not picked:
            self.log_blank()
            self.log("(no models selected)")
            return
        selected_configs: list[ModelConfig] = []
        for idx in picked:
            cfg = resolved.classified[idx].config
            if cfg is not None and cfg not in selected_configs:
                selected_configs.append(cfg)
        if not selected_configs:
            self.log_blank()
            self.log("(no usable models selected)")
            return

        run_id = make_run_id()
        run_dir = (
            self.ctx.repo_root
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
                "suite_path": str(candidate.path),
                "experiment": self.ctx.experiment.name,
                "platform": self.ctx.platform.name,
                "backend": self.ctx.backend_config.name.value,
                "models": [cfg.name for cfg in selected_configs],
                "task_count": len(loaded.tasks),
                "mode": loaded.mode,
                "auto_detected_models": [
                    cfg.name
                    for cfg in resolved.auto_detected_configs
                    if cfg in selected_configs
                ],
                "source": "shell",
            },
        )

        self.log_blank()
        self.log("=== Custom run ===")
        self.log(f"Models:   {', '.join(cfg.name for cfg in selected_configs)}")
        self.log(f"Run dir:  {run_dir}")
        self.log_blank()
        self.draw(stdscr)

        backend = build_backend(self.ctx.backend_config)

        def progress(message: str) -> None:
            self.log(message)
            self.draw(stdscr)

        try:
            run_custom_suite_quick(
                suite=loaded,
                backend=backend,
                backend_config=self.ctx.backend_config,
                model_configs=selected_configs,
                run_dir=run_dir,
                default_sampling=self.ctx.experiment.sampling,
                progress_callback=progress,
                resume=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_blank()
            self.log(f"Run failed mid-flight: {exc}")
            self.log(f"Partial results: {run_dir / 'results.jsonl'}")
            return

        summary_md = run_dir / "summary.md"
        summary_html = run_dir / "summary.html"
        summary_json = run_dir / "summary.json"
        self.log_blank()
        self.log("Done.")
        self.log(f"  results:  {run_dir / 'results.jsonl'}")
        self.log(f"  summary:  {summary_md}")
        self.log(f"  html:     {summary_html}")
        self.log(f"  json:     {summary_json}")
        if loaded.mode == "scored":
            self.log_blank()
            self.log("  Scoring results:")
            for bucket in summary.get("per_model", []):
                scored = bucket.get("scored", 0)
                if scored:
                    pr = bucket.get("pass_rate") or 0
                    pct = f"{100 * pr:.1f} %"
                    bar_len = int(pr * 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    self.log(f"    {bucket['model']:30s}  {bucket['passes']}/{scored}  [{bar}] {pct}")
                else:
                    self.log(f"    {bucket['model']:30s}  no scored tasks")

    @staticmethod
    def _suite_default_model_indices(
        suite: CustomSuiteDefinition, classified: list[OllamaModelInfo]
    ) -> set[int] | None:
        """Translate ``suite.models`` into indices into ``classified``.

        Returns ``None`` when the suite did not declare a ``models:`` list,
        which means the TUI should fall back to its own default
        (preselect everything that has a config).
        """
        if not suite.models:
            return None
        configs = [item.config for item in classified if item.has_config and item.config]
        wanted: set[int] = set()
        for raw in suite.models:
            cfg = find_config_by_name_or_tag(raw, configs=configs, classified=classified)
            if cfg is None:
                continue
            for idx, item in enumerate(classified):
                if item.config is cfg:
                    wanted.add(idx)
                    break
        return wanted

    def do_quick(self, stdscr: Any) -> None:
        """Fan out a single ad-hoc prompt to every selected model.

        BYOT Mode A's lightest entry point — no YAML required up front.
        Mirrors ``spark-bench quick "..."`` on the CLI but adds an
        interactive prompt input step and a post-run "save?" prompt.
        """
        resolved = resolve_runnable_models(
            backend_config=self.ctx.backend_config,
            experiment_model_configs=self.ctx.model_configs,
            allow_auto_detected=True,
        )
        if not resolved.classified:
            self.log_blank()
            self.log("── Quick (ad-hoc prompt) ──")
            self.log("  No models detected via Ollama. Is the daemon running?")
            return

        model_labels: list[str] = []
        disabled: set[int] = set()
        defaults: set[int] = set()
        for idx, item in enumerate(resolved.classified):
            model_labels.append(_model_label(item))
            if not item.has_config:
                disabled.add(idx)
            else:
                defaults.add(idx)

        picked = _curses_multiselect(
            stdscr,
            "Select models for the quick prompt:",
            model_labels,
            preselected=defaults,
            disabled=disabled,
            header_lines=BANNER_LINES,
        )
        if picked is None:
            return  # Esc/q — clean cancel back to the menu, no notice
        if not picked:
            self.log_blank()
            self.log("── Quick (ad-hoc prompt) ──")
            self.log("(no models selected)")
            return
        selected_configs: list[ModelConfig] = []
        for idx in picked:
            cfg = resolved.classified[idx].config
            if cfg is not None and cfg not in selected_configs:
                selected_configs.append(cfg)
        if not selected_configs:
            self.log_blank()
            self.log("(no usable models selected)")
            return

        # Drop out of curses to read the prompt on the regular TTY.
        curses.endwin()
        print()
        print("=== Quick prompt ===")
        print("Type a single line and press Enter. Empty input cancels.")
        print("(For multi-line prompts, save them as a YAML and use Custom.)")
        try:
            prompt_text = typer.prompt("prompt", default="", show_default=False).strip()
        except (EOFError, KeyboardInterrupt):
            prompt_text = ""
        # Hand the screen back to curses for the run.
        stdscr.clear()
        stdscr.refresh()
        if not prompt_text:
            self.log_blank()
            self.log("── Quick (ad-hoc prompt) ──")
            self.log("(cancelled — empty prompt)")
            return

        try:
            suite = build_quick_suite(prompt_text, sampling=self.ctx.experiment.sampling)
        except ValueError as exc:
            self.log_blank()
            self.log(f"Failed to build quick suite: {exc}")
            return

        run_id = make_run_id()
        run_dir = (
            self.ctx.repo_root
            / "results"
            / "custom"
            / slugify_suite_name(suite.name)
            / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "kind": "custom",
            "run_id": run_id,
            "suite": suite.name,
            "suite_version": suite.version,
            # Filled in if the user opts to save the prompt after the run.
            "suite_path": None,
            "experiment": self.ctx.experiment.name,
            "platform": self.ctx.platform.name,
            "backend": self.ctx.backend_config.name.value,
            "models": [cfg.name for cfg in selected_configs],
            "task_count": len(suite.tasks),
            "mode": suite.mode,
            "auto_detected_models": [
                cfg.name
                for cfg in resolved.auto_detected_configs
                if cfg in selected_configs
            ],
            "source": "shell-quick",
            "ad_hoc_prompt": True,
        }
        write_json(run_dir / "manifest.json", manifest)

        self.log_blank()
        self.log("── Quick (ad-hoc prompt) ──")
        self.log(f"  Suite name: {suite.name}")
        self.log(f"  Models:     {', '.join(cfg.name for cfg in selected_configs)}")
        self.log(f"  Run dir:    {run_dir}")
        self.log_blank()
        # Show the prompt itself, truncated, so the user can confirm it's
        # what they typed before the slow part starts.
        preview = prompt_text if len(prompt_text) <= 200 else prompt_text[:197] + "..."
        self.log(f"  Prompt:     {preview}")
        self.log_blank()
        self.draw(stdscr)

        backend = build_backend(self.ctx.backend_config)

        def progress(message: str) -> None:
            self.log(message)
            self.draw(stdscr)

        try:
            run_custom_suite_quick(
                suite=suite,
                backend=backend,
                backend_config=self.ctx.backend_config,
                model_configs=selected_configs,
                run_dir=run_dir,
                default_sampling=self.ctx.experiment.sampling,
                progress_callback=progress,
                resume=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_blank()
            self.log(f"Run failed mid-flight: {exc}")
            self.log(f"Partial results: {run_dir / 'results.jsonl'}")
            return

        self.log_blank()
        self.log("Done.")
        self.log(f"  results:  {run_dir / 'results.jsonl'}")
        self.log(f"  summary:  {run_dir / 'summary.md'}")
        self.log(f"  html:     {run_dir / 'summary.html'}")
        self.log(f"  json:     {run_dir / 'summary.json'}")

        # Post-run "save?" prompt — outside curses so prompt/confirm work.
        curses.endwin()
        print()
        try:
            wants_save = typer.confirm(
                "Save this prompt as a reusable custom suite?", default=False
            )
        except (EOFError, KeyboardInterrupt):
            wants_save = False
        saved_path: Path | None = None
        if wants_save:
            try:
                default_name = suite.name
                save_name = typer.prompt(
                    "Suite name", default=default_name, show_default=True
                ).strip() or default_name
                save_root = default_save_dir(self.ctx.repo_root)
                saved_path = save_quick_suite_as_yaml(
                    suite, save_root=save_root, name=save_name
                )
                print(f"saved: {saved_path}")
                # Patch the manifest so discover_custom_suites can surface it.
                manifest["suite_path"] = str(saved_path)
                write_json(run_dir / "manifest.json", manifest)
            except FileExistsError as exc:
                print(
                    f"a suite already exists at {exc}; not overwriting. "
                    "Re-run save manually with `spark-bench quick ... --save --overwrite`."
                )
            except Exception as exc:  # noqa: BLE001
                print(f"save failed: {exc}")
        stdscr.clear()
        stdscr.refresh()
        if saved_path is not None:
            self.log_blank()
            self.log(f"Saved as: {saved_path}")

    def do_cloud(self, stdscr: Any) -> None:
        """Prompt for an Ollama Cloud API key and re-probe models."""
        curses.endwin()
        print()
        print("=== Ollama Cloud ===")
        current_key = os.environ.get("OLLAMA_API_KEY", "")
        if current_key:
            masked = current_key[:6] + "..." + current_key[-4:] if len(current_key) > 10 else "***"
            print(f"Current key : {masked}  (Enter to keep, new value to replace, '-' to clear)")
        else:
            print("No API key set. Enter your Ollama Cloud key (or press Enter to cancel).")
        try:
            raw = input("API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        stdscr.clear()
        stdscr.refresh()
        self.clear_log()
        self.log_blank()
        self.log("── Cloud API key ──")
        if raw == "-":
            os.environ.pop("OLLAMA_API_KEY", None)
            self.log("API key cleared. Cloud models will not be detected.")
        elif raw == "":
            if current_key:
                self.log("Key unchanged.")
            else:
                self.log("(cancelled)")
            return
        else:
            os.environ["OLLAMA_API_KEY"] = raw
            self.log("API key set.")
        self.log("Re-probing Ollama (local + cloud)…")
        try:
            self.ctx = load_default_context(
                experiment_path=self._experiment_path,
                platform_name=self._platform_name,
            )
            self.log("Done — cloud models are now visible in the model picker.")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Reload failed: {exc}")

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


def run_shell(
    experiment_path: Path | None = None,
    platform_name: str | None = None,
) -> None:
    try:
        ctx = load_default_context(
            experiment_path=experiment_path,
            platform_name=platform_name,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load default context:[/red] {exc}")
        sys.exit(1)

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print("[red]The interactive shell needs a real terminal.[/red]")
        sys.exit(1)

    app = TUIApp(
        ctx=ctx,
        _experiment_path=experiment_path,
        _platform_name=platform_name,
    )
    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        pass
    console.print("[#a3e635]bye.[/#a3e635]")
