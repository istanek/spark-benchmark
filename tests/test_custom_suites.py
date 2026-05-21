from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from spark_benchmark.custom_suites import (
    CustomSuiteDefinition,
    CustomSuiteTask,
    already_completed_pairs,
    build_custom_summary,
    load_custom_suite,
    render_custom_summary_markdown,
    run_custom_suite_quick,
    slugify_suite_name,
    validate_custom_suite,
)
from spark_benchmark.models import (
    BackendConfig,
    BackendKind,
    GenerationResult,
    InferenceMetrics,
    ModelConfig,
    SamplingConfig,
)


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


def _make_model(name: str, tag: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        family=name.split("-")[0],
        revision=tag,
        quantization="ollama-default",
        source="ollama-local",
        context_length=4096,
        artifact_path=tag,
    )


def _make_backend() -> BackendConfig:
    return BackendConfig(
        name=BackendKind.OLLAMA,
        entrypoint="ollama",
        version="local",
        transport="http",
        options={"endpoint": "http://localhost:11434/api/generate"},
    )


class _FakeBackend:
    """Backend stub that returns a deterministic GenerationResult per task.

    Records the order load_model / generate / unload were called in so the
    runner contract is explicit.
    """

    def __init__(self, replies: dict[tuple[str, str], str] | None = None,
                 raise_on: tuple[str, str] | None = None) -> None:
        self.replies = replies or {}
        self.raise_on = raise_on
        self._loaded: ModelConfig | None = None
        self.calls: list[tuple[str, str]] = []  # (event, value)

    def load_model(self, cfg: ModelConfig) -> None:
        self._loaded = cfg
        self.calls.append(("load", cfg.name))

    def unload(self) -> None:
        if self._loaded is not None:
            self.calls.append(("unload", self._loaded.name))
        self._loaded = None

    def generate(self, prompt: str, sampling: SamplingConfig) -> GenerationResult:
        assert self._loaded is not None
        # Tag the response with the prompt's first 20 chars so we can
        # reason about replies in tests without smuggling task IDs.
        tag = (self._loaded.name, prompt.strip().split("\n")[0][:32])
        self.calls.append(("generate", self._loaded.name))
        if self.raise_on == (self._loaded.name, prompt.strip().split("\n")[0][:32]):
            raise RuntimeError("simulated backend failure")
        canned = self.replies.get(tag, f"reply from {self._loaded.name} to {prompt[:40]!r}")
        return GenerationResult(
            prompt=prompt,
            output=canned,
            finish_reason="stop",
            metrics=InferenceMetrics(
                prefill_tokens=10,
                decode_tokens=20,
                prefill_time_s=0.1,
                decode_time_s=0.5,
                ttft_ms=120.0,
                peak_memory_mb=4096.0,
                backend_version="local",
                quantization="ollama-default",
            ),
        )


# --------------------------------------------------------------------- #
# Schema validation                                                     #
# --------------------------------------------------------------------- #


def test_custom_suite_rejects_duplicate_task_ids() -> None:
    try:
        CustomSuiteDefinition(
            name="dup",
            tasks=[
                CustomSuiteTask(task_id="a", prompt="x"),
                CustomSuiteTask(task_id="a", prompt="y"),
            ],
        )
    except Exception as exc:  # pydantic ValidationError wraps ValueError
        assert "duplicate" in str(exc).lower()
        return
    raise AssertionError("expected duplicate task_id to raise")


def test_custom_suite_rejects_empty_prompt() -> None:
    try:
        CustomSuiteDefinition(
            name="empty",
            tasks=[CustomSuiteTask(task_id="a", prompt="   ")],
        )
    except Exception as exc:
        assert "empty prompt" in str(exc).lower()
        return
    raise AssertionError("expected empty prompt to raise")


def test_custom_suite_rejects_empty_tasks() -> None:
    try:
        CustomSuiteDefinition(name="empty", tasks=[])
    except Exception as exc:
        assert "at least one task" in str(exc).lower()
        return
    raise AssertionError("expected empty tasks list to raise")


# --------------------------------------------------------------------- #
# Loader                                                                #
# --------------------------------------------------------------------- #


_YAML_VALID = """
name: my-test
version: "1.0"
mode: quick
tasks:
  - task_id: t1
    prompt: |
      Hello world.
"""


def test_load_custom_suite_from_yaml() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "suite.yaml"
        path.write_text(_YAML_VALID, encoding="utf-8")
        suite = load_custom_suite(path)
        assert suite.name == "my-test"
        assert suite.version == "1.0"
        assert suite.mode == "quick"
        assert len(suite.tasks) == 1
        assert suite.tasks[0].task_id == "t1"


def test_load_custom_suite_rejects_scored_mode() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "suite.yaml"
        path.write_text(_YAML_VALID.replace("mode: quick", "mode: scored"), encoding="utf-8")
        try:
            load_custom_suite(path)
        except ValueError as exc:
            assert "v0.3.0" in str(exc)
            return
    raise AssertionError("expected scored mode to be rejected today")


def test_load_custom_suite_from_json_round_trip() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "suite.json"
        path.write_text(
            json.dumps(
                {
                    "name": "json-suite",
                    "mode": "quick",
                    "tasks": [{"task_id": "t1", "prompt": "Hello."}],
                }
            ),
            encoding="utf-8",
        )
        suite = load_custom_suite(path)
        assert suite.name == "json-suite"


# --------------------------------------------------------------------- #
# validate_custom_suite (soft checks)                                   #
# --------------------------------------------------------------------- #


def test_validate_warns_on_long_prompt() -> None:
    suite = CustomSuiteDefinition(
        name="long",
        tasks=[CustomSuiteTask(task_id="t1", prompt="x" * 33_000)],
    )
    issues = validate_custom_suite(suite)
    assert any(issue.severity == "warning" for issue in issues)


def test_validate_errors_on_unknown_model_in_suite() -> None:
    suite = CustomSuiteDefinition(
        name="unknown-models",
        tasks=[CustomSuiteTask(task_id="t1", prompt="hi")],
        models=["nonexistent-model"],
    )
    issues = validate_custom_suite(suite, available_models=["qwen-3.6"])
    assert any(issue.severity == "error" and "unknown" in issue.message.lower() for issue in issues)


def test_validate_errors_on_bad_sampling() -> None:
    suite = CustomSuiteDefinition(
        name="bad-sampling",
        tasks=[
            CustomSuiteTask(
                task_id="t1",
                prompt="hi",
                sampling=SamplingConfig(temperature=5.0, max_tokens=128),
            )
        ],
    )
    issues = validate_custom_suite(suite)
    assert any("temperature=5.0" in issue.message for issue in issues)


# --------------------------------------------------------------------- #
# Runner                                                                #
# --------------------------------------------------------------------- #


def _suite_with_two_tasks() -> CustomSuiteDefinition:
    return CustomSuiteDefinition(
        name="my-test",
        version="0.1",
        mode="quick",
        tasks=[
            CustomSuiteTask(task_id="t1", prompt="Hello"),
            CustomSuiteTask(task_id="t2", prompt="World"),
        ],
        sampling=SamplingConfig(temperature=0.0, max_tokens=512, seed=42),
    )


def test_run_custom_suite_quick_writes_one_row_per_pair() -> None:
    suite = _suite_with_two_tasks()
    backend = _FakeBackend()
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        summary = run_custom_suite_quick(
            suite=suite,
            backend=backend,
            backend_config=_make_backend(),
            model_configs=[_make_model("qwen-3.6", "qwen3.6:35b"), _make_model("gemma-4", "gemma4:31b")],
            run_dir=run_dir,
            default_sampling=suite.sampling,
        )
        rows = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 4
        per_model = {bucket["model"]: bucket for bucket in summary["per_model"]}
        assert per_model["qwen-3.6"]["tasks_completed"] == 2
        assert per_model["gemma-4"]["tasks_completed"] == 2
        assert per_model["qwen-3.6"]["tasks_errored"] == 0
        # Markdown summary contains the side-by-side block per task.
        md = (run_dir / "summary.md").read_text(encoding="utf-8")
        assert "### Task `t1`" in md
        assert "### Task `t2`" in md
        # And the manifest of calls in the backend confirms load → generate → unload per model.
        events = [event for event, _ in backend.calls]
        assert events.count("load") == 2
        assert events.count("generate") == 4
        assert events.count("unload") == 2


def test_run_custom_suite_records_errors_without_aborting() -> None:
    suite = _suite_with_two_tasks()
    backend = _FakeBackend(raise_on=("qwen-3.6", "Hello"))
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        summary = run_custom_suite_quick(
            suite=suite,
            backend=backend,
            backend_config=_make_backend(),
            model_configs=[_make_model("qwen-3.6", "qwen3.6:35b")],
            run_dir=run_dir,
            default_sampling=suite.sampling,
        )
        per_model = {bucket["model"]: bucket for bucket in summary["per_model"]}
        assert per_model["qwen-3.6"]["tasks_completed"] == 1
        assert per_model["qwen-3.6"]["tasks_errored"] == 1
        # Error row was written, not silently dropped.
        rows = [
            json.loads(line)
            for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        errored = [row for row in rows if row.get("error") is not None]
        assert len(errored) == 1
        assert errored[0]["task_id"] == "t1"
        assert errored[0]["error"]["type"] == "RuntimeError"


def test_run_custom_suite_resume_skips_already_done_pairs() -> None:
    suite = _suite_with_two_tasks()
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # First pass: only run model A.
        backend1 = _FakeBackend()
        run_custom_suite_quick(
            suite=suite,
            backend=backend1,
            backend_config=_make_backend(),
            model_configs=[_make_model("qwen-3.6", "qwen3.6:35b")],
            run_dir=run_dir,
            default_sampling=suite.sampling,
        )
        first_lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(first_lines) == 2

        # Second pass: resume with model A still in the lineup; the runner
        # should skip both pairs and only re-run anything that wasn't already done.
        backend2 = _FakeBackend()
        run_custom_suite_quick(
            suite=suite,
            backend=backend2,
            backend_config=_make_backend(),
            model_configs=[_make_model("qwen-3.6", "qwen3.6:35b")],
            run_dir=run_dir,
            default_sampling=suite.sampling,
        )
        # JSONL grew by zero new rows because everything was resumed.
        second_lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(second_lines) == 2
        # Backend was loaded but no generate calls were issued.
        assert all(event != "generate" for event, _ in backend2.calls)


def test_already_completed_pairs_handles_missing_or_bad_lines() -> None:
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        assert already_completed_pairs(run_dir) == set()
        (run_dir / "results.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"model": "a", "task_id": "1"}),
                    "not-json",
                    json.dumps({"model": "b", "task_id": "2"}),
                    json.dumps({"task_id": "3"}),  # missing model
                ]
            ),
            encoding="utf-8",
        )
        assert already_completed_pairs(run_dir) == {("a", "1"), ("b", "2")}


# --------------------------------------------------------------------- #
# Summary rendering                                                     #
# --------------------------------------------------------------------- #


def test_build_custom_summary_aggregates_per_model_metrics() -> None:
    suite = _suite_with_two_tasks()
    rows = [
        {
            "suite": suite.name,
            "suite_version": suite.version,
            "mode": "quick",
            "model": "qwen-3.6",
            "model_tag": "qwen3.6:35b",
            "task_id": "t1",
            "prompt": "Hello",
            "tags": [],
            "generation": {
                "prompt": "Hello",
                "output": "Hi",
                "finish_reason": "stop",
                "metrics": {
                    "ttft_ms": 100.0,
                    "decode_time_s": 1.0,
                    "decode_tokens": 50,
                    "prefill_time_s": 0.1,
                },
                "raw": {},
            },
            "error": None,
        },
        {
            "suite": suite.name,
            "suite_version": suite.version,
            "mode": "quick",
            "model": "qwen-3.6",
            "model_tag": "qwen3.6:35b",
            "task_id": "t2",
            "prompt": "World",
            "tags": [],
            "generation": {
                "prompt": "World",
                "output": "Hello",
                "finish_reason": "stop",
                "metrics": {
                    "ttft_ms": 300.0,
                    "decode_time_s": 2.0,
                    "decode_tokens": 100,
                    "prefill_time_s": 0.1,
                },
                "raw": {},
            },
            "error": None,
        },
    ]
    summary = build_custom_summary(suite, rows, _make_backend())
    bucket = summary["per_model"][0]
    assert bucket["mean_ttft_ms"] == 200.0
    # decode_tps per row: row1 = 50/1.0 = 50, row2 = 100/2.0 = 50, mean = 50.0
    assert bucket["mean_decode_tps"] == 50.0
    assert bucket["total_decode_tokens"] == 150
    assert bucket["wall_time_s"] == 3.2  # 0.1 + 1.0 + 0.1 + 2.0


def test_render_custom_summary_markdown_has_telemetry_table_and_per_task_blocks() -> None:
    rows = [
        {
            "suite": "x",
            "suite_version": "0.1",
            "mode": "quick",
            "model": "qwen-3.6",
            "model_tag": "qwen3.6:35b",
            "task_id": "t1",
            "prompt": "Hello",
            "tags": [],
            "generation": {
                "prompt": "Hello",
                "output": "Hi.",
                "finish_reason": "stop",
                "metrics": {"ttft_ms": 100.0, "decode_time_s": 1.0, "decode_tokens": 50},
                "raw": {},
            },
            "error": None,
        }
    ]
    summary = build_custom_summary(_suite_with_two_tasks(), rows, _make_backend())
    md = render_custom_summary_markdown(summary)
    assert md.startswith("# Custom suite:")
    assert "## Per-model telemetry" in md
    assert "## Side-by-side outputs" in md
    assert "### Task `t1`" in md
    assert "**qwen-3.6**" in md
    assert "Hi." in md


# --------------------------------------------------------------------- #
# Slug helper                                                           #
# --------------------------------------------------------------------- #


def test_slugify_suite_name_strips_punctuation_and_spaces() -> None:
    assert slugify_suite_name("My Czech RAG Test!") == "my-czech-rag-test"
    assert slugify_suite_name("    ") == "custom-suite"
    assert slugify_suite_name("foo--bar__baz") == "foo-bar-baz"


# --------------------------------------------------------------------- #
# Plain-python entrypoint                                               #
# --------------------------------------------------------------------- #


def _run_all() -> int:
    import inspect
    import sys

    failures: list[str] = []
    module = sys.modules[__name__]
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"ok  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc!r}")
            print(f"FAIL {name}: {exc!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_all())
