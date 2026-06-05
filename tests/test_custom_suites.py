from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from spark_benchmark.custom_suites import (
    CustomSuiteDefinition,
    CustomSuiteTask,
    ScoreResult,
    ScoringConfig,
    already_completed_pairs,
    build_custom_summary,
    load_custom_suite,
    render_custom_summary_markdown,
    run_custom_suite_quick,
    score_response,
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


def test_load_custom_suite_accepts_scored_mode() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "suite.yaml"
        path.write_text(_YAML_VALID.replace("mode: quick", "mode: scored"), encoding="utf-8")
        suite = load_custom_suite(path)
        assert suite.mode == "scored"


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


def test_render_custom_summary_markdown_uses_per_model_summary_header() -> None:
    """The heading changes depending on mode."""
    rows: list[dict] = []
    suite_quick = _suite_with_two_tasks()
    md_quick = render_custom_summary_markdown(build_custom_summary(suite_quick, rows, _make_backend()))
    assert "## Per-model summary" in md_quick
    assert "quick (no scoring)" in md_quick

    suite_scored = CustomSuiteDefinition(
        name="scored-test",
        mode="scored",
        tasks=[CustomSuiteTask(task_id="t1", prompt="hello")],
    )
    md_scored = render_custom_summary_markdown(build_custom_summary(suite_scored, rows, _make_backend()))
    assert "scored (deterministic scorers)" in md_scored
    assert "Pass rate" in md_scored


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
            "score": None,
        }
    ]
    summary = build_custom_summary(_suite_with_two_tasks(), rows, _make_backend())
    md = render_custom_summary_markdown(summary)
    assert md.startswith("# Custom suite:")
    assert "## Per-model summary" in md
    assert "## Side-by-side outputs" in md
    assert "### Task `t1`" in md
    assert "**qwen-3.6**" in md
    assert "Hi." in md


# --------------------------------------------------------------------- #
# ScoringConfig schema                                                  #
# --------------------------------------------------------------------- #


def test_scoring_config_rejects_exact_match_without_expected() -> None:
    try:
        ScoringConfig(method="exact_match")
    except Exception as exc:
        assert "expected" in str(exc).lower()
        return
    raise AssertionError("expected exact_match without 'expected' to raise")


def test_scoring_config_rejects_substring_match_empty_list() -> None:
    try:
        ScoringConfig(method="substring_match", must_contain=[])
    except Exception as exc:
        assert "must_contain" in str(exc).lower()
        return
    raise AssertionError("expected empty must_contain to raise")


def test_scoring_config_rejects_regex_without_pattern() -> None:
    try:
        ScoringConfig(method="regex_match")
    except Exception as exc:
        assert "pattern" in str(exc).lower()
        return
    raise AssertionError("expected regex_match without 'pattern' to raise")


def test_scoring_config_rejects_json_fields_empty_dict() -> None:
    try:
        ScoringConfig(method="json_fields_match", expected_fields={})
    except Exception as exc:
        assert "expected_fields" in str(exc).lower()
        return
    raise AssertionError("expected empty expected_fields to raise")


def test_scoring_config_rejects_multiple_choice_without_expected() -> None:
    try:
        ScoringConfig(method="multiple_choice")
    except Exception as exc:
        assert "expected" in str(exc).lower()
        return
    raise AssertionError("expected multiple_choice without 'expected' to raise")


# --------------------------------------------------------------------- #
# score_response                                                        #
# --------------------------------------------------------------------- #


def test_score_exact_match_pass_and_fail() -> None:
    sc = ScoringConfig(method="exact_match", expected="Paris")
    assert score_response("Paris", sc).passed is True
    assert score_response("paris", sc).passed is True  # case-insensitive default
    assert score_response("  Paris  ", sc).passed is True  # whitespace stripped
    assert score_response("Not Paris", sc).passed is False


def test_score_exact_match_case_sensitive() -> None:
    sc = ScoringConfig(method="exact_match", expected="Paris", case_sensitive=True)
    assert score_response("Paris", sc).passed is True
    assert score_response("paris", sc).passed is False


def test_score_substring_match_all_required() -> None:
    sc = ScoringConfig(method="substring_match", must_contain=["alpha", "beta"])
    assert score_response("alpha and beta are here", sc).passed is True
    assert score_response("alpha is here", sc).passed is False
    result = score_response("nothing", sc)
    assert result.passed is False
    assert "alpha" in result.reason or "beta" in result.reason


def test_score_regex_match_pass_and_fail() -> None:
    sc = ScoringConfig(method="regex_match", pattern=r"\d{4}")
    assert score_response("the year is 2024", sc).passed is True
    assert score_response("no number", sc).passed is False


def test_score_regex_match_invalid_pattern() -> None:
    sc = ScoringConfig(method="regex_match", pattern=r"[invalid")
    result = score_response("anything", sc)
    assert result.passed is False
    assert "invalid regex" in result.reason


def test_score_json_fields_match_pass() -> None:
    sc = ScoringConfig(method="json_fields_match", expected_fields={"city": "Prague", "country": "Czechia"})
    output = '{"city": "Prague", "country": "Czechia", "extra": 1}'
    assert score_response(output, sc).passed is True


def test_score_json_fields_match_with_markdown_fence() -> None:
    sc = ScoringConfig(method="json_fields_match", expected_fields={"key": "value"})
    output = '```json\n{"key": "value"}\n```'
    assert score_response(output, sc).passed is True


def test_score_json_fields_match_missing_key() -> None:
    sc = ScoringConfig(method="json_fields_match", expected_fields={"city": "Prague"})
    output = '{"country": "Czechia"}'
    result = score_response(output, sc)
    assert result.passed is False
    assert "missing key" in result.reason


def test_score_json_fields_match_wrong_value() -> None:
    sc = ScoringConfig(method="json_fields_match", expected_fields={"city": "Prague"})
    output = '{"city": "Brno"}'
    result = score_response(output, sc)
    assert result.passed is False
    assert "Prague" in result.reason or "Brno" in result.reason


def test_score_json_fields_match_invalid_json() -> None:
    sc = ScoringConfig(method="json_fields_match", expected_fields={"key": "val"})
    result = score_response("this is not json at all", sc)
    assert result.passed is False
    assert "not valid JSON" in result.reason


def test_score_multiple_choice_pass_and_fail() -> None:
    sc = ScoringConfig(method="multiple_choice", expected="B")
    assert score_response("The answer is B.", sc).passed is True
    assert score_response("I choose B because…", sc).passed is True
    assert score_response("Not A at all.", sc).passed is False


def test_score_multiple_choice_word_boundary() -> None:
    # "AB" should not match expected="A" when there is no standalone "A"
    sc = ScoringConfig(method="multiple_choice", expected="A")
    # Neither "AB" nor "BA" provides a word-boundary standalone match for "A"
    assert score_response("Option AB is correct.", sc).passed is False
    assert score_response("The answer is A.", sc).passed is True


# --------------------------------------------------------------------- #
# mode: scored — runner and summary integration                         #
# --------------------------------------------------------------------- #


def _suite_scored() -> CustomSuiteDefinition:
    return CustomSuiteDefinition(
        name="scored-suite",
        version="0.1",
        mode="scored",
        tasks=[
            CustomSuiteTask(
                task_id="s1",
                prompt="What is 2+2?",
                scoring=ScoringConfig(method="substring_match", must_contain=["4"]),
            ),
            CustomSuiteTask(
                task_id="s2",
                prompt="Capital of France?",
                scoring=ScoringConfig(method="exact_match", expected="Paris"),
            ),
        ],
        sampling=SamplingConfig(temperature=0.0, max_tokens=64, seed=42),
    )


def test_run_custom_suite_scored_pass_and_fail() -> None:
    suite = _suite_scored()
    # Model replies: s1 gets "4", s2 gets "London" (fail)
    backend = _FakeBackend(replies={
        ("qwen-3.6", "What is 2+2?"): "The answer is 4.",
        ("qwen-3.6", "Capital of France?"): "London",
    })
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
        per_model = {b["model"]: b for b in summary["per_model"]}
        bucket = per_model["qwen-3.6"]
        assert bucket["scored"] == 2
        assert bucket["passes"] == 1  # s1 passes (contains "4"), s2 fails (London ≠ Paris)
        assert bucket["pass_rate"] == 0.5
        # Rows must have score fields
        rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
        s1_row = next(r for r in rows if r["task_id"] == "s1")
        s2_row = next(r for r in rows if r["task_id"] == "s2")
        assert s1_row["score"]["passed"] is True
        assert s2_row["score"]["passed"] is False
        # Markdown shows PASS / FAIL
        md = (run_dir / "summary.md").read_text()
        assert "PASS" in md
        assert "FAIL" in md


def test_run_custom_suite_scored_suite_level_default_scorer() -> None:
    suite = CustomSuiteDefinition(
        name="suite-default-scorer",
        mode="scored",
        scoring=ScoringConfig(method="substring_match", must_contain=["yes"]),
        tasks=[
            CustomSuiteTask(task_id="q1", prompt="Do you agree?"),
            CustomSuiteTask(task_id="q2", prompt="Any other thoughts?",
                            scoring=ScoringConfig(method="exact_match", expected="no")),
        ],
        sampling=SamplingConfig(temperature=0.0, max_tokens=32, seed=42),
    )
    backend = _FakeBackend(replies={
        ("qwen-3.6", "Do you agree?"): "yes I do",
        ("qwen-3.6", "Any other thoughts?"): "no",
    })
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        summary = run_custom_suite_quick(
            suite=suite,
            backend=backend,
            backend_config=_make_backend(),
            model_configs=[_make_model("qwen-3.6", "qwen3.6:35b")],
            run_dir=run_dir,
        )
        bucket = summary["per_model"][0]
        assert bucket["passes"] == 2
        assert bucket["scored"] == 2


def test_run_custom_suite_scored_unscored_task_gets_null_verdict() -> None:
    suite = CustomSuiteDefinition(
        name="partial-scoring",
        mode="scored",
        tasks=[
            CustomSuiteTask(task_id="no-scorer", prompt="Tell me something."),
        ],
        sampling=SamplingConfig(temperature=0.0, max_tokens=32, seed=42),
    )
    backend = _FakeBackend()
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        summary = run_custom_suite_quick(
            suite=suite,
            backend=backend,
            backend_config=_make_backend(),
            model_configs=[_make_model("qwen-3.6", "qwen3.6:35b")],
            run_dir=run_dir,
        )
        rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
        assert rows[0]["score"] is None
        bucket = summary["per_model"][0]
        assert bucket["scored"] == 0
        assert bucket["pass_rate"] is None


def test_validate_warns_on_scored_mode_without_any_scorer() -> None:
    suite = CustomSuiteDefinition(
        name="unscored",
        mode="scored",
        tasks=[
            CustomSuiteTask(task_id="t1", prompt="Hello"),
            CustomSuiteTask(task_id="t2", prompt="World"),
        ],
    )
    issues = validate_custom_suite(suite)
    assert any(
        i.severity == "warning" and "no verdict" in i.message.lower()
        for i in issues
    )


def test_run_custom_suite_dry_run_no_files_written() -> None:
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
            dry_run=True,
        )
        # dry_run flag is set in summary
        assert summary.get("dry_run") is True
        # Only one row was run
        assert len(summary["rows"]) == 1
        assert summary["rows"][0]["task_id"] == "t1"
        # No files were written
        assert not (run_dir / "results.jsonl").exists()
        assert not (run_dir / "summary.md").exists()
        # Backend was called only once for generate
        assert backend.calls.count(("generate", "qwen-3.6")) == 1


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
