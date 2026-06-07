from pathlib import Path

from spark_benchmark.config import load_backend
from spark_benchmark.reliability import (
    build_summary,
    load_reliability_suite,
    score_hallucination_task,
    score_structured_output_task,
)
from spark_benchmark.suites import SuiteTask


def test_load_reliability_suite_fixture() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite = load_reliability_suite(repo_root, "hallucination_grounding")
    assert suite.name == "hallucination_grounding_v1"
    assert len(suite.tasks) == 9
    behaviors = {task.metadata.get("expected_behavior") for task in suite.tasks}
    assert behaviors == {"answer_from_context", "abstain", "correct_user"}


def test_score_hallucination_task_behaviors() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite = load_reliability_suite(repo_root, "hallucination_grounding")

    grounded = score_hallucination_task(suite.tasks[0], "Atlas-3 was launched in 2019.")
    abstain = score_hallucination_task(suite.tasks[1], "The context does not mention the lead engineer.")
    correct_user = score_hallucination_task(suite.tasks[2], "No, that is incorrect. The context says 2019.")

    assert grounded["passed"] is True
    assert abstain["passed"] is True
    assert correct_user["passed"] is True


def test_score_answer_from_context_fail_when_missing_reference() -> None:
    task = SuiteTask(
        task_id="t1",
        prompt="What year?",
        reference="2019",
        metadata={"expected_behavior": "answer_from_context"},
    )
    result = score_hallucination_task(task, "I am unable to determine that.")
    assert result["passed"] is False
    assert result["score"] == 0


def test_score_abstain_fail_when_fabricated() -> None:
    task = SuiteTask(
        task_id="t2",
        prompt="Who was the lead?",
        reference="The context does not mention the lead engineer.",
        metadata={"expected_behavior": "abstain"},
    )
    result = score_hallucination_task(task, "The lead engineer was Jane Doe.")
    assert result["passed"] is False


def test_build_summary_counts_scores() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite = load_reliability_suite(repo_root, "hallucination_grounding")
    backend = load_backend(repo_root / "configs" / "backends" / "ollama.yaml")
    summary = build_summary(
        [
            {"model": "qwen-3.6", "task_id": "t1", "evaluation": {"score": 1, "passed": True}},
            {"model": "qwen-3.6", "task_id": "t2", "evaluation": {"score": 0, "passed": False}},
            {"model": "gemma-4", "task_id": "t3", "evaluation": {"score": 1, "passed": True}},
        ],
        suite,
        backend,
    )
    assert summary["total_rows"] == 3
    model_rows = {item["model"]: item for item in summary["models"]}
    assert model_rows["qwen-3.6"]["passes"] == 1
    assert model_rows["qwen-3.6"]["total"] == 2
    assert model_rows["qwen-3.6"]["failed_task_ids"] == ["t2"]
    assert model_rows["gemma-4"]["pass_rate"] == 1.0


def test_load_practical_structured_output_fixture() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite = load_reliability_suite(repo_root, "practical_structured_output")
    assert suite.name == "practical_structured_output_v1"
    assert len(suite.tasks) == 6
    assert {task.metadata.get("expected_behavior") for task in suite.tasks} == {"json_exact_match"}


def test_load_openclaw_speed_fixture() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite = load_reliability_suite(repo_root, "openclaw_speed")
    assert suite.name == "openclaw_speed_v1"
    assert len(suite.tasks) == 3
    assert {task.metadata.get("expected_behavior") for task in suite.tasks} == {"performance_probe"}


def test_score_structured_output_task_exact_match() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite = load_reliability_suite(repo_root, "practical_structured_output")
    result = score_structured_output_task(
        suite.tasks[0],
        '{"issue_type":"incident","priority":"critical","needs_followup":true,"owner":"ops-oncall"}',
    )
    assert result["passed"] is True
    assert result["reason"] == "exact_json_match"


def test_score_structured_output_task_rejects_trailing_text() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite = load_reliability_suite(repo_root, "practical_structured_output")
    result = score_structured_output_task(
        suite.tasks[0],
        '{"issue_type":"incident","priority":"critical","needs_followup":true,"owner":"ops-oncall"} done',
    )
    assert result["passed"] is False
    assert result["reason"] == "trailing_text_after_json"


def _make_suite_entry(suite_name: str, models: list[dict]) -> dict:
    return {"suite": suite_name, "models": models}


def _make_model_entry(name: str, pass_rate: float, **kwargs) -> dict:
    return {"model": name, "pass_rate": pass_rate, "passes": 1, "total": 1, **kwargs}


def test_overall_rank_includes_all_quality_suites_that_ran() -> None:
    """All quality suites that ran contribute equally to quality_score."""
    from spark_benchmark.reporting import _overall_rank_rows

    # Model A: perfect grounding but 0% code gen
    # Model B: 50% grounding but 100% code gen
    # With old formula (hallucination 60%, no code gen): A wins easily
    # With new formula (equal average of present suites): B should score higher
    aggregate = {
        "suites": [
            _make_suite_entry("hallucination_grounding_v1", [
                _make_model_entry("model-a", 1.0),
                _make_model_entry("model-b", 0.5),
            ]),
            _make_suite_entry("code_generation_v1", [
                _make_model_entry("model-a", 0.0),
                _make_model_entry("model-b", 1.0),
            ]),
        ],
        "runs": [],
    }
    ranking = _overall_rank_rows(aggregate, ["model-a", "model-b"])
    # Both suites ran → quality_score = average of [hallucination, code_gen]
    # model-a: (1.0 + 0.0) / 2 = 0.50
    # model-b: (0.5 + 1.0) / 2 = 0.75 → model-b wins
    assert ranking[0]["model"] == "model-b", (
        f"model-b should win when averaging all quality suites, got {ranking[0]['model']}"
    )
    assert ranking[0]["quality_score"] == 0.75
    assert ranking[1]["quality_score"] == 0.50


def test_overall_rank_excludes_missing_suites_from_denominator() -> None:
    """Suites that didn't run are not counted against models that weren't tested on them."""
    from spark_benchmark.reporting import _overall_rank_rows

    # Only hallucination ran — code_gen is absent
    aggregate = {
        "suites": [
            _make_suite_entry("hallucination_grounding_v1", [
                _make_model_entry("qwen-3.6", 0.8),
            ]),
        ],
        "runs": [],
    }
    ranking = _overall_rank_rows(aggregate, ["qwen-3.6"])
    # quality_score should be 0.8 (hallucination only, code_gen excluded from denominator)
    assert ranking[0]["quality_score"] == 0.8


def test_overall_rank_long_context_contributes_when_run() -> None:
    """long_context_retrieval enters the score when it's part of the bundle."""
    from spark_benchmark.reporting import _overall_rank_rows

    # Model A: 100% hallucination, 33% long context
    # Model B: 80% hallucination, 90% long context
    aggregate = {
        "suites": [
            _make_suite_entry("hallucination_grounding_v1", [
                _make_model_entry("model-a", 1.0),
                _make_model_entry("model-b", 0.8),
            ]),
            _make_suite_entry("long_context_retrieval_v1", [
                _make_model_entry("model-a", 0.33),
                _make_model_entry("model-b", 0.90),
            ]),
        ],
        "runs": [],
    }
    ranking = _overall_rank_rows(aggregate, ["model-a", "model-b"])
    # model-a: (1.0 + 0.33) / 2 = 0.665
    # model-b: (0.8 + 0.90) / 2 = 0.850 → model-b wins
    assert ranking[0]["model"] == "model-b"
    assert ranking[0]["long_context_rate"] == 0.90


def _run_all() -> int:
    """Lightweight runner so tests work without pytest installed system-wide."""
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
