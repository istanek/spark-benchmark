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
