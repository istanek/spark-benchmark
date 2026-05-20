from pathlib import Path

from spark_benchmark.code_generation import (
    DEFAULT_TOLERANCE_PP,
    extract_code,
    load_code_generation_suite,
    load_reference_scores,
    pass_at_k,
    sandbox_run,
    validate_reference_scores,
)
from spark_benchmark.reliability import load_reliability_suite


REPO_ROOT = Path(__file__).resolve().parents[1]


def _approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _expect_raises(exc_type: type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} but no exception was raised")


def test_pass_at_k_all_correct() -> None:
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 10, 10) == 1.0


def test_pass_at_k_none_correct() -> None:
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 0, 10) == 0.0


def test_pass_at_k_single_sample_pass_at_1() -> None:
    assert pass_at_k(1, 0, 1) == 0.0
    assert pass_at_k(1, 1, 1) == 1.0


def test_pass_at_k_partial() -> None:
    assert _approx_equal(pass_at_k(5, 2, 1), 0.4)
    assert pass_at_k(10, 5, 10) == 1.0
    assert pass_at_k(10, 9, 10) == 1.0


def test_pass_at_k_rejects_bad_input() -> None:
    _expect_raises(ValueError, lambda: pass_at_k(0, 0, 1))
    _expect_raises(ValueError, lambda: pass_at_k(10, -1, 1))
    _expect_raises(ValueError, lambda: pass_at_k(10, 11, 1))
    _expect_raises(ValueError, lambda: pass_at_k(10, 5, 0))


def test_extract_code_from_markdown_fence() -> None:
    output = (
        "Sure, here's the solution:\n\n"
        "```python\n"
        "def foo(x):\n"
        "    return x * 2\n"
        "```\n"
        "Hope that helps!\n"
    )
    extracted = extract_code(output, prompt="def foo(x):\n", entry_point="foo")
    assert "def foo(x):" in extracted
    assert "return x * 2" in extracted
    assert "Hope that helps" not in extracted


def test_extract_code_when_def_inline() -> None:
    output = "def foo(x):\n    return x + 1\n"
    extracted = extract_code(output, prompt="def foo(x):\n    pass\n", entry_point="foo")
    assert extracted.startswith("def foo(x):")
    assert "return x + 1" in extracted


def test_extract_code_treats_raw_continuation_as_body() -> None:
    prompt = "def foo(x):\n    \"\"\"docstring\"\"\"\n"
    output = "    return x * 3\n"
    extracted = extract_code(output, prompt=prompt, entry_point="foo")
    assert extracted == prompt + output


def test_sandbox_run_passes_known_good_program() -> None:
    program = "def f(x):\n    return x + 1\nassert f(2) == 3\n"
    result = sandbox_run(program, timeout_s=5.0)
    assert result.passed is True
    assert result.status == "passed"
    assert result.exit_code == 0


def test_sandbox_run_marks_assertion_failure() -> None:
    program = "def f(x):\n    return 0\nassert f(2) == 3\n"
    result = sandbox_run(program, timeout_s=5.0)
    assert result.passed is False
    assert result.status == "failed"
    assert result.exit_code not in (0, None)


def test_sandbox_run_marks_syntax_error_as_compile_error() -> None:
    program = "def f(x):\n    return x +\n"
    result = sandbox_run(program, timeout_s=5.0)
    assert result.passed is False
    assert result.status == "compile_error"


def test_sandbox_run_kills_infinite_loop_on_timeout() -> None:
    program = "while True:\n    pass\n"
    result = sandbox_run(program, timeout_s=1.5)
    assert result.passed is False
    assert result.status == "timeout"


def test_load_code_generation_fixture() -> None:
    suite = load_code_generation_suite(REPO_ROOT)
    assert suite.name == "code_generation_v1"
    assert suite.category.value == "quality"
    assert len(suite.tasks) >= 5
    for task in suite.tasks:
        assert "entry_point" in task.metadata
        assert "tests" in task.metadata
        assert "benchmark" in task.metadata


def test_code_generation_fixture_via_reliability_loader() -> None:
    suite = load_reliability_suite(REPO_ROOT, "code_generation")
    assert suite.name == "code_generation_v1"


def test_canonical_solutions_pass_sandbox() -> None:
    """Sanity check: every starter task's canonical_solution + tests must pass."""
    suite = load_code_generation_suite(REPO_ROOT)
    for task in suite.tasks:
        canonical = task.metadata["canonical_solution"]
        tests = task.metadata["tests"]
        program = task.prompt + canonical + "\n\n" + tests
        result = sandbox_run(program, timeout_s=10.0)
        assert result.passed, (
            f"Canonical solution for {task.task_id} did not pass sandbox: "
            f"{result.status} / {result.stderr[-500:]}"
        )


def test_reference_scores_yaml_loads() -> None:
    reference = load_reference_scores(REPO_ROOT / "data" / "code" / "reference_scores.yaml")
    assert "models" in reference
    assert reference["tolerance_pp"] == DEFAULT_TOLERANCE_PP


def test_validate_reference_scores_emits_warning_outside_tolerance() -> None:
    reference = {
        "tolerance_pp": 3.0,
        "models": {
            "qwen-3.6": {
                "humaneval": {
                    "pass_at_1": 80.0,
                    "source": "test",
                    "enforce": True,
                }
            }
        },
    }
    comparisons = validate_reference_scores(
        per_model_per_benchmark={"qwen-3.6": {"humaneval": 0.70}},
        reference=reference,
    )
    assert len(comparisons) == 1
    cmp = comparisons[0]
    assert cmp.expected_pct == 80.0
    assert cmp.observed_pct == 70.0
    assert _approx_equal(cmp.delta_pp, -10.0)
    assert cmp.within_tolerance is False
    assert cmp.enforce is True


def test_validate_reference_scores_within_tolerance() -> None:
    reference = {
        "tolerance_pp": 3.0,
        "models": {
            "qwen-3.6": {
                "humaneval": {
                    "pass_at_1": 80.0,
                    "source": "test",
                    "enforce": True,
                }
            }
        },
    }
    comparisons = validate_reference_scores(
        per_model_per_benchmark={"qwen-3.6": {"humaneval": 0.785}},
        reference=reference,
    )
    assert comparisons[0].within_tolerance is True


def test_validate_reference_scores_missing_expected_is_not_enforced() -> None:
    reference = {"tolerance_pp": 3.0, "models": {"qwen-3.6": {"humaneval": {"pass_at_1": None}}}}
    comparisons = validate_reference_scores(
        per_model_per_benchmark={"qwen-3.6": {"humaneval": 0.5}},
        reference=reference,
    )
    assert comparisons[0].expected_pct is None
    assert comparisons[0].enforce is False
    assert comparisons[0].within_tolerance is None


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
