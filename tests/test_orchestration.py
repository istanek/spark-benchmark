from spark_benchmark.orchestration import parse_benchmark_request


def test_parse_benchmark_request_defaults_to_all_models_and_core_suites() -> None:
    plan = parse_benchmark_request("udělej benchmark pro OpenClaw use case", ["qwen-3.6", "gemma-4", "nemotron-3"])
    assert plan.selected_models == ["qwen-3.6", "gemma-4", "nemotron-3"]
    assert "openclaw_speed" in plan.selected_suites
    assert "practical_structured_output" in plan.selected_suites


def test_parse_benchmark_request_selects_requested_models_and_suites() -> None:
    plan = parse_benchmark_request(
        "otestuj qwen a gemma, zamer se na rychlost a spolehlivost",
        ["qwen-3.6", "gemma-4", "nemotron-3"],
    )
    assert plan.selected_models == ["qwen-3.6", "gemma-4"]
    assert plan.selected_suites == ["openclaw_speed", "hallucination_grounding"]


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
