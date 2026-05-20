from pathlib import Path

from spark_benchmark.sustained_throughput import (
    GenerationRecord,
    TelemetrySample,
    compute_derived_metrics,
    compute_windows,
    load_sustained_throughput_suite,
)


def _rec(seq: int, start: float, end: float, tokens: int) -> GenerationRecord:
    return GenerationRecord(
        seq=seq,
        prompt_family="a",
        task_id="t",
        started_s=start,
        finished_s=end,
        decode_tokens=tokens,
        decode_time_s=end - start,
        ttft_ms=10.0,
    )


def test_compute_windows_splits_records_into_walls() -> None:
    records = [
        _rec(0, 0.0, 0.5, 50),
        _rec(1, 0.5, 1.0, 50),
        _rec(2, 1.0, 2.0, 50),
    ]
    windows = compute_windows(records, duration_s=2.0, window_s=1.0)
    assert len(windows) == 2
    assert windows[0]["decode_tokens"] == 100
    assert windows[0]["tokens_per_s"] == 100.0
    assert windows[0]["generations_completed"] == 1
    assert windows[1]["decode_tokens"] == 50
    assert windows[1]["tokens_per_s"] == 50.0


def test_compute_windows_empty_records_returns_zeroed_windows() -> None:
    windows = compute_windows([], duration_s=2.0, window_s=1.0)
    assert windows == [
        {"window_index": 0, "start_s": 0.0, "end_s": 1.0, "decode_tokens": 0, "decode_time_s": 0.0, "tokens_per_s": 0.0, "generations_completed": 0},
        {"window_index": 1, "start_s": 1.0, "end_s": 2.0, "decode_tokens": 0, "decode_time_s": 0.0, "tokens_per_s": 0.0, "generations_completed": 0},
    ]


def test_compute_derived_detects_throttling() -> None:
    records = [
        _rec(0, 0.0, 0.5, 50),
        _rec(1, 0.5, 1.0, 50),
        _rec(2, 1.0, 2.0, 50),  # half the rate in window 2
    ]
    windows = compute_windows(records, duration_s=2.0, window_s=1.0)
    derived = compute_derived_metrics(records, windows, samples=[], duration_s=2.0)
    assert derived["initial_tokens_per_s"] == 100.0
    assert derived["sustained_tokens_per_s"] == 50.0
    assert derived["peak_tokens_per_s"] == 100.0
    assert derived["throttle_ratio"] == 0.5
    assert derived["time_to_throttle_s"] == 1.0


def test_compute_derived_handles_stable_throughput() -> None:
    records = [
        _rec(0, 0.0, 0.5, 50),
        _rec(1, 0.5, 1.0, 50),
        _rec(2, 1.0, 1.5, 50),
        _rec(3, 1.5, 2.0, 50),
    ]
    windows = compute_windows(records, duration_s=2.0, window_s=1.0)
    derived = compute_derived_metrics(records, windows, samples=[], duration_s=2.0)
    assert derived["throttle_ratio"] == 1.0
    assert derived["time_to_throttle_s"] is None


def test_compute_derived_calculates_energy_when_power_samples_present() -> None:
    records = [_rec(0, 0.0, 1.0, 100), _rec(1, 1.0, 2.0, 100)]
    windows = compute_windows(records, duration_s=2.0, window_s=1.0)
    samples = [
        TelemetrySample(timestamp_s=0.5, gpu_power_w=80.0, gpu_temp_c=70.0),
        TelemetrySample(timestamp_s=1.5, gpu_power_w=100.0, gpu_temp_c=82.0),
    ]
    derived = compute_derived_metrics(records, windows, samples=samples, duration_s=2.0)
    assert derived["avg_power_w"] == 90.0
    assert derived["peak_temp_c"] == 82.0
    # energy = avg_power(W) * duration(s) / tokens = 90 * 2 / 200 = 0.9 J/token
    assert derived["energy_j_per_token"] == 0.9


def test_load_sustained_throughput_fixture_has_three_prompts() -> None:
    suite = load_sustained_throughput_suite(Path(__file__).resolve().parent.parent)
    assert suite.name == "sustained_throughput_v1"
    assert len(suite.tasks) == 3
    families = {t.metadata.get("prompt_family") for t in suite.tasks}
    assert families == {"explain", "code", "summarize"}


def _run_all() -> int:
    import inspect, sys
    failures: list[str] = []
    module = sys.modules[__name__]
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"ok  {name}")
        except Exception as exc:
            failures.append(f"{name}: {exc!r}")
            print(f"FAIL {name}: {exc!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_run_all())
