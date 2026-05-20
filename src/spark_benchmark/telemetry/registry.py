from __future__ import annotations

from spark_benchmark.telemetry.stub import StubTelemetryCollector


def build_collectors(names: list[str]) -> list[StubTelemetryCollector]:
    return [StubTelemetryCollector(name) for name in names]
