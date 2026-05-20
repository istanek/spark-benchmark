from __future__ import annotations

import time


class StubTelemetryCollector:
    def __init__(self, name: str) -> None:
        self.name = name
        self.started_at: float | None = None

    def start(self) -> None:
        self.started_at = time.time()

    def stop(self) -> None:
        self.started_at = None

    def snapshot(self) -> dict[str, object]:
        return {
            "collector": self.name,
            "running": self.started_at is not None,
        }
