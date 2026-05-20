from __future__ import annotations

import json
from pathlib import Path
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SuiteCategory(str, Enum):
    QUALITY = "quality"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    CUSTOM = "custom"


class SuiteTask(BaseModel):
    """Single evaluation task inside a suite."""

    task_id: str
    prompt: str
    context: str | None = None
    reference: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SuiteDefinition(BaseModel):
    """Static description of a suite and its tasks."""

    name: str
    category: SuiteCategory
    description: str = ""
    version: str = "0.0.1"
    tasks: list[SuiteTask] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_name(self) -> "SuiteDefinition":
        if not self.name.strip():
            raise ValueError("suite name must not be empty")
        return self


def load_suite_definition(path: Path | str) -> SuiteDefinition:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SuiteDefinition.model_validate(payload)
