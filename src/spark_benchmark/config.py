from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from .models import BackendConfig, ExperimentFile, ModelConfig, PlatformConfig

T = TypeVar("T", bound=BaseModel)


def load_yaml_model(path: Path, model_type: type[T]) -> T:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return model_type.model_validate(payload)


def load_experiment(path: Path) -> ExperimentFile:
    return load_yaml_model(path, ExperimentFile)


def load_platform(path: Path) -> PlatformConfig:
    return load_yaml_model(path, PlatformConfig)


def load_model_config(path: Path) -> ModelConfig:
    return load_yaml_model(path, ModelConfig)


def load_backend(path: Path) -> BackendConfig:
    return load_yaml_model(path, BackendConfig)
