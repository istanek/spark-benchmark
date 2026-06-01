from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class BackendKind(str, Enum):
    LLAMACPP = "llamacpp"
    TRT_LLM = "trt-llm"
    VLLM = "vllm"
    OLLAMA = "ollama"


class SamplingConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    max_tokens: int = 2048


class ExperimentSpec(BaseModel):
    name: str
    description: str
    platforms: list[str]
    backend: BackendKind
    backend_version: str
    models: list[str]
    suites: list[str]
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    context_lengths: list[int] = Field(default_factory=lambda: [512, 4096, 16384, 65536])
    repetitions: int = 3
    warmup_runs: int = 1

    @model_validator(mode="after")
    def validate_lists(self) -> "ExperimentSpec":
        for field_name in ("platforms", "models", "suites", "context_lengths"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        return self


class ExperimentFile(BaseModel):
    experiment: ExperimentSpec


class PlatformConfig(BaseModel):
    name: str
    display_name: str
    architecture: str
    os: str
    telemetry: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    name: str
    family: str
    revision: str
    quantization: str
    source: str
    context_length: int
    artifact_path: str | None = None
    # Optional grouping key linking quantization variants of the same base
    # model (e.g. "llama-3.3-70b"). Set explicitly in YAML — never inferred
    # from `name` (brittle for odd names). Primarily consumed by the
    # quantization_sweep post-processor; long_context uses it for labels.
    base_model: str | None = None
    notes: list[str] = Field(default_factory=list)


class BackendConfig(BaseModel):
    name: BackendKind
    entrypoint: str
    version: str
    transport: str = "subprocess"
    executable: str | None = None
    default_args: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class InferenceMetrics(BaseModel):
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    ttft_ms: float = 0.0
    peak_memory_mb: float = 0.0
    backend_version: str = ""
    quantization: str = ""


class GenerationResult(BaseModel):
    prompt: str
    output: str
    finish_reason: str = "unknown"
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)
    raw: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSnapshot(BaseModel):
    platform_name: str
    backend_name: str
    backend_version: str
    python_version: str
    os: str
    hostname: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    experiment: ExperimentSpec
    platform: PlatformConfig
    backend: BackendConfig
    model_names: list[str]
    environment: EnvironmentSnapshot
    results_dir: Path
