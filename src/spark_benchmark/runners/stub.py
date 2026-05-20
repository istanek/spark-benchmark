from __future__ import annotations

from spark_benchmark.models import BackendConfig, GenerationResult, InferenceMetrics, ModelConfig, SamplingConfig


class StubBackendAdapter:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.model: ModelConfig | None = None

    def load_model(self, model_config: ModelConfig) -> None:
        self.model = model_config

    def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return GenerationResult(
            prompt=prompt,
            output=f"[stub:{self.config.name}] generation not implemented for model {self.model.name}",
            finish_reason="stub",
            metrics=self.get_metrics(),
            raw={"sampling": params.model_dump()},
        )

    def get_metrics(self) -> InferenceMetrics:
        return InferenceMetrics(
            backend_version=self.config.version,
            quantization=self.model.quantization if self.model else "",
        )

    def unload(self) -> None:
        self.model = None
