from __future__ import annotations

from spark_benchmark.models import BackendConfig, BackendKind
from spark_benchmark.runners.llamacpp import LlamaCppAdapter
from spark_benchmark.runners.ollama import OllamaAdapter
from spark_benchmark.runners.stub import StubBackendAdapter


def build_backend(config: BackendConfig) -> StubBackendAdapter | LlamaCppAdapter | OllamaAdapter:
    if config.name == BackendKind.LLAMACPP:
        return LlamaCppAdapter(config)
    if config.name == BackendKind.OLLAMA:
        return OllamaAdapter(config)
    return StubBackendAdapter(config)
