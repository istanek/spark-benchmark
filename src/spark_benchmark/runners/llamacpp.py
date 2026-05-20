from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from spark_benchmark.models import BackendConfig, GenerationResult, InferenceMetrics, ModelConfig, SamplingConfig


class LlamaCppAdapter:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.model: ModelConfig | None = None
        self.last_metrics = InferenceMetrics(
            backend_version=config.version,
            quantization="",
        )

    def load_model(self, model_config: ModelConfig) -> None:
        self.model = model_config
        self.last_metrics.quantization = model_config.quantization

    def _resolve_executable(self) -> str:
        candidates = []
        if self.config.executable:
            candidates.append(self.config.executable)
        entrypoint = self.config.entrypoint.strip()
        if entrypoint:
            candidates.append(entrypoint)
        candidates.extend(["llama-cli", "llama.cpp", "llama-server"])

        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
            if Path(candidate).exists():
                return candidate
        raise RuntimeError(
            "No llama.cpp executable found. Set backends/llamacpp.yaml executable or add llama-cli to PATH."
        )

    def _require_model_path(self) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if not self.model.artifact_path:
            raise RuntimeError(
                f"Model {self.model.name} is missing artifact_path. Point it to a local GGUF file in configs/models/*.yaml."
            )
        model_path = Path(self.model.artifact_path)
        if not model_path.exists():
            raise RuntimeError(f"Model artifact not found: {model_path}")
        return str(model_path)

    def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult:
        executable = self._resolve_executable()
        model_path = self._require_model_path()
        started = time.perf_counter()
        cmd = [
            executable,
            "-m",
            model_path,
            "-p",
            prompt,
            "-n",
            str(params.max_tokens),
            "--temp",
            str(params.temperature),
            "--top-p",
            str(params.top_p),
            "--seed",
            str(params.seed),
            "--ctx-size",
            str(self.model.context_length if self.model else 0),
            "--no-display-prompt",
        ]
        cmd.extend(self.config.default_args)
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"llama.cpp failed with exit code {completed.returncode}: {stderr}")

        output = completed.stdout.strip()
        self.last_metrics = InferenceMetrics(
            decode_time_s=elapsed,
            backend_version=self.config.version,
            quantization=self.model.quantization if self.model else "",
        )
        return GenerationResult(
            prompt=prompt,
            output=output,
            finish_reason="stop",
            metrics=self.last_metrics,
            raw={
                "command": cmd,
                "stderr": completed.stderr,
            },
        )

    def get_metrics(self) -> InferenceMetrics:
        return self.last_metrics

    def unload(self) -> None:
        self.model = None
