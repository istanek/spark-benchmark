from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from spark_benchmark.models import (
    BackendConfig,
    GenerationResult,
    InferenceMetrics,
    ModelConfig,
    SamplingConfig,
)


DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_TIMEOUT_S = 300.0


class OllamaAdapter:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.model: ModelConfig | None = None
        self.endpoint = str(config.options.get("endpoint") or DEFAULT_ENDPOINT)
        self.timeout_s = float(config.options.get("request_timeout_s") or DEFAULT_TIMEOUT_S)
        self.last_metrics = InferenceMetrics(
            backend_version=config.version,
            quantization="",
        )

    def load_model(self, model_config: ModelConfig) -> None:
        self.model = model_config
        self.last_metrics.quantization = model_config.quantization

    def _model_tag(self) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # artifact_path holds the Ollama tag, e.g. "qwen3.6:35b"; fall back to revision/name.
        tag = self.model.artifact_path or self.model.revision or self.model.name
        if not tag:
            raise RuntimeError(f"Model {self.model.name} has no Ollama tag (artifact_path/revision)")
        return tag

    def _build_payload(self, prompt: str, params: SamplingConfig) -> dict[str, Any]:
        return {
            "model": self._model_tag(),
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "seed": params.seed,
                "num_predict": params.max_tokens,
            },
        }

    def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult:
        payload = self._build_payload(prompt, params)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code} from {self.endpoint}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed to {self.endpoint}: {exc.reason}") from exc
        elapsed = time.perf_counter() - started

        data = json.loads(raw_text)
        output = data.get("response", "")
        if not output:
            output = data.get("thinking", "")
        finish_reason = "stop" if data.get("done") else "incomplete"

        prefill_tokens = int(data.get("prompt_eval_count") or 0)
        decode_tokens = int(data.get("eval_count") or 0)
        prefill_ns = int(data.get("prompt_eval_duration") or 0)
        decode_ns = int(data.get("eval_duration") or 0)
        # Ollama returns nanoseconds for durations.
        prefill_time_s = prefill_ns / 1e9 if prefill_ns else 0.0
        decode_time_s = decode_ns / 1e9 if decode_ns else elapsed
        # Ollama does not separate TTFT; approximate with prefill duration.
        ttft_ms = prefill_time_s * 1000.0

        self.last_metrics = InferenceMetrics(
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            prefill_time_s=prefill_time_s,
            decode_time_s=decode_time_s,
            ttft_ms=ttft_ms,
            peak_memory_mb=0.0,
            backend_version=self.config.version,
            quantization=self.model.quantization if self.model else "",
        )

        return GenerationResult(
            prompt=prompt,
            output=output,
            finish_reason=finish_reason,
            metrics=self.last_metrics,
            raw={
                "endpoint": self.endpoint,
                "request": payload,
                "response": data,
                "wall_time_s": elapsed,
            },
        )

    def get_metrics(self) -> InferenceMetrics:
        return self.last_metrics

    def unload(self) -> None:
        """Evict the current model from Ollama's memory so the next model fits.

        Ollama keeps a model resident based on its ``keep_alive`` setting. Posting
        a generate request with ``keep_alive: 0`` and an empty prompt tells the
        server to release the weights immediately. We swallow errors so a
        failed unload does not abort the benchmark — the model name is still
        cleared locally either way.
        """
        if self.model is None:
            return
        try:
            tag = self._model_tag()
        except RuntimeError:
            self.model = None
            return
        payload = json.dumps({"model": tag, "keep_alive": 0}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15.0) as response:
                response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            pass
        self.model = None
