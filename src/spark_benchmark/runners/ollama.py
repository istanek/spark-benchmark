from __future__ import annotations

import json
import os
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

# Environment overrides — standard Ollama conventions. OLLAMA_HOST redirects
# every request (e.g. to https://ollama.com for Ollama Cloud); OLLAMA_API_KEY
# is sent as a Bearer token. The key is read from the environment only and is
# never copied into BackendConfig.options, so it cannot leak into a manifest.
ENV_HOST = "OLLAMA_HOST"
ENV_API_KEY = "OLLAMA_API_KEY"


def resolve_ollama_base(options: dict[str, Any] | None = None) -> str:
    """Return the Ollama base URL (scheme://host[:port], no /api path).

    Precedence: ``$OLLAMA_HOST`` (bare host gets ``https://``) → the
    ``endpoint`` option → the local default. Used for both /api/generate
    and sibling endpoints (/api/tags, /api/ps).
    """
    host = os.environ.get(ENV_HOST, "").strip()
    if host:
        if not host.startswith(("http://", "https://")):
            host = "https://" + host
        return host.rstrip("/")
    endpoint = str((options or {}).get("endpoint") or DEFAULT_ENDPOINT)
    return endpoint.rsplit("/api/", 1)[0]


def ollama_auth_headers() -> dict[str, str]:
    """Bearer auth header from ``$OLLAMA_API_KEY`` (empty dict when unset)."""
    key = os.environ.get(ENV_API_KEY, "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def is_cloud_endpoint() -> bool:
    """True when talking to Ollama Cloud (auth key present or ollama.com host)."""
    if os.environ.get(ENV_API_KEY, "").strip():
        return True
    return "ollama.com" in os.environ.get(ENV_HOST, "").lower()


def model_is_cloud(model: ModelConfig | None) -> bool:
    """True when *model* should be routed to Ollama Cloud rather than localhost.

    A model is cloud-routed when its ``source`` is ``ollama-cloud`` or its
    Ollama tag carries the ``-cloud`` suffix (the convention Ollama Cloud uses,
    e.g. ``gpt-oss:120b-cloud``).
    """
    if model is None:
        return False
    if model.source == "ollama-cloud":
        return True
    tag = model.artifact_path or model.revision or ""
    return tag.endswith("-cloud")


class OllamaAdapter:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.model: ModelConfig | None = None
        # Endpoint is resolved per-model in `_base_for_model` so a single run
        # can mix local and cloud models. `self.endpoint` is kept as a default
        # for callers that read it before a model is loaded.
        self.endpoint = resolve_ollama_base(config.options) + "/api/generate"
        self.timeout_s = float(config.options.get("request_timeout_s") or DEFAULT_TIMEOUT_S)
        self.last_metrics = InferenceMetrics(
            backend_version=config.version,
            quantization="",
        )

    def load_model(self, model_config: ModelConfig) -> None:
        self.model = model_config
        self.last_metrics.quantization = model_config.quantization

    def _base_for_model(self) -> str:
        """Base URL (scheme://host[:port]) for the currently loaded model.

        Routing is per-model so local and cloud models can be benchmarked in
        the same run:

        - **Cloud model** (``source == "ollama-cloud"`` or a ``-cloud`` tag) →
          ``https://ollama.com`` (or an explicit cloud ``$OLLAMA_HOST``).
        - **Local model** → the configured backend endpoint / localhost. A
          cloud ``$OLLAMA_HOST`` is deliberately ignored here so a local model
          is never sent to ollama.com; a non-cloud custom ``$OLLAMA_HOST``
          (e.g. a private remote Ollama) is still honoured.
        """
        if model_is_cloud(self.model):
            host = os.environ.get(ENV_HOST, "").strip()
            if host and "ollama.com" in host.lower():
                return (host if host.startswith(("http://", "https://")) else "https://" + host).rstrip("/")
            return "https://ollama.com"
        # Local model: configured endpoint, ignoring a cloud $OLLAMA_HOST.
        base = str(self.config.options.get("endpoint") or DEFAULT_ENDPOINT).rsplit("/api/", 1)[0]
        env_host = os.environ.get(ENV_HOST, "").strip()
        if env_host and "ollama.com" not in env_host.lower():
            base = (env_host if env_host.startswith(("http://", "https://")) else "https://" + env_host).rstrip("/")
        return base

    def _generate_endpoint(self) -> str:
        return self._base_for_model() + "/api/generate"

    def _headers(self) -> dict[str, str]:
        """JSON content-type, plus a Bearer token only for cloud-routed models.

        The API key is never sent to a local endpoint — both to avoid leaking
        it and because the local daemon doesn't expect it.
        """
        headers = {"Content-Type": "application/json"}
        if model_is_cloud(self.model):
            headers.update(ollama_auth_headers())
        return headers

    def _model_tag(self) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # artifact_path holds the Ollama tag, e.g. "qwen3.6:35b"; fall back to revision/name.
        tag = self.model.artifact_path or self.model.revision or self.model.name
        if not tag:
            raise RuntimeError(f"Model {self.model.name} has no Ollama tag (artifact_path/revision)")
        return tag

    def _build_payload(self, prompt: str, params: SamplingConfig) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "seed": params.seed,
            "num_predict": params.max_tokens,
        }
        # Explicit context window: without this a long prompt is silently
        # truncated to the server default. Only sent when set so short-suite
        # behaviour is unchanged.
        if params.num_ctx is not None:
            options["num_ctx"] = params.num_ctx
        return {
            "model": self._model_tag(),
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options,
        }

    def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult:
        payload = self._build_payload(prompt, params)
        body = json.dumps(payload).encode("utf-8")
        endpoint = self._generate_endpoint()
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code} from {endpoint}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed to {endpoint}: {exc.reason}") from exc
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
                "endpoint": endpoint,
                "request": payload,
                "response": data,
                "wall_time_s": elapsed,
            },
        )

    def get_metrics(self) -> InferenceMetrics:
        return self.last_metrics

    def _ps_endpoint(self) -> str:
        # Route /api/ps to the same host as the current model's generate call.
        return f"{self._base_for_model()}/api/ps"

    def memory_snapshot(self) -> dict[str, Any] | None:
        """Best-effort memory for the loaded model via Ollama ``/api/ps``.

        On the DGX Spark ``nvidia-smi`` reports ``memory.used`` as N/A
        (unified memory), so ``/api/ps`` — which lists resident models with
        their ``size`` / ``size_vram`` in bytes — is the reliable source for
        the long-context memory-growth story. Returns ``None`` if the
        endpoint is unreachable or the model is not resident.
        """
        if self.model is None:
            return None
        try:
            tag = self._model_tag()
        except RuntimeError:
            return None
        request = urllib.request.Request(
            self._ps_endpoint(), headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=15.0) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
            return None
        for entry in data.get("models") or []:
            if entry.get("name") == tag or entry.get("model") == tag:
                size = int(entry.get("size") or 0)
                size_vram = int(entry.get("size_vram") or 0)
                return {
                    "size_bytes": size,
                    "size_vram_bytes": size_vram,
                    "size_mb": round(size / (1024 * 1024), 1),
                    "size_vram_mb": round(size_vram / (1024 * 1024), 1),
                }
        return None

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
            self._generate_endpoint(),
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15.0) as response:
                response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            pass
        self.model = None
