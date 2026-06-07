"""Shared Ollama model registry / auto-detection.

Every entrypoint (curses TUI, wizard, console REPL, NL benchmark, plain run)
needs the same logic for "what models can I actually use right now?". This
module owns that logic so the four CLI surfaces don't drift.

Hierarchy of trust:

1. Experiment YAML (`configs/experiments/*.yaml` -> `configs/models/*.yaml`)
   is the canonical, reviewed lineup. Sampling defaults, expected family,
   notes, etc. all come from there. By default this is the *only* source.

2. Auto-detection from a running Ollama (`/api/tags`) is opt-in via the
   per-command ``--allow-auto-detected`` flag (already always-on inside
   the curses TUI, where the user is sitting in front of the screen and
   can see the ``auto-detected`` label next to every synthesized entry).

   Auto-detected configs carry ``notes=["auto-detected from Ollama
   (no YAML config)"]`` and use Ollama defaults for everything we cannot
   read off `/api/tags`. Their `name` is a slugified version of the tag
   (``phi4:14b`` -> ``phi4-14b``).

Vision and embedding tags are filtered out either way: chat / completion
suites cannot meaningfully run them.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Sequence

from spark_benchmark.models import BackendConfig, ModelConfig


@dataclass
class DetectedOllamaModel:
    tag: str
    family: str = ""
    families: tuple[str, ...] = ()
    parameter_size: str = ""
    quantization_level: str = ""
    source: str = "local"
    """'local' for models pulled to the local Ollama, 'cloud' for Ollama Cloud."""


@dataclass
class OllamaModelInfo:
    tag: str
    config: ModelConfig | None
    auto_detected: bool = False
    disable_reason: str | None = None
    is_cloud: bool = False
    """True when this model was detected from Ollama Cloud (not the local daemon)."""

    @property
    def has_config(self) -> bool:
        return self.config is not None

    @property
    def display_name(self) -> str:
        return self.config.name if self.config else self.tag


_VISION_FAMILY_HINTS = ("vl", "vision", "clip", "pixtral", "llava", "moondream")
_VISION_TAG_HINTS = ("pixtral", "llava", "vision", "-vl", ":vl", "/vl")
_EMBEDDING_FAMILY_HINTS = ("bert", "embed")
_EMBEDDING_TAG_HINTS = ("embed",)


def _family_haystack(detected: DetectedOllamaModel) -> str:
    return " ".join((detected.family, *detected.families)).lower()


def is_vision_model(detected: DetectedOllamaModel) -> bool:
    fams = _family_haystack(detected)
    if any(hint in fams for hint in _VISION_FAMILY_HINTS):
        return True
    tag = detected.tag.lower()
    return any(hint in tag for hint in _VISION_TAG_HINTS)


def is_embedding_model(detected: DetectedOllamaModel) -> bool:
    fams = _family_haystack(detected)
    if any(hint in fams for hint in _EMBEDDING_FAMILY_HINTS):
        return True
    return any(hint in detected.tag.lower() for hint in _EMBEDDING_TAG_HINTS)


def slugify_tag(tag: str) -> str:
    return tag.replace(":", "-").replace("/", "_")


def synthesize_model_config(detected: DetectedOllamaModel) -> ModelConfig:
    """Build a ``ModelConfig`` for an Ollama tag that has no curated YAML.

    Defaults are deliberately conservative: 131072 ctx (Ollama default for
    most chat models), Ollama-default quantization, and an explicit note so
    reports/manifests can flag the run as not-fully-curated.

    Cloud models (``detected.source == "cloud"``) are tagged ``source="ollama-cloud"``
    and carry a no-local-telemetry note, mirroring ``synthesize_cloud_model_config``.
    """
    if detected.source == "cloud":
        return ModelConfig(
            name=slugify_tag(detected.tag),
            family=detected.family or (detected.families[0] if detected.families else "cloud"),
            revision=detected.tag,
            quantization=detected.quantization_level or "cloud",
            source="ollama-cloud",
            context_length=131072,
            artifact_path=detected.tag,
            notes=[
                "auto-detected from Ollama Cloud (no YAML config)",
                "no local GPU telemetry",
            ],
        )
    return ModelConfig(
        name=slugify_tag(detected.tag),
        family=detected.family or (detected.families[0] if detected.families else "unknown"),
        revision=detected.tag,
        quantization=detected.quantization_level or "ollama-default",
        source="ollama-local",
        context_length=131072,
        artifact_path=detected.tag,
        notes=["auto-detected from Ollama (no YAML config)"],
    )


def synthesize_cloud_model_config(tag: str) -> ModelConfig:
    """Build a ``ModelConfig`` for an Ollama Cloud tag supplied directly.

    Ollama Cloud tags carry a ``-cloud`` suffix (e.g. ``gpt-oss:120b-cloud``)
    and may not appear in ``/api/tags``, so a user can name one explicitly
    and we run it as-is. Marked ``source="ollama-cloud"`` so manifests /
    reports flag the run as remote (no local GPU telemetry).
    """
    return ModelConfig(
        name=slugify_tag(tag),
        family="cloud",
        revision=tag,
        quantization="cloud",
        source="ollama-cloud",
        context_length=131072,
        artifact_path=tag,
        notes=["Ollama Cloud model (tag supplied directly); no local GPU telemetry"],
    )


def detect_ollama_models(
    backend_config: BackendConfig, *, timeout: float = 5.0
) -> list[DetectedOllamaModel]:
    """Probe Ollama for available models, combining local and cloud sources.

    Probes two endpoints independently and merges the results:

    1. **Local Ollama** — always ``http://localhost:11434`` (the Spark daemon),
       no auth. Models here are tagged ``source="local"``.
    2. **Ollama Cloud** — ``https://ollama.com``, only when ``$OLLAMA_API_KEY``
       is set in the environment. Models here are tagged ``source="cloud"``.

    When ``$OLLAMA_HOST`` is set to a non-cloud host (e.g. a remote Ollama
    instance), it replaces the local probe target — the cloud probe still runs
    independently if ``$OLLAMA_API_KEY`` is present.

    Duplicate tags are deduplicated: a tag that exists both locally and in the
    cloud catalogue is kept with ``source="local"`` (local always wins, because
    local runs are free and produce GPU telemetry).

    Returns ``[]`` when neither probe succeeds.
    """
    import os

    from spark_benchmark.runners.ollama import (
        DEFAULT_ENDPOINT,
        ollama_auth_headers,
        resolve_ollama_base,
    )

    detected: list[DetectedOllamaModel] = []
    seen_tags: set[str] = set()

    def _probe_endpoint(base: str, headers: dict[str, str], source: str) -> None:
        tags_url = base + "/api/tags"
        try:
            request = urllib.request.Request(tags_url, headers=headers, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
        except Exception:
            return
        for item in payload.get("models", []):
            name = item.get("name")
            if not name or name in seen_tags:
                continue
            seen_tags.add(name)
            details = item.get("details") or {}
            detected.append(
                DetectedOllamaModel(
                    tag=str(name),
                    family=str(details.get("family") or ""),
                    families=tuple(str(f) for f in (details.get("families") or [])),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization_level=str(details.get("quantization_level") or ""),
                    source=source,
                )
            )

    cloud_base = "https://ollama.com"
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()

    # Probe 1: local Ollama daemon — always http://localhost:11434 (or the
    # backend YAML endpoint), NO auth.  We deliberately ignore $OLLAMA_HOST
    # here because that variable only controls *generation* routing.  Discovery
    # must always reach the local daemon so locally-pulled models are visible
    # even when the user has OLLAMA_HOST=https://ollama.com set for cloud runs.
    local_base = DEFAULT_ENDPOINT.rsplit("/api/", 1)[0]  # http://localhost:11434
    # Respect a custom non-cloud $OLLAMA_HOST (e.g. a remote private Ollama)
    # but skip it when it points at ollama.com (that's the cloud probe below).
    env_host = os.environ.get("OLLAMA_HOST", "").strip()
    if env_host and "ollama.com" not in env_host.lower():
        local_base = env_host if env_host.startswith("http") else "https://" + env_host
    _probe_endpoint(local_base, {}, "local")

    # Probe 2: Ollama Cloud — only when API key is present.
    if api_key:
        _probe_endpoint(cloud_base, ollama_auth_headers(), "cloud")

    return detected


def classify_detected(
    model_configs: Sequence[ModelConfig],
    detected: Sequence[DetectedOllamaModel],
) -> list[OllamaModelInfo]:
    """Classify a list of detected tags into matched / disabled / synthesized.

    - Tags whose ``cfg.artifact_path or cfg.revision`` matches an entry in
      ``model_configs`` are emitted first, in the same order as
      ``model_configs``, with the curated config attached.
    - Remaining vision tags are emitted as ``disable_reason="vision model"``.
    - Remaining embedding tags as ``disable_reason="embedding model"``.
    - Everything else is wrapped into a ``synthesize_model_config`` result
      with ``auto_detected=True``.

    The unmatched groups are returned in tag-sorted order to keep UI
    listings stable across reruns.
    """
    detected_by_tag = {item.tag: item for item in detected}
    items: list[OllamaModelInfo] = []
    matched_tags: set[str] = set()
    for cfg in model_configs:
        tag = cfg.artifact_path or cfg.revision
        if tag in detected_by_tag:
            det = detected_by_tag[tag]
            items.append(OllamaModelInfo(tag=tag, config=cfg, is_cloud=det.source == "cloud"))
            matched_tags.add(tag)
    for tag in sorted(detected_by_tag.keys() - matched_tags):
        info = detected_by_tag[tag]
        is_cloud_model = info.source == "cloud"
        if is_vision_model(info):
            items.append(OllamaModelInfo(tag=tag, config=None, disable_reason="vision model", is_cloud=is_cloud_model))
        elif is_embedding_model(info):
            items.append(
                OllamaModelInfo(tag=tag, config=None, disable_reason="embedding model", is_cloud=is_cloud_model)
            )
        else:
            items.append(
                OllamaModelInfo(
                    tag=tag,
                    config=synthesize_model_config(info),
                    auto_detected=True,
                    is_cloud=is_cloud_model,
                )
            )
    return items


@dataclass
class ResolvedModels:
    """What a CLI command should expose to its users.

    - ``configs`` is the list of usable ``ModelConfig`` (curated + optional
      auto-detected, in that order).
    - ``classified`` is the full classification when auto-detection ran;
      empty list when ``allow_auto_detected=False``. Useful for surfaces
      that want to say e.g. "found 7 Ollama tags, 2 disabled (vision)".
    """

    configs: list[ModelConfig]
    classified: list[OllamaModelInfo]

    @property
    def auto_detected_configs(self) -> list[ModelConfig]:
        return [
            item.config
            for item in self.classified
            if item.auto_detected and item.config is not None
        ]


def resolve_runnable_models(
    *,
    backend_config: BackendConfig,
    experiment_model_configs: Sequence[ModelConfig],
    allow_auto_detected: bool,
) -> ResolvedModels:
    """Return the set of ``ModelConfig`` an entrypoint can offer for a run.

    ``experiment_model_configs`` always wins on naming conflicts: an Ollama
    tag whose synthesized name collides with an experiment model is
    dropped from the auto-detected extras (the curated entry stays).
    """
    if not allow_auto_detected:
        return ResolvedModels(configs=list(experiment_model_configs), classified=[])

    detected = detect_ollama_models(backend_config)
    classified = classify_detected(experiment_model_configs, detected)

    extras: list[ModelConfig] = []
    seen_names = {cfg.name for cfg in experiment_model_configs}
    for item in classified:
        if not item.auto_detected or item.config is None:
            continue
        if item.config.name in seen_names:
            continue
        extras.append(item.config)
        seen_names.add(item.config.name)

    return ResolvedModels(
        configs=list(experiment_model_configs) + extras,
        classified=classified,
    )


def find_config_by_name_or_tag(
    needle: str,
    *,
    configs: Sequence[ModelConfig],
    classified: Sequence[OllamaModelInfo] = (),
) -> ModelConfig | None:
    """Resolve a user-supplied ``--model X`` to a concrete ``ModelConfig``.

    Lookup order:
      1. exact match on ``ModelConfig.name`` in ``configs``;
      2. exact match on the underlying Ollama tag (``artifact_path`` or
         ``revision``) in ``configs``;
      3. exact match on the slugified tag in ``configs`` (so the user can
         pass ``phi4-14b`` even when the tag is ``phi4:14b``);
      4. fallback to any ``OllamaModelInfo`` from ``classified`` whose tag
         matches verbatim — covers the case where the caller didn't pass
         ``--allow-auto-detected`` but supplied a tag we can still see.
    """
    for cfg in configs:
        if cfg.name == needle:
            return cfg
    for cfg in configs:
        if (cfg.artifact_path or cfg.revision) == needle:
            return cfg
    slug = slugify_tag(needle)
    for cfg in configs:
        if cfg.name == slug:
            return cfg
    for item in classified:
        if item.tag == needle and item.config is not None:
            return item.config
    # Explicit Ollama Cloud tag the catalog didn't surface — run it as-is.
    if needle.endswith("-cloud"):
        return synthesize_cloud_model_config(needle)
    return None
