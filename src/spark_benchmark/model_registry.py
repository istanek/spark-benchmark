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


@dataclass
class OllamaModelInfo:
    tag: str
    config: ModelConfig | None
    auto_detected: bool = False
    disable_reason: str | None = None

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
    most chat models), Ollama-default quantization, generic ``ollama-local``
    source, and an explicit note so reports/manifests can flag the run as
    not-fully-curated.
    """
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


def detect_ollama_models(
    backend_config: BackendConfig, *, timeout: float = 5.0
) -> list[DetectedOllamaModel]:
    """Probe ``backend_config.options['endpoint']`` for `/api/tags`.

    Returns ``[]`` on any error (no endpoint, network failure, parse failure)
    so callers can use this as a non-fatal probe.
    """
    endpoint = str(backend_config.options.get("endpoint") or "")
    if not endpoint:
        return []
    tags_url = endpoint.rsplit("/", 1)[0] + "/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as response:
            payload = json.load(response)
    except Exception:
        return []
    detected: list[DetectedOllamaModel] = []
    for item in payload.get("models", []):
        name = item.get("name")
        if not name:
            continue
        details = item.get("details") or {}
        detected.append(
            DetectedOllamaModel(
                tag=str(name),
                family=str(details.get("family") or ""),
                families=tuple(str(f) for f in (details.get("families") or [])),
                parameter_size=str(details.get("parameter_size") or ""),
                quantization_level=str(details.get("quantization_level") or ""),
            )
        )
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
            items.append(OllamaModelInfo(tag=tag, config=cfg))
            matched_tags.add(tag)
    for tag in sorted(detected_by_tag.keys() - matched_tags):
        info = detected_by_tag[tag]
        if is_vision_model(info):
            items.append(OllamaModelInfo(tag=tag, config=None, disable_reason="vision model"))
        elif is_embedding_model(info):
            items.append(
                OllamaModelInfo(tag=tag, config=None, disable_reason="embedding model")
            )
        else:
            items.append(
                OllamaModelInfo(
                    tag=tag,
                    config=synthesize_model_config(info),
                    auto_detected=True,
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
    return None
