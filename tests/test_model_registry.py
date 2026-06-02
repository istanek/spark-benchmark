import os

from spark_benchmark import model_registry
from spark_benchmark.model_registry import (
    DetectedOllamaModel,
    classify_detected,
    find_config_by_name_or_tag,
    is_embedding_model,
    is_vision_model,
    resolve_runnable_models,
    slugify_tag,
    synthesize_cloud_model_config,
    synthesize_model_config,
)
from spark_benchmark.models import BackendConfig, BackendKind, ModelConfig
from spark_benchmark.runners.ollama import (
    is_cloud_endpoint,
    ollama_auth_headers,
    resolve_ollama_base,
)


def _clear_ollama_env() -> dict[str, str | None]:
    saved = {k: os.environ.get(k) for k in ("OLLAMA_HOST", "OLLAMA_API_KEY")}
    for k in saved:
        os.environ.pop(k, None)
    return saved


def _restore_env(saved: dict[str, str | None]) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _make_backend() -> BackendConfig:
    return BackendConfig(
        name=BackendKind.OLLAMA,
        entrypoint="ollama",
        version="local",
        transport="http",
        options={"endpoint": "http://localhost:11434/api/generate"},
    )


def _make_model(name: str, tag: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        family=name.split("-")[0],
        revision=tag,
        quantization="ollama-default",
        source="ollama-local",
        context_length=4096,
        artifact_path=tag,
    )


def _detected(tag: str, family: str = "", families: tuple[str, ...] = ()) -> DetectedOllamaModel:
    return DetectedOllamaModel(tag=tag, family=family, families=families)


def test_slugify_tag_replaces_colon_and_slash() -> None:
    assert slugify_tag("phi4:14b") == "phi4-14b"
    assert slugify_tag("hf.co/foo/bar:Q4") == "hf.co_foo_bar-Q4"


def test_synthesize_model_config_defaults() -> None:
    cfg = synthesize_model_config(_detected("phi4:14b", family="phi", families=("phi", "transformer")))
    assert cfg.name == "phi4-14b"
    assert cfg.family == "phi"
    assert cfg.revision == "phi4:14b"
    assert cfg.artifact_path == "phi4:14b"
    assert cfg.quantization == "ollama-default"
    assert cfg.source == "ollama-local"
    assert cfg.context_length == 131072
    assert "auto-detected" in cfg.notes[0].lower()


def test_synthesize_model_config_falls_back_when_family_missing() -> None:
    cfg = synthesize_model_config(_detected("mystery:latest"))
    assert cfg.family == "unknown"


def test_classify_detected_keeps_curated_order_and_filters_vision_embedding() -> None:
    curated = [
        _make_model("qwen-3.6", "qwen3.6:35b"),
        _make_model("gemma-4", "gemma4:31b"),
    ]
    detected = [
        _detected("qwen3.6:35b", family="qwen35moe"),
        _detected("nomic-embed-text:latest", family="nomic-bert"),
        _detected("qwen3-vl:30b", family="qwen3vlmoe"),
        _detected("phi4:14b", family="phi"),
    ]
    classified = classify_detected(curated, detected)
    by_tag = {item.tag: item for item in classified}

    assert by_tag["qwen3.6:35b"].has_config
    assert by_tag["qwen3.6:35b"].auto_detected is False

    assert by_tag["nomic-embed-text:latest"].disable_reason == "embedding model"
    assert by_tag["qwen3-vl:30b"].disable_reason == "vision model"

    assert by_tag["phi4:14b"].auto_detected is True
    assert by_tag["phi4:14b"].config is not None
    assert by_tag["phi4:14b"].config.name == "phi4-14b"

    # Curated configs that aren't currently in Ollama just don't appear.
    assert "gemma4:31b" not in by_tag


def test_resolve_runnable_models_default_returns_curated_only(monkeypatch) -> None:
    curated = [_make_model("qwen-3.6", "qwen3.6:35b")]

    def _boom(*args, **kwargs):
        raise AssertionError("detect_ollama_models must not be called when allow_auto_detected=False")

    monkeypatch.setattr(model_registry, "detect_ollama_models", _boom)

    resolved = resolve_runnable_models(
        backend_config=_make_backend(),
        experiment_model_configs=curated,
        allow_auto_detected=False,
    )
    assert resolved.configs == curated
    assert resolved.classified == []
    assert resolved.auto_detected_configs == []


def test_resolve_runnable_models_appends_auto_detected_extras(monkeypatch) -> None:
    curated = [_make_model("qwen-3.6", "qwen3.6:35b")]
    detected = [
        _detected("qwen3.6:35b", family="qwen35moe"),
        _detected("phi4:14b", family="phi"),
        _detected("nomic-embed-text:latest", family="nomic-bert"),
        _detected("llava:13b", family="llama"),
    ]
    monkeypatch.setattr(model_registry, "detect_ollama_models", lambda backend_config: detected)

    resolved = resolve_runnable_models(
        backend_config=_make_backend(),
        experiment_model_configs=curated,
        allow_auto_detected=True,
    )
    names = [cfg.name for cfg in resolved.configs]
    assert names == ["qwen-3.6", "phi4-14b"]
    assert all(cfg.notes for cfg in resolved.auto_detected_configs)
    assert resolved.auto_detected_configs[0].name == "phi4-14b"


def test_resolve_runnable_models_skips_collisions(monkeypatch) -> None:
    curated_cfg = ModelConfig(
        name="phi4-14b",
        family="phi",
        revision="phi4:14b",
        quantization="ollama-default",
        source="ollama-local",
        context_length=4096,
        artifact_path="phi4:14b",
        notes=["curated"],
    )
    detected = [_detected("phi4:14b", family="phi")]
    monkeypatch.setattr(model_registry, "detect_ollama_models", lambda backend_config: detected)

    resolved = resolve_runnable_models(
        backend_config=_make_backend(),
        experiment_model_configs=[curated_cfg],
        allow_auto_detected=True,
    )
    assert [cfg.name for cfg in resolved.configs] == ["phi4-14b"]
    assert resolved.configs[0].notes == ["curated"]
    assert resolved.auto_detected_configs == []


def test_find_config_by_name_or_tag_resolution_order() -> None:
    cfg = _make_model("qwen-3.6", "qwen3.6:35b")
    auto = synthesize_model_config(_detected("phi4:14b", family="phi"))
    configs = [cfg, auto]
    classified = []  # don't need this path here

    assert find_config_by_name_or_tag("qwen-3.6", configs=configs, classified=classified) is cfg
    assert find_config_by_name_or_tag("qwen3.6:35b", configs=configs, classified=classified) is cfg
    assert find_config_by_name_or_tag("phi4:14b", configs=configs, classified=classified) is auto
    assert find_config_by_name_or_tag("phi4-14b", configs=configs, classified=classified) is auto
    assert find_config_by_name_or_tag("does-not-exist", configs=configs, classified=classified) is None


def test_vision_and_embedding_detectors_match_shell_expectations() -> None:
    assert is_vision_model(_detected("qwen3-vl:30b", family="qwen3vlmoe"))
    assert is_vision_model(_detected("hf.co/x/pixtral-12b:Q4", family="llama"))
    assert is_vision_model(_detected("llava:13b", family="llama"))
    assert not is_vision_model(_detected("qwen3.6:35b", family="qwen35moe"))

    assert is_embedding_model(_detected("bge-m3:latest", family="bert"))
    assert is_embedding_model(_detected("nomic-embed-text:latest", family="nomic-bert"))
    assert not is_embedding_model(_detected("gpt-oss:120b", family="gptoss"))


def _run_all() -> int:
    import inspect
    import sys

    failures: list[str] = []
    module = sys.modules[__name__]
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        sig = inspect.signature(fn)
        if "monkeypatch" in sig.parameters:
            # Plain-python fallback for tests that need pytest's monkeypatch.
            mp = _PlainMonkeypatch()
            try:
                fn(mp)
                print(f"ok  {name}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {exc!r}")
                print(f"FAIL {name}: {exc!r}")
            finally:
                mp.undo()
            continue
        try:
            fn()
            print(f"ok  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc!r}")
            print(f"FAIL {name}: {exc!r}")
    return 1 if failures else 0


class _PlainMonkeypatch:
    """Tiny stand-in for pytest's monkeypatch when running without pytest."""

    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, target: object, name: str, value: object) -> None:
        original = getattr(target, name)
        self._undo.append((target, name, original))
        setattr(target, name, value)

    def undo(self) -> None:
        for target, name, original in reversed(self._undo):
            setattr(target, name, original)
        self._undo.clear()


def test_resolve_ollama_base_prefers_env_host() -> None:
    saved = _clear_ollama_env()
    try:
        os.environ["OLLAMA_HOST"] = "https://ollama.com"
        assert resolve_ollama_base({"endpoint": "http://localhost:11434/api/generate"}) == "https://ollama.com"
        # Bare host gets https://.
        os.environ["OLLAMA_HOST"] = "ollama.com"
        assert resolve_ollama_base({}) == "https://ollama.com"
    finally:
        _restore_env(saved)


def test_resolve_ollama_base_falls_back_to_option_then_default() -> None:
    saved = _clear_ollama_env()
    try:
        assert resolve_ollama_base({"endpoint": "http://host:9/api/generate"}) == "http://host:9"
        assert resolve_ollama_base({}) == "http://localhost:11434"
    finally:
        _restore_env(saved)


def test_ollama_auth_headers_from_env() -> None:
    saved = _clear_ollama_env()
    try:
        assert ollama_auth_headers() == {}
        os.environ["OLLAMA_API_KEY"] = "sk-test-123"
        assert ollama_auth_headers() == {"Authorization": "Bearer sk-test-123"}
        assert is_cloud_endpoint() is True
    finally:
        _restore_env(saved)


def test_find_config_synthesizes_cloud_tag() -> None:
    cfg = find_config_by_name_or_tag("gpt-oss:120b-cloud", configs=[], classified=[])
    assert cfg is not None
    assert cfg.artifact_path == "gpt-oss:120b-cloud"
    assert cfg.source == "ollama-cloud"


def test_find_config_non_cloud_unknown_returns_none() -> None:
    assert find_config_by_name_or_tag("does-not-exist", configs=[], classified=[]) is None


def test_synthesize_cloud_model_config_fields() -> None:
    cfg = synthesize_cloud_model_config("deepseek-v3.1:671b-cloud")
    assert cfg.name == "deepseek-v3.1-671b-cloud"
    assert cfg.revision == "deepseek-v3.1:671b-cloud"
    assert cfg.source == "ollama-cloud"
    assert any("no local GPU telemetry" in n for n in cfg.notes)


if __name__ == "__main__":
    import sys

    sys.exit(_run_all())
