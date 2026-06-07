import json
import tempfile
from pathlib import Path

from spark_benchmark.models import BackendConfig, BackendKind, ModelConfig
from spark_benchmark.shell import (
    SUITE_REGISTRY,
    CustomSuiteCandidate,
    DetectedOllamaModel,
    ShellContext,
    classify_models,
    discover_custom_suites,
    is_embedding_model,
    is_vision_model,
    load_default_context,
    load_suite_metadata,
    missing_haystacks,
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


def _make_shell_context(tmp_path: Path) -> ShellContext:
    """Minimal ShellContext that doesn't need real YAML configs on disk."""
    real = load_default_context()
    return ShellContext(
        repo_root=tmp_path,
        experiment=real.experiment,
        platform=real.platform,
        backend_config=real.backend_config,
        model_configs=[],
    )


def _detected(tag: str, family: str = "", families: tuple[str, ...] = ()) -> DetectedOllamaModel:
    return DetectedOllamaModel(tag=tag, family=family, families=families)


def test_classify_models_disables_embedding_and_vision_extras() -> None:
    ctx = ShellContext(
        repo_root=load_default_context().repo_root,
        experiment=load_default_context().experiment,
        platform=load_default_context().platform,
        backend_config=BackendConfig(
            name=BackendKind.OLLAMA,
            entrypoint="ollama",
            version="local",
            transport="http",
            options={"endpoint": "http://localhost:11434/api/generate"},
        ),
        model_configs=[_make_model("qwen-3.6", "qwen3.6:35b"), _make_model("gemma-4", "gemma4:31b")],
    )
    detected = [
        _detected("qwen3.6:35b", family="qwen35moe"),
        _detected("bge-m3:latest", family="bert"),
        _detected("nomic-embed-text:latest", family="nomic-bert"),
        _detected("qwen3-vl:30b", family="qwen3vlmoe"),
    ]
    classified = classify_models(ctx, detected)

    by_tag = {item.tag: item for item in classified}
    assert by_tag["qwen3.6:35b"].has_config
    assert by_tag["qwen3.6:35b"].auto_detected is False
    assert by_tag["bge-m3:latest"].has_config is False
    assert by_tag["bge-m3:latest"].disable_reason == "embedding model"
    assert by_tag["nomic-embed-text:latest"].disable_reason == "embedding model"
    assert by_tag["qwen3-vl:30b"].disable_reason == "vision model"
    assert "gemma4:31b" not in by_tag, "configured model not in Ollama should not be returned"


def test_classify_models_auto_synthesizes_non_vision_extras() -> None:
    ctx = ShellContext(
        repo_root=load_default_context().repo_root,
        experiment=load_default_context().experiment,
        platform=load_default_context().platform,
        backend_config=BackendConfig(
            name=BackendKind.OLLAMA, entrypoint="ollama", version="local", transport="http"
        ),
        model_configs=[],
    )
    detected = [
        _detected("z-model:8b", family="llama"),
        _detected("a-model:7b", family="llama"),
        _detected("m-model:4b", family="llama"),
    ]
    classified = classify_models(ctx, detected)
    assert [item.tag for item in classified] == ["a-model:7b", "m-model:4b", "z-model:8b"]
    assert all(item.has_config for item in classified)
    assert all(item.auto_detected for item in classified)
    # Auto-synthesized configs carry the Ollama tag as artifact_path.
    assert classified[0].config is not None
    assert classified[0].config.artifact_path == "a-model:7b"


def test_vision_and_embedding_detectors() -> None:
    assert is_vision_model(_detected("qwen3-vl:30b", family="qwen3vlmoe"))
    assert is_vision_model(_detected("hf.co/x/pixtral-12b:Q4", family="llama"))
    assert is_vision_model(_detected("llava:13b", family="llama"))
    assert not is_vision_model(_detected("qwen3.6:35b", family="qwen35moe"))

    assert is_embedding_model(_detected("bge-m3:latest", family="bert"))
    assert is_embedding_model(_detected("nomic-embed-text:latest", family="nomic-bert"))
    assert not is_embedding_model(_detected("gpt-oss:120b", family="gptoss"))


def test_load_suite_metadata_returns_expected_fields() -> None:
    ctx = load_default_context()
    for name in SUITE_REGISTRY:
        meta = load_suite_metadata(ctx.repo_root, name)
        assert meta is not None, f"suite metadata missing for {name}"
        assert meta.get("name")
        assert meta.get("description")
        # Task-list suites carry a non-empty ``tasks`` array; grid-based
        # suites (long_context_retrieval) carry a ``test_matrix`` instead.
        if meta.get("test_matrix"):
            assert meta["test_matrix"].get("context_lengths_tokens")
        else:
            assert isinstance(meta.get("tasks"), list)
            assert len(meta["tasks"]) > 0


def test_load_suite_metadata_unknown_suite_returns_none() -> None:
    ctx = load_default_context()
    assert load_suite_metadata(ctx.repo_root, "no_such_suite") is None


def test_missing_haystacks_flags_unfetched_corpora() -> None:
    ctx = load_default_context()
    # The corpora are git-ignored and not present in a fresh checkout, so
    # the long-context suite should report them as missing (unless a dev
    # has fetched them locally — then the list is simply empty).
    missing = missing_haystacks(ctx.repo_root, "long_context_retrieval")
    haystack_dir = ctx.repo_root / "data" / "long_context" / "haystacks"
    fetched = list(haystack_dir.glob("*.txt")) if haystack_dir.exists() else []
    if fetched:
        assert missing == []
    else:
        assert missing  # at least one text_file flagged
        assert all(path.endswith(".txt") for path in missing)


def test_missing_haystacks_empty_for_task_suites() -> None:
    ctx = load_default_context()
    # Suites without a needs_haystacks marker never require a fetch.
    assert missing_haystacks(ctx.repo_root, "openclaw_speed") == []


def test_fast_profile_registered_and_metadata_uses_fast_grid() -> None:
    ctx = load_default_context()
    assert "long_context_retrieval_fast" in SUITE_REGISTRY
    full = load_suite_metadata(ctx.repo_root, "long_context_retrieval")
    fast = load_suite_metadata(ctx.repo_root, "long_context_retrieval_fast")
    assert full is not None and fast is not None
    # The fast entry surfaces the fast profile's (smaller) grid.
    full_cells = (
        len(full["test_matrix"]["context_lengths_tokens"])
        * len(full["test_matrix"]["depth_percentages"])
        * full["test_matrix"]["needles_per_cell"]
    )
    fast_cells = (
        len(fast["test_matrix"]["context_lengths_tokens"])
        * len(fast["test_matrix"]["depth_percentages"])
        * fast["test_matrix"]["needles_per_cell"]
    )
    assert fast_cells < full_cells


def test_discover_custom_suites_finds_examples_and_recent_runs() -> None:
    """Discovery surfaces shipped examples and recent custom-runs."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        # Shipped example.
        ex_dir = repo / "examples" / "custom-tests" / "demo"
        ex_dir.mkdir(parents=True)
        ex_yaml = ex_dir / "suite.yaml"
        ex_yaml.write_text("name: demo\n", encoding="utf-8")
        # User-written suite + two prior custom runs (older + newer) pointing
        # at the same suite_path; the newer run-id wins.
        user_yaml = repo / "user-suite.yaml"
        user_yaml.write_text("name: user\n", encoding="utf-8")
        for run_id in ("20260520-100000", "20260522-090000"):
            run_dir = repo / "results" / "custom" / "user" / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "custom",
                        "run_id": run_id,
                        "suite": "user",
                        "suite_path": str(user_yaml.resolve()),
                    }
                ),
                encoding="utf-8",
            )

        results = discover_custom_suites(repo)
        # 1 example + 1 deduped recent run.
        assert len(results) == 2
        assert all(isinstance(c, CustomSuiteCandidate) for c in results)
        assert results[0].origin == "example"
        assert results[0].path == ex_yaml.resolve()
        assert results[1].origin == "recent"
        assert results[1].path == user_yaml.resolve()
        assert results[1].last_run == "20260522-090000"


def test_discover_custom_suites_empty_when_nothing_present() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert discover_custom_suites(Path(tmp)) == []


def test_discover_custom_suites_skips_recent_pointing_at_missing_file() -> None:
    """A manifest pointing at a now-deleted suite_path is ignored."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        run_dir = repo / "results" / "custom" / "ghost" / "20260522-100000"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "kind": "custom",
                    "run_id": "20260522-100000",
                    "suite": "ghost",
                    "suite_path": str(repo / "missing-suite.yaml"),
                }
            ),
            encoding="utf-8",
        )
        assert discover_custom_suites(repo) == []


def test_do_cloud_sets_api_key() -> None:
    """do_cloud() stores the entered key in os.environ and reloads context."""
    import os
    import tempfile
    from unittest.mock import MagicMock, patch

    from spark_benchmark.shell import TUIApp

    saved = os.environ.pop("OLLAMA_API_KEY", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _make_shell_context(Path(tmp))
            app = TUIApp(ctx=ctx)
            stdscr = MagicMock()
            with patch("curses.endwin"):
                with patch("builtins.input", return_value="sk-testkey123"):
                    with patch("spark_benchmark.shell.load_default_context", return_value=ctx) as mock_load:
                        app.do_cloud(stdscr)
            assert os.environ.get("OLLAMA_API_KEY") == "sk-testkey123"
            mock_load.assert_called_once()
    finally:
        os.environ.pop("OLLAMA_API_KEY", None)
        if saved is not None:
            os.environ["OLLAMA_API_KEY"] = saved


def test_do_cloud_clears_key_on_dash() -> None:
    """Entering '-' removes OLLAMA_API_KEY from the environment."""
    import os
    import tempfile
    from unittest.mock import MagicMock, patch

    from spark_benchmark.shell import TUIApp

    saved = os.environ.get("OLLAMA_API_KEY")
    os.environ["OLLAMA_API_KEY"] = "old-key"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _make_shell_context(Path(tmp))
            app = TUIApp(ctx=ctx)
            stdscr = MagicMock()
            with patch("curses.endwin"):
                with patch("builtins.input", return_value="-"):
                    with patch("spark_benchmark.shell.load_default_context", return_value=ctx):
                        app.do_cloud(stdscr)
            assert "OLLAMA_API_KEY" not in os.environ
    finally:
        if saved is not None:
            os.environ["OLLAMA_API_KEY"] = saved
        else:
            os.environ.pop("OLLAMA_API_KEY", None)


def test_do_cloud_empty_input_keeps_existing_key() -> None:
    """Empty input keeps the existing key and does NOT reload context."""
    import os
    import tempfile
    from unittest.mock import MagicMock, patch

    from spark_benchmark.shell import TUIApp

    saved = os.environ.get("OLLAMA_API_KEY")
    os.environ["OLLAMA_API_KEY"] = "existing-key"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _make_shell_context(Path(tmp))
            app = TUIApp(ctx=ctx)
            stdscr = MagicMock()
            with patch("curses.endwin"):
                with patch("builtins.input", return_value=""):
                    with patch("spark_benchmark.shell.load_default_context", return_value=ctx) as mock_load:
                        app.do_cloud(stdscr)
            assert os.environ.get("OLLAMA_API_KEY") == "existing-key"
            mock_load.assert_not_called()
    finally:
        if saved is not None:
            os.environ["OLLAMA_API_KEY"] = saved
        else:
            os.environ.pop("OLLAMA_API_KEY", None)


def _run_all() -> int:
    import inspect
    import sys

    failures: list[str] = []
    module = sys.modules[__name__]
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"ok  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc!r}")
            print(f"FAIL {name}: {exc!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_all())
