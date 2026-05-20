from spark_benchmark.config import load_backend
from spark_benchmark.models import BackendKind
from spark_benchmark.runners.llamacpp import LlamaCppAdapter
from spark_benchmark.runners.registry import build_backend


def test_llamacpp_registry_returns_real_adapter() -> None:
    backend = load_backend(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "backends" / "llamacpp.yaml"
    )
    adapter = build_backend(backend)
    assert backend.name == BackendKind.LLAMACPP
    assert isinstance(adapter, LlamaCppAdapter)
