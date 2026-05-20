from pathlib import Path

from spark_benchmark.config import load_experiment


def test_load_sample_experiment() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment = load_experiment(repo_root / "configs" / "experiments" / "spark-llamacpp-baseline.yaml")
    assert experiment.experiment.name == "spark-llamacpp-v1-baseline"
