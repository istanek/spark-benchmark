from __future__ import annotations

import platform
import socket
import sys
from pathlib import Path

from spark_benchmark.models import BackendConfig, EnvironmentSnapshot, ExperimentSpec, PlatformConfig, RunManifest


def build_environment_snapshot(platform_config: PlatformConfig, backend_config: BackendConfig) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        platform_name=platform_config.name,
        backend_name=backend_config.name,
        backend_version=backend_config.version,
        python_version=sys.version.split()[0],
        os=platform.platform(),
        hostname=socket.gethostname(),
    )


def build_manifest(
    experiment: ExperimentSpec,
    platform_config: PlatformConfig,
    backend_config: BackendConfig,
    model_names: list[str],
    results_dir: Path,
) -> RunManifest:
    return RunManifest(
        experiment=experiment,
        platform=platform_config,
        backend=backend_config,
        model_names=model_names,
        environment=build_environment_snapshot(platform_config, backend_config),
        results_dir=results_dir,
    )
