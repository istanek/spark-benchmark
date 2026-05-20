"""Sustained-throughput benchmark.

Runs each model in a continuous-decode loop for a fixed wall-clock window
(cycling several long prompts) and measures whether short-burst throughput
holds up under sustained pressure. Background NVML / nvidia-smi sampling
captures GPU temperature, power, and throttle reasons; if neither is
available the suite still records token throughput from Ollama's own metrics.

See ``docs/extensions-spec.md`` for the methodology this implements.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from spark_benchmark.models import (
    BackendConfig,
    GenerationResult,
    ModelConfig,
    SamplingConfig,
)
from spark_benchmark.results_bundle import write_json, write_result
from spark_benchmark.suites import SuiteDefinition, load_suite_definition


DEFAULT_DURATION_S = 300.0      # 5 minutes per model
DEFAULT_WINDOW_S = 60.0         # per-minute aggregation windows
DEFAULT_TELEMETRY_HZ = 2.0      # samples per second
DEFAULT_WARMUP_GENERATIONS = 1


@dataclass
class TelemetrySample:
    timestamp_s: float
    gpu_power_w: float | None = None
    gpu_temp_c: float | None = None
    gpu_mem_used_mb: float | None = None
    gpu_clock_mhz: float | None = None
    throttle_reasons: list[str] = field(default_factory=list)
    source: str = "none"


class TelemetrySampler:
    """Background poller for GPU stats.

    Detection order: pynvml → nvidia-smi → none. A missing GPU library is not
    an error — we simply record fewer fields and the suite still produces a
    throughput-only summary.
    """

    NVML_THROTTLE_BITS = [
        ("hw_slowdown", "nvmlClocksThrottleReasonHwSlowdown"),
        ("hw_thermal_slowdown", "nvmlClocksThrottleReasonHwThermalSlowdown"),
        ("hw_power_brake_slowdown", "nvmlClocksThrottleReasonHwPowerBrakeSlowdown"),
        ("sw_thermal_slowdown", "nvmlClocksThrottleReasonSwThermalSlowdown"),
        ("sw_power_cap", "nvmlClocksThrottleReasonSwPowerCap"),
        ("sync_boost", "nvmlClocksThrottleReasonSyncBoost"),
    ]

    def __init__(self, hz: float = DEFAULT_TELEMETRY_HZ) -> None:
        self.hz = max(hz, 0.1)
        self.interval = 1.0 / self.hz
        self.samples: list[TelemetrySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._pynvml: Any | None = None
        self._nvml_handle: Any | None = None
        self._source = self._detect_source()

    @property
    def source(self) -> str:
        return self._source

    def _detect_source(self) -> str:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return "nvml"
        except Exception:
            self._pynvml = None
            self._nvml_handle = None
        if shutil.which("nvidia-smi"):
            return "nvidia-smi"
        return "none"

    def start(self, t0: float) -> None:
        self._t0 = t0
        if self._source == "none":
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._source == "nvml" and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def _loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            sample = self._poll()
            if sample is not None:
                self.samples.append(sample)
            next_t += self.interval
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                self._stop.wait(timeout=sleep_s)

    def _poll(self) -> TelemetrySample | None:
        ts = time.monotonic() - self._t0
        if self._source == "nvml":
            return self._poll_nvml(ts)
        if self._source == "nvidia-smi":
            return self._poll_smi(ts)
        return None

    def _poll_nvml(self, ts: float) -> TelemetrySample | None:
        pynvml = self._pynvml
        handle = self._nvml_handle
        if pynvml is None or handle is None:
            return None
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            power_w = None
        try:
            temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            temp_c = None
        mem_used: float | None = None
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = mem.used / (1024 * 1024)
        except Exception:
            pass
        clock_mhz: float | None = None
        try:
            clock_mhz = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
        except Exception:
            pass
        throttle: list[str] = []
        try:
            flags = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            for label, attr in self.NVML_THROTTLE_BITS:
                bit = getattr(pynvml, attr, 0)
                if bit and (flags & bit):
                    throttle.append(label)
        except Exception:
            pass
        return TelemetrySample(
            timestamp_s=ts,
            gpu_power_w=power_w,
            gpu_temp_c=temp_c,
            gpu_mem_used_mb=mem_used,
            gpu_clock_mhz=clock_mhz,
            throttle_reasons=throttle,
            source="nvml",
        )

    def _poll_smi(self, ts: float) -> TelemetrySample | None:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=power.draw,temperature.gpu,memory.used,clocks.gr",
                "--format=csv,noheader,nounits",
                "-i", "0",
            ]
            out = subprocess.run(cmd, capture_output=True, timeout=2.0, text=True)
            if out.returncode != 0:
                return None
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) < 4:
                return None
            def _maybe_float(v: str) -> float | None:
                if not v or v in {"[N/A]", "N/A"}:
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None
            return TelemetrySample(
                timestamp_s=ts,
                gpu_power_w=_maybe_float(parts[0]),
                gpu_temp_c=_maybe_float(parts[1]),
                gpu_mem_used_mb=_maybe_float(parts[2]),
                gpu_clock_mhz=_maybe_float(parts[3]),
                source="nvidia-smi",
            )
        except Exception:
            return None


@dataclass
class GenerationRecord:
    seq: int
    prompt_family: str
    task_id: str
    started_s: float
    finished_s: float
    decode_tokens: int
    decode_time_s: float
    ttft_ms: float

    @property
    def tokens_per_s(self) -> float:
        return (self.decode_tokens / self.decode_time_s) if self.decode_time_s > 0 else 0.0


def compute_windows(
    records: list[GenerationRecord],
    duration_s: float,
    window_s: float,
) -> list[dict[str, Any]]:
    """Aggregate decode tokens into fixed wall-clock windows."""
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    n_windows = max(1, math.ceil(duration_s / window_s))
    windows: list[dict[str, Any]] = [
        {
            "window_index": i,
            "start_s": i * window_s,
            "end_s": (i + 1) * window_s,
            "decode_tokens": 0.0,
            "decode_time_s": 0.0,
            "tokens_per_s": 0.0,
            "generations_completed": 0,
        }
        for i in range(n_windows)
    ]
    for record in records:
        if record.decode_time_s <= 0 or record.decode_tokens <= 0:
            continue
        gen_start = record.finished_s - record.decode_time_s
        gen_end = record.finished_s
        tok_per_s = record.decode_tokens / record.decode_time_s
        for window in windows:
            overlap = max(0.0, min(gen_end, window["end_s"]) - max(gen_start, window["start_s"]))
            if overlap > 0:
                window["decode_time_s"] += overlap
                window["decode_tokens"] += tok_per_s * overlap
            if window["start_s"] <= gen_end < window["end_s"]:
                window["generations_completed"] += 1
    for window in windows:
        if window["decode_time_s"] > 0:
            window["tokens_per_s"] = round(window["decode_tokens"] / window["decode_time_s"], 2)
        window["decode_tokens"] = int(round(window["decode_tokens"]))
        window["decode_time_s"] = round(window["decode_time_s"], 3)
    return windows


def compute_derived_metrics(
    records: list[GenerationRecord],
    windows: list[dict[str, Any]],
    samples: list[TelemetrySample],
    duration_s: float,
) -> dict[str, Any]:
    """Distil the sustained-throughput KPIs the spec lists."""
    populated = [w for w in windows if w["decode_time_s"] > 0]
    initial = populated[0]["tokens_per_s"] if populated else 0.0
    sustained = populated[-1]["tokens_per_s"] if populated else 0.0
    peak = max((w["tokens_per_s"] for w in windows), default=0.0)

    throttle_ratio = round(sustained / initial, 3) if initial > 0 else None
    time_to_throttle_s: float | None = None
    drop_threshold = peak * 0.9
    if peak > 0:
        for window in windows:
            if window["tokens_per_s"] > 0 and window["tokens_per_s"] < drop_threshold:
                time_to_throttle_s = window["start_s"]
                break

    avg_power: float | None = None
    if samples:
        powers = [s.gpu_power_w for s in samples if s.gpu_power_w is not None]
        if powers:
            avg_power = sum(powers) / len(powers)

    total_decode_tokens = sum(r.decode_tokens for r in records)
    energy_j_per_token: float | None = None
    if avg_power is not None and total_decode_tokens > 0 and duration_s > 0:
        energy_j_per_token = round((avg_power * duration_s) / total_decode_tokens, 4)

    peak_temp: float | None = None
    if samples:
        temps = [s.gpu_temp_c for s in samples if s.gpu_temp_c is not None]
        if temps:
            peak_temp = max(temps)

    throttle_reasons = sorted({reason for s in samples for reason in s.throttle_reasons})

    return {
        "initial_tokens_per_s": round(initial, 2),
        "sustained_tokens_per_s": round(sustained, 2),
        "peak_tokens_per_s": round(peak, 2),
        "throttle_ratio": throttle_ratio,
        "time_to_throttle_s": round(time_to_throttle_s, 1) if time_to_throttle_s is not None else None,
        "avg_power_w": round(avg_power, 2) if avg_power is not None else None,
        "peak_temp_c": peak_temp,
        "energy_j_per_token": energy_j_per_token,
        "throttle_reasons_observed": throttle_reasons,
    }


def run_sustained_throughput_suite(
    *,
    run_dir: Path,
    suite: SuiteDefinition,
    backend: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    sampling: SamplingConfig,
    duration_seconds: float = DEFAULT_DURATION_S,
    window_seconds: float = DEFAULT_WINDOW_S,
    telemetry_hz: float = DEFAULT_TELEMETRY_HZ,
    warmup_generations: int = DEFAULT_WARMUP_GENERATIONS,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not suite.tasks:
        raise ValueError("sustained_throughput suite must declare at least one prompt task")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    # The probe wants long outputs; never run with absurdly low max_tokens.
    sus_sampling = sampling.model_copy(update={"max_tokens": max(sampling.max_tokens, 512)})

    summary_models: list[dict[str, Any]] = []
    for model_config in model_configs:
        sampler = TelemetrySampler(hz=telemetry_hz)
        records: list[GenerationRecord] = []

        if progress_callback:
            progress_callback(
                f"  loading {model_config.name} for sustained probe "
                f"(telemetry={sampler.source})"
            )
        backend.load_model(model_config)

        # Warmup — discard token counts so cold-start doesn't bias the first window.
        for _ in range(max(0, warmup_generations)):
            try:
                backend.generate(suite.tasks[0].prompt, sus_sampling)
            except Exception:
                pass

        if progress_callback:
            progress_callback(
                f"  {model_config.name} → starting {int(duration_seconds)}s sustained loop"
            )

        t0 = time.monotonic()
        sampler.start(t0)
        seq = 0
        prompts = list(suite.tasks)
        try:
            while time.monotonic() - t0 < duration_seconds:
                task = prompts[seq % len(prompts)]
                start_offset = time.monotonic() - t0
                generation: GenerationResult = backend.generate(task.prompt, sus_sampling)
                end_offset = time.monotonic() - t0
                metrics = generation.metrics
                record = GenerationRecord(
                    seq=seq,
                    prompt_family=str(task.metadata.get("prompt_family") or task.task_id),
                    task_id=task.task_id,
                    started_s=start_offset,
                    finished_s=end_offset,
                    decode_tokens=int(metrics.decode_tokens or 0),
                    decode_time_s=float(metrics.decode_time_s or 0.0),
                    ttft_ms=float(metrics.ttft_ms or 0.0),
                )
                records.append(record)
                row = {
                    "suite": suite.name,
                    "suite_version": suite.version,
                    "model": model_config.name,
                    "model_tag": model_config.artifact_path or model_config.revision,
                    "task_id": task.task_id,
                    "tags": task.tags,
                    "prompt": task.prompt,
                    "context": task.context,
                    "reference": task.reference,
                    "generation": generation.model_dump(mode="json"),
                    "evaluation": {
                        "expected_behavior": "sustained_throughput_probe",
                        "passed": True,
                        "score": 1,
                        "reason": "captured_metrics",
                        "matched_reference_tokens": [],
                        "iteration": seq,
                        "wall_time_started_s": round(record.started_s, 3),
                        "wall_time_finished_s": round(record.finished_s, 3),
                    },
                }
                write_result(run_dir, row)
                seq += 1
                if progress_callback and seq % 3 == 0:
                    elapsed = time.monotonic() - t0
                    progress_callback(
                        f"  {model_config.name} → {seq} gens, {elapsed:.0f}s/{int(duration_seconds)}s, "
                        f"last {record.tokens_per_s:.1f} tok/s"
                    )
        finally:
            sampler.stop()
            if progress_callback:
                progress_callback(f"  unloading {model_config.name} from Ollama")
            backend.unload()

        windows = compute_windows(records, duration_seconds, window_seconds)
        derived = compute_derived_metrics(records, windows, sampler.samples, duration_seconds)
        total_decode_tokens = sum(r.decode_tokens for r in records)
        total_decode_time = round(sum(r.decode_time_s for r in records), 3)

        summary_models.append(
            {
                "model": model_config.name,
                "model_tag": model_config.artifact_path or model_config.revision,
                # `passes` / `total` keep the cross-suite aggregator happy; the
                # suite never fails a quality check, so every completed
                # generation counts as a pass.
                "passes": len(records),
                "total": len(records),
                "generations": len(records),
                "total_decode_tokens": total_decode_tokens,
                "total_decode_time_s": total_decode_time,
                "telemetry_source": sampler.source,
                "telemetry_samples": len(sampler.samples),
                **derived,
                "windows": windows,
            }
        )

        telemetry_path = run_dir / f"telemetry-{model_config.name}.jsonl"
        with telemetry_path.open("w", encoding="utf-8") as fh:
            for sample in sampler.samples:
                fh.write(
                    json.dumps(
                        {
                            "timestamp_s": round(sample.timestamp_s, 3),
                            "gpu_power_w": sample.gpu_power_w,
                            "gpu_temp_c": sample.gpu_temp_c,
                            "gpu_mem_used_mb": sample.gpu_mem_used_mb,
                            "gpu_clock_mhz": sample.gpu_clock_mhz,
                            "throttle_reasons": sample.throttle_reasons,
                            "source": sample.source,
                        }
                    )
                    + "\n"
                )

    summary = {
        "suite": suite.name,
        "suite_version": suite.version,
        "backend": backend_config.name.value,
        "duration_seconds": duration_seconds,
        "window_seconds": window_seconds,
        "warmup_generations": warmup_generations,
        "telemetry_hz": telemetry_hz,
        "models": summary_models,
    }
    write_json(run_dir / "summary.json", summary)
    write_summary_markdown(run_dir, summary)
    return summary


def write_summary_markdown(run_dir: Path, summary: dict[str, Any]) -> Path:
    def fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:g}{suffix}"
        return f"{value}{suffix}"

    lines = [
        f"# {summary['suite']} summary",
        "",
        f"- backend: {summary['backend']}",
        f"- duration: {summary['duration_seconds']:.0f} s",
        f"- window: {summary['window_seconds']:.0f} s",
        f"- warmup generations: {summary['warmup_generations']}",
        "",
        "| model | gens | init tok/s | sustained tok/s | peak tok/s | throttle ratio | time→throttle | avg W | peak °C | J/tok | telemetry |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in summary["models"]:
        ttt = model.get("time_to_throttle_s")
        ttt_str = "-" if ttt is None else f"{ttt:.0f} s"
        lines.append(
            "| {model} | {gens} | {init} | {sustained} | {peak} | {ratio} | {ttt} | {power} | {temp} | {energy} | {tel} |".format(
                model=model["model"],
                gens=model["generations"],
                init=fmt(model.get("initial_tokens_per_s")),
                sustained=fmt(model.get("sustained_tokens_per_s")),
                peak=fmt(model.get("peak_tokens_per_s")),
                ratio=fmt(model.get("throttle_ratio")),
                ttt=ttt_str,
                power=fmt(model.get("avg_power_w")),
                temp=fmt(model.get("peak_temp_c")),
                energy=fmt(model.get("energy_j_per_token")),
                tel=model.get("telemetry_source") or "none",
            )
        )
        reasons = model.get("throttle_reasons_observed") or []
        if reasons:
            lines.append(f"  - throttle reasons observed: {', '.join(reasons)}")
    path = run_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_sustained_throughput_suite(repo_root: Path) -> SuiteDefinition:
    return load_suite_definition(repo_root / "data" / "performance" / "sustained_throughput_v1.json")
