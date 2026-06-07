"""Quantization sweep — post-processor and Pydantic models.

Re-uses existing suite runners. This module groups their results by
base_model × quantization so the HTML renderer can draw the tradeoff
table (quality vs. speed vs. VRAM) described in
docs/quantization-sweep-spec.md.

Public API
----------
load_quant_sweep_fixture   — load data/quant/quantization_sweep_v1.json
aggregate_quant_sweep      — group aggregate_runs() output by base_model
check_quant_regressions    — warn when quality drops >5pp below reference
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from spark_benchmark.models import ModelConfig


# --------------------------------------------------------------------- #
# Fixture schema                                                         #
# --------------------------------------------------------------------- #

class BaseModelSpec(BaseModel):
    base_model: str
    display_name: str
    variants: list[str]
    reference_variant: str
    reference_pass_rates: dict[str, float | None] = Field(default_factory=dict)
    enforce: bool = False


class QuantSweepFixture(BaseModel):
    name: str
    category: str = "quant"
    version: str
    description: str = ""
    notes: list[str] = Field(default_factory=list)
    recommended_suites: list[str] = Field(default_factory=list)
    base_models: list[BaseModelSpec] = Field(default_factory=list)


def load_quant_sweep_fixture(path: Path | str) -> QuantSweepFixture:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return QuantSweepFixture.model_validate(payload)


# --------------------------------------------------------------------- #
# Aggregation                                                            #
# --------------------------------------------------------------------- #

_SUITE_TO_METRIC: dict[str, str] = {
    "hallucination_grounding": "hallucination_pass_rate",
    "practical_structured_output": "structured_output_pass_rate",
    "code_generation": "code_pass_rate",
}


def _normalize_suite_name(name: str) -> str:
    """Strip version suffix so 'hallucination_grounding_v1' → 'hallucination_grounding'."""
    for canonical in _SUITE_TO_METRIC:
        if name == canonical or name.startswith(f"{canonical}_"):
            return canonical
    if name == "openclaw_speed" or name.startswith("openclaw_speed_"):
        return "openclaw_speed"
    return name


def aggregate_quant_sweep(
    aggregate: dict[str, Any],
    model_configs: list[ModelConfig],
    fixture: QuantSweepFixture,
) -> dict[str, Any]:
    """Group ``aggregate_runs()`` output by base_model × quantization.

    Returns a dict keyed by ``base_model`` string. Each value is a dict
    ready to be passed to ``_render_quant_sweep_card`` in reporting_html.
    Models not present in any suite are omitted from the variant list but
    the base_model entry still appears (with fewer rows) so the card
    renders partially rather than crashing.
    """
    # model_name → (base_model, quantization)
    model_meta: dict[str, tuple[str, str]] = {
        mc.name: (mc.base_model, mc.quantization or "default")
        for mc in model_configs
        if mc.base_model
    }

    # Flatten aggregate into model_name → normalized_suite → bucket
    model_suite: dict[str, dict[str, dict[str, Any]]] = {}
    for suite in aggregate.get("suites") or []:
        canonical = _normalize_suite_name(str(suite.get("suite") or ""))
        for model in suite.get("models") or []:
            mname = str(model.get("model") or "")
            model_suite.setdefault(mname, {})[canonical] = model

    result: dict[str, Any] = {}
    for spec in fixture.base_models:
        variants: list[dict[str, Any]] = []
        for variant_name in spec.variants:
            if variant_name not in model_meta:
                continue
            _, quant = model_meta[variant_name]
            s = model_suite.get(variant_name, {})

            hg = s.get("hallucination_grounding") or {}
            pso = s.get("practical_structured_output") or {}
            cg = s.get("code_generation") or {}
            spd = s.get("openclaw_speed") or {}

            peak_vram: float | None = None
            for bucket in s.values():
                v = bucket.get("peak_vram_mb")
                if v is not None:
                    peak_vram = float(v)
                    break

            def _pr(b: dict[str, Any]) -> float | None:
                v = b.get("pass_rate")
                return float(v) if v is not None else None

            def _f(b: dict[str, Any], key: str) -> float | None:
                v = b.get(key)
                return float(v) if v is not None else None

            variants.append({
                "model": variant_name,
                "quantization": quant,
                "hallucination_pass_rate": _pr(hg),
                "structured_output_pass_rate": _pr(pso),
                "code_pass_rate": _pr(cg),
                "ttft_ms": _f(spd, "avg_ttft_ms"),
                "decode_tps": _f(spd, "avg_tokens_per_s"),
                "peak_vram_mb": peak_vram,
            })

        result[spec.base_model] = {
            "base_model": spec.base_model,
            "display_name": spec.display_name,
            "reference_variant": spec.reference_variant,
            "reference_pass_rates": dict(spec.reference_pass_rates),
            "enforce": spec.enforce,
            "variants": variants,
        }

    return result


# --------------------------------------------------------------------- #
# Regression check                                                       #
# --------------------------------------------------------------------- #

_DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data" / "quant" / "quantization_sweep_v1.json"


def enrich_with_quant_sweep(
    aggregate: dict[str, Any],
    model_configs: list[ModelConfig],
    repo_root: Path | None = None,
    fixture_path: Path | None = None,
) -> dict[str, Any]:
    """Inject ``aggregate["quant_sweep"]`` when any model carries a ``base_model``.

    Safe to call unconditionally — returns the aggregate unchanged when no
    model has a ``base_model`` field or the fixture file is missing.
    """
    if not any(mc.base_model for mc in model_configs):
        return aggregate

    path = fixture_path or (
        (repo_root / "data" / "quant" / "quantization_sweep_v1.json") if repo_root else _DEFAULT_FIXTURE_PATH
    )
    if not path.exists():
        return aggregate

    fixture = load_quant_sweep_fixture(path)
    sweep = aggregate_quant_sweep(aggregate, model_configs, fixture)
    if sweep:
        aggregate["quant_sweep"] = sweep
    return aggregate


def check_quant_regressions(
    sweep: dict[str, Any],
    fixture: QuantSweepFixture,
) -> list[str]:
    """Return warning strings for quality regressions vs reference thresholds.

    Only fires when ``enforce=True`` on a base model and the corresponding
    ``reference_pass_rates`` entry is non-null.
    """
    spec_by_base = {s.base_model: s for s in fixture.base_models}
    warnings: list[str] = []

    for base_name, result in sweep.items():
        spec = spec_by_base.get(base_name)
        if not spec or not spec.enforce:
            continue
        ref_name = spec.reference_variant
        ref = next((v for v in result["variants"] if v["model"] == ref_name), None)
        if not ref:
            continue

        checks = [
            ("hallucination_grounding", "hallucination_pass_rate"),
            ("practical_structured_output", "structured_output_pass_rate"),
            ("code_generation", "code_pass_rate"),
        ]
        for variant in result["variants"]:
            if variant["model"] == ref_name:
                continue
            quant = variant["quantization"]
            for suite_key, metric_key in checks:
                threshold = spec.reference_pass_rates.get(suite_key)
                actual = variant.get(metric_key)
                if threshold is None or actual is None:
                    continue
                if actual < threshold - 0.05:
                    warnings.append(
                        f"{result['display_name']} {variant['model']} ({quant}): "
                        f"{suite_key} {actual:.1%} is >5pp below reference {threshold:.1%}"
                    )

    return warnings
