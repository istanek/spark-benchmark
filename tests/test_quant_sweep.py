"""Tests for spark_benchmark.quant_sweep."""

from pathlib import Path

import pytest

from spark_benchmark.models import ModelConfig
from spark_benchmark.quant_sweep import (
    BaseModelSpec,
    QuantSweepFixture,
    aggregate_quant_sweep,
    check_quant_regressions,
    enrich_with_quant_sweep,
    load_quant_sweep_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "data" / "quant" / "quantization_sweep_v1.json"


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #

def _model_configs() -> list[ModelConfig]:
    return [
        ModelConfig(name="model-fp16", family="test", revision="fp16", quantization="FP16",
                    source="ollama-local", context_length=131072, base_model="test-7b"),
        ModelConfig(name="model-q8", family="test", revision="q8", quantization="Q8_0",
                    source="ollama-local", context_length=131072, base_model="test-7b"),
        ModelConfig(name="model-q4", family="test", revision="q4", quantization="Q4_K_M",
                    source="ollama-local", context_length=131072, base_model="test-7b"),
        # Model without base_model — should be ignored by aggregate_quant_sweep
        ModelConfig(name="other-model", family="other", revision="fp16", quantization="FP16",
                    source="ollama-local", context_length=32768),
    ]


def _fixture() -> QuantSweepFixture:
    return QuantSweepFixture(
        name="test_sweep",
        version="0.1.0",
        recommended_suites=["hallucination_grounding", "openclaw_speed"],
        base_models=[
            BaseModelSpec(
                base_model="test-7b",
                display_name="Test 7B",
                variants=["model-fp16", "model-q8", "model-q4"],
                reference_variant="model-fp16",
                reference_pass_rates={
                    "hallucination_grounding": 0.90,
                    "practical_structured_output": 0.95,
                    "code_generation": 0.80,
                },
                enforce=False,
            )
        ],
    )


def _aggregate(
    hg_rates: dict[str, float] | None = None,
    pso_rates: dict[str, float] | None = None,
    cg_rates: dict[str, float] | None = None,
    spd_ttft: dict[str, float] | None = None,
    spd_tps: dict[str, float] | None = None,
) -> dict:
    hg_rates = hg_rates or {"model-fp16": 0.90, "model-q8": 0.88, "model-q4": 0.75}
    pso_rates = pso_rates or {"model-fp16": 0.95, "model-q8": 0.94, "model-q4": 0.85}
    cg_rates = cg_rates or {"model-fp16": 0.80, "model-q8": 0.79, "model-q4": 0.65}
    spd_ttft = spd_ttft or {"model-fp16": 200.0, "model-q8": 130.0, "model-q4": 95.0}
    spd_tps = spd_tps or {"model-fp16": 25.0, "model-q8": 38.0, "model-q4": 52.0}

    def _models(rates: dict, extra: dict | None = None) -> list:
        result = []
        for name, rate in rates.items():
            entry: dict = {"model": name, "pass_rate": rate, "passes": 0, "total": 100, "runs": 1}
            if extra and name in extra:
                entry.update(extra[name])
            result.append(entry)
        return result

    speed_models = []
    for name in spd_ttft:
        speed_models.append({
            "model": name,
            "pass_rate": 1.0,
            "passes": 10,
            "total": 10,
            "runs": 1,
            "avg_ttft_ms": spd_ttft[name],
            "avg_tokens_per_s": spd_tps.get(name),
        })

    return {
        "runs_root": "/tmp/results",
        "total_runs": 3,
        "suites": [
            {"suite": "hallucination_grounding", "models": _models(hg_rates)},
            {"suite": "practical_structured_output", "models": _models(pso_rates)},
            {"suite": "code_generation", "models": _models(cg_rates)},
            {"suite": "openclaw_speed", "models": speed_models},
        ],
        "runs": [],
    }


# --------------------------------------------------------------------- #
# Fixture loading                                                        #
# --------------------------------------------------------------------- #

def test_load_real_fixture():
    fixture = load_quant_sweep_fixture(FIXTURE_PATH)
    assert fixture.name == "quantization_sweep_v1"
    assert len(fixture.base_models) == 3
    base_names = {s.base_model for s in fixture.base_models}
    assert "qwen3-35b" in base_names
    assert "gemma4-27b" in base_names
    assert "nemotron3-33b" in base_names


def test_fixture_reference_variants():
    fixture = load_quant_sweep_fixture(FIXTURE_PATH)
    for spec in fixture.base_models:
        assert spec.reference_variant in spec.variants


def test_fixture_enforce_false_by_default():
    fixture = load_quant_sweep_fixture(FIXTURE_PATH)
    for spec in fixture.base_models:
        assert spec.enforce is False, "baselines not confirmed yet — enforce must stay False"


# --------------------------------------------------------------------- #
# aggregate_quant_sweep                                                  #
# --------------------------------------------------------------------- #

def test_aggregate_groups_by_base_model():
    result = aggregate_quant_sweep(_aggregate(), _model_configs(), _fixture())
    assert "test-7b" in result
    variants = result["test-7b"]["variants"]
    assert len(variants) == 3
    model_names = {v["model"] for v in variants}
    assert model_names == {"model-fp16", "model-q8", "model-q4"}


def test_aggregate_extracts_pass_rates():
    result = aggregate_quant_sweep(_aggregate(), _model_configs(), _fixture())
    variants = {v["model"]: v for v in result["test-7b"]["variants"]}
    assert variants["model-fp16"]["hallucination_pass_rate"] == pytest.approx(0.90)
    assert variants["model-q4"]["code_pass_rate"] == pytest.approx(0.65)
    assert variants["model-fp16"]["ttft_ms"] == pytest.approx(200.0)
    assert variants["model-q4"]["decode_tps"] == pytest.approx(52.0)


def test_aggregate_ignores_model_without_base_model():
    result = aggregate_quant_sweep(_aggregate(), _model_configs(), _fixture())
    for bm_result in result.values():
        names = {v["model"] for v in bm_result["variants"]}
        assert "other-model" not in names


def test_aggregate_suite_version_suffix_stripped():
    agg = _aggregate()
    # Rename suite to versioned name
    for suite in agg["suites"]:
        if suite["suite"] == "hallucination_grounding":
            suite["suite"] = "hallucination_grounding_v1"
    result = aggregate_quant_sweep(agg, _model_configs(), _fixture())
    variants = {v["model"]: v for v in result["test-7b"]["variants"]}
    assert variants["model-fp16"]["hallucination_pass_rate"] == pytest.approx(0.90)


def test_aggregate_missing_suite_yields_none():
    agg = _aggregate()
    agg["suites"] = [s for s in agg["suites"] if s["suite"] != "code_generation"]
    result = aggregate_quant_sweep(agg, _model_configs(), _fixture())
    variants = {v["model"]: v for v in result["test-7b"]["variants"]}
    assert variants["model-fp16"]["code_pass_rate"] is None


def test_aggregate_preserves_reference_metadata():
    result = aggregate_quant_sweep(_aggregate(), _model_configs(), _fixture())
    bm = result["test-7b"]
    assert bm["reference_variant"] == "model-fp16"
    assert bm["display_name"] == "Test 7B"
    assert bm["enforce"] is False


# --------------------------------------------------------------------- #
# check_quant_regressions                                                #
# --------------------------------------------------------------------- #

def test_regressions_silent_when_enforce_false():
    result = aggregate_quant_sweep(_aggregate(), _model_configs(), _fixture())
    warnings = check_quant_regressions(result, _fixture())
    assert warnings == []


def test_regressions_fires_when_enforce_true():
    fixture = QuantSweepFixture(
        name="test_sweep",
        version="0.1.0",
        base_models=[
            BaseModelSpec(
                base_model="test-7b",
                display_name="Test 7B",
                variants=["model-fp16", "model-q8", "model-q4"],
                reference_variant="model-fp16",
                reference_pass_rates={"hallucination_grounding": 0.90},
                enforce=True,
            )
        ],
    )
    # model-q4 hallucination is 0.75, reference threshold 0.90 → delta -0.15 → >5pp → warning
    result = aggregate_quant_sweep(_aggregate(), _model_configs(), fixture)
    warnings = check_quant_regressions(result, fixture)
    assert any("model-q4" in w and "hallucination_grounding" in w for w in warnings)


def test_regressions_silent_when_within_5pp():
    fixture = QuantSweepFixture(
        name="test_sweep",
        version="0.1.0",
        base_models=[
            BaseModelSpec(
                base_model="test-7b",
                display_name="Test 7B",
                variants=["model-fp16", "model-q8", "model-q4"],
                reference_variant="model-fp16",
                reference_pass_rates={"hallucination_grounding": 0.90},
                enforce=True,
            )
        ],
    )
    # model-q8 hallucination is 0.88, threshold 0.90 → delta -0.02 → within 5pp → no warning
    result = aggregate_quant_sweep(_aggregate(), _model_configs(), fixture)
    warnings = check_quant_regressions(result, fixture)
    assert not any("model-q8" in w and "hallucination_grounding" in w for w in warnings)


def test_regressions_skips_null_reference():
    fixture = QuantSweepFixture(
        name="test_sweep",
        version="0.1.0",
        base_models=[
            BaseModelSpec(
                base_model="test-7b",
                display_name="Test 7B",
                variants=["model-fp16", "model-q4"],
                reference_variant="model-fp16",
                reference_pass_rates={"hallucination_grounding": None},
                enforce=True,
            )
        ],
    )
    result = aggregate_quant_sweep(_aggregate(), _model_configs(), fixture)
    warnings = check_quant_regressions(result, fixture)
    assert warnings == []


# --------------------------------------------------------------------- #
# HTML rendering (smoke tests)                                           #
# --------------------------------------------------------------------- #

def test_html_card_renders():
    from spark_benchmark.reporting_html import _render_quant_sweep_card
    result_data = {
        "base_model": "test-7b",
        "display_name": "Test 7B",
        "reference_variant": "model-fp16",
        "reference_pass_rates": {},
        "enforce": False,
        "variants": [
            {"model": "model-fp16", "quantization": "FP16",
             "hallucination_pass_rate": 0.90, "structured_output_pass_rate": 0.95,
             "code_pass_rate": 0.80, "ttft_ms": 200.0, "decode_tps": 25.0, "peak_vram_mb": 70000},
            {"model": "model-q4", "quantization": "Q4_K_M",
             "hallucination_pass_rate": 0.75, "structured_output_pass_rate": 0.85,
             "code_pass_rate": 0.65, "ttft_ms": 95.0, "decode_tps": 52.0, "peak_vram_mb": 20000},
        ],
    }
    html = _render_quant_sweep_card(result_data)
    assert "Test 7B" in html
    assert "model-fp16" in html
    assert "model-q4" in html
    assert "Q4_K_M" in html
    assert 'data-band="good"' in html   # reference row quality cells
    assert 'data-band="bad"' in html    # q4 hallucination dropped >5pp
    assert "<table" in html
    assert "ref</span>" in html


def test_html_section_empty_on_no_sweep():
    from spark_benchmark.reporting_html import _render_quant_sweep_section
    assert _render_quant_sweep_section({}) == ""


def test_html_report_includes_sweep_from_aggregate():
    from spark_benchmark.reporting_html import render_canonical_report_html
    agg = _aggregate()
    sweep = aggregate_quant_sweep(agg, _model_configs(), _fixture())
    agg["quant_sweep"] = sweep
    html = render_canonical_report_html(agg)
    assert "Quantization tradeoff" in html
    assert "Test 7B" in html


def test_html_report_without_sweep_unchanged():
    from spark_benchmark.reporting_html import render_canonical_report_html
    agg = _aggregate()
    html = render_canonical_report_html(agg)
    assert "Quantization tradeoff" not in html


# --------------------------------------------------------------------- #
# enrich_with_quant_sweep                                                #
# --------------------------------------------------------------------- #

def test_enrich_injects_quant_sweep(tmp_path):
    fixture_path = tmp_path / "quantization_sweep_v1.json"
    fixture_path.write_text(_fixture().model_dump_json(), encoding="utf-8")
    agg = _aggregate()
    result = enrich_with_quant_sweep(agg, _model_configs(), fixture_path=fixture_path)
    assert "quant_sweep" in result
    assert "test-7b" in result["quant_sweep"]


def test_enrich_skips_when_no_base_model(tmp_path):
    fixture_path = tmp_path / "quantization_sweep_v1.json"
    fixture_path.write_text(_fixture().model_dump_json(), encoding="utf-8")
    configs_no_base = [
        ModelConfig(name="other", family="x", revision="x:7b", quantization="Q4_K_M",
                    source="ollama-local", context_length=4096),
    ]
    agg = _aggregate()
    result = enrich_with_quant_sweep(agg, configs_no_base, fixture_path=fixture_path)
    assert "quant_sweep" not in result


def test_enrich_skips_when_fixture_missing(tmp_path):
    agg = _aggregate()
    result = enrich_with_quant_sweep(agg, _model_configs(), fixture_path=tmp_path / "nonexistent.json")
    assert "quant_sweep" not in result


def test_enrich_end_to_end_html(tmp_path):
    from spark_benchmark.reporting_html import render_canonical_report_html
    fixture_path = tmp_path / "quantization_sweep_v1.json"
    fixture_path.write_text(_fixture().model_dump_json(), encoding="utf-8")
    agg = _aggregate()
    enrich_with_quant_sweep(agg, _model_configs(), fixture_path=fixture_path)
    html = render_canonical_report_html(agg)
    assert "Quantization tradeoff" in html
    assert "Test 7B" in html
