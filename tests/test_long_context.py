from pathlib import Path

from pydantic import ValidationError

from spark_benchmark.long_context import (
    HaystackSpec,
    LongContextFixture,
    Needle,
    TestMatrix,
    _stable_hash,
    load_long_context_fixture,
    score_niah,
    select_haystack,
    select_needle_index,
)
from spark_benchmark.models import ModelConfig
from spark_benchmark.reliability import fixture_path_for_suite_name


REPO_ROOT = Path(__file__).resolve().parents[1]


def _expect_raises(exc_type: type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} but no exception was raised")


def _minimal_fixture_kwargs() -> dict:
    return {
        "name": "lc_test",
        "haystacks": {
            "h1": HaystackSpec(source_url="u", license="PD", text_file="a.txt"),
        },
        "needles": [
            Needle(
                id="n1",
                category="date",
                text="X happened on May 1, 2020.",
                question="When?",
                expected_substring="May 1, 2020",
            )
        ],
        "test_matrix": TestMatrix(
            context_lengths_tokens=[4096],
            depth_percentages=[0, 50, 100],
            needles_per_cell=1,
            haystacks=["h1"],
        ),
    }


# --------------------------------------------------------------------- #
# Shipped fixture                                                        #
# --------------------------------------------------------------------- #


def test_shipped_fixture_loads_via_registry() -> None:
    path = fixture_path_for_suite_name(REPO_ROOT, "long_context_retrieval")
    fixture = load_long_context_fixture(path)
    assert fixture.name == "long_context_retrieval_v1"
    assert fixture.category == "reliability"


def test_shipped_fixture_has_enough_needles_and_categories() -> None:
    path = fixture_path_for_suite_name(REPO_ROOT, "long_context_retrieval_v1")
    fixture = load_long_context_fixture(path)
    assert len(fixture.needles) >= fixture.test_matrix.needles_per_cell
    assert len(fixture.needles) >= 8
    categories = {n.category for n in fixture.needles}
    assert len(categories) >= 3


def test_shipped_fixture_haystacks_are_public_domain() -> None:
    path = fixture_path_for_suite_name(REPO_ROOT, "long_context_retrieval")
    fixture = load_long_context_fixture(path)
    assert len(fixture.haystacks) >= 2
    for spec in fixture.haystacks.values():
        assert spec.source_url
        assert spec.license


# --------------------------------------------------------------------- #
# Needle validation                                                      #
# --------------------------------------------------------------------- #


def test_needle_rejects_expected_not_in_text() -> None:
    _expect_raises(
        ValidationError,
        lambda: Needle(
            id="bad",
            category="date",
            text="nothing relevant here",
            question="q",
            expected_substring="ABSENT",
        ),
    )


def test_needle_rejects_empty_expected() -> None:
    _expect_raises(
        ValidationError,
        lambda: Needle(id="bad", category="c", text="abc", question="q", expected_substring="   "),
    )


# --------------------------------------------------------------------- #
# TestMatrix + fixture validation                                        #
# --------------------------------------------------------------------- #


def test_matrix_rejects_empty_lengths() -> None:
    _expect_raises(
        ValidationError,
        lambda: TestMatrix(
            context_lengths_tokens=[], depth_percentages=[0], needles_per_cell=1, haystacks=["h1"]
        ),
    )


def test_matrix_rejects_out_of_range_depth() -> None:
    _expect_raises(
        ValidationError,
        lambda: TestMatrix(
            context_lengths_tokens=[4096],
            depth_percentages=[0, 150],
            needles_per_cell=1,
            haystacks=["h1"],
        ),
    )


def test_matrix_rejects_zero_needles_per_cell() -> None:
    _expect_raises(
        ValidationError,
        lambda: TestMatrix(
            context_lengths_tokens=[4096], depth_percentages=[0], needles_per_cell=0, haystacks=["h1"]
        ),
    )


def test_fixture_rejects_undefined_haystack_reference() -> None:
    kwargs = _minimal_fixture_kwargs()
    kwargs["test_matrix"] = TestMatrix(
        context_lengths_tokens=[4096],
        depth_percentages=[0],
        needles_per_cell=1,
        haystacks=["does_not_exist"],
    )
    _expect_raises(ValidationError, lambda: LongContextFixture(**kwargs))


def test_fixture_rejects_needles_per_cell_exceeding_pool() -> None:
    kwargs = _minimal_fixture_kwargs()
    kwargs["test_matrix"] = TestMatrix(
        context_lengths_tokens=[4096],
        depth_percentages=[0],
        needles_per_cell=5,  # only 1 needle defined
        haystacks=["h1"],
    )
    _expect_raises(ValidationError, lambda: LongContextFixture(**kwargs))


def test_minimal_fixture_is_valid() -> None:
    fixture = LongContextFixture(**_minimal_fixture_kwargs())
    assert fixture.test_matrix.needles_per_cell == 1


# --------------------------------------------------------------------- #
# Scoring                                                                #
# --------------------------------------------------------------------- #


def test_score_niah_exact_match() -> None:
    passed, details = score_niah("The code is 7B-MIRA-4419.", "7B-MIRA-4419")
    assert passed is True
    assert details["matched"] is True


def test_score_niah_case_insensitive() -> None:
    passed, _ = score_niah("the answer is ostrava, definitely", "Ostrava")
    assert passed is True


def test_score_niah_whitespace_normalised() -> None:
    passed, _ = score_niah("launched on November   14,\n2023", "November 14, 2023")
    assert passed is True


def test_score_niah_no_match() -> None:
    passed, details = score_niah("I don't know the code", "7B-MIRA-4419")
    assert passed is False
    assert details["matched"] is False


# --------------------------------------------------------------------- #
# Deterministic selection                                                #
# --------------------------------------------------------------------- #


def test_stable_hash_is_deterministic() -> None:
    assert _stable_hash(65536, 33, 2) == _stable_hash(65536, 33, 2)


def test_select_needle_index_in_range_and_deterministic() -> None:
    a = select_needle_index(65536, 33, 2, n_needles=8)
    b = select_needle_index(65536, 33, 2, n_needles=8)
    assert a == b
    assert 0 <= a < 8


def test_select_needle_index_varies_across_cells() -> None:
    indices = {
        select_needle_index(length, depth, rep, n_needles=8)
        for length in (4096, 16384, 65536, 131072)
        for depth in (0, 33, 66, 100)
        for rep in range(8)
    }
    # Should hit more than one needle across the whole grid (not degenerate).
    assert len(indices) > 1


def test_select_haystack_deterministic_and_valid() -> None:
    pool = ["literary_melville", "scientific_darwin"]
    a = select_haystack(65536, 33, pool)
    b = select_haystack(65536, 33, pool)
    assert a == b
    assert a in pool


# --------------------------------------------------------------------- #
# ModelConfig.base_model                                                 #
# --------------------------------------------------------------------- #


def test_model_config_base_model_defaults_none() -> None:
    cfg = ModelConfig(
        name="qwen-3.6",
        family="qwen",
        revision="qwen3.6:35b",
        quantization="ollama-default",
        source="ollama-local",
        context_length=131072,
    )
    assert cfg.base_model is None


def test_model_config_base_model_set() -> None:
    cfg = ModelConfig(
        name="llama-3.3-70b-q4km",
        family="llama",
        revision="llama3.3:70b-q4km",
        quantization="Q4_K_M",
        source="ollama-local",
        context_length=131072,
        base_model="llama-3.3-70b",
    )
    assert cfg.base_model == "llama-3.3-70b"


def test_existing_model_yaml_still_loads_without_base_model() -> None:
    import yaml

    raw = (REPO_ROOT / "configs" / "models" / "qwen-3.6.yaml").read_text(encoding="utf-8")
    cfg = ModelConfig.model_validate(yaml.safe_load(raw))
    assert cfg.base_model is None
