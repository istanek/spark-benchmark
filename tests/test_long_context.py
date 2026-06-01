import tempfile
from pathlib import Path

from pydantic import ValidationError

from spark_benchmark.long_context import (
    HaystackSpec,
    LongContextFixture,
    Needle,
    TestMatrix,
    _stable_hash,
    build_cell_prompt,
    build_long_context_summary,
    cell_nonce,
    estimate_chars_for_tokens,
    insert_needle,
    load_haystack_texts,
    load_long_context_fixture,
    run_long_context_suite,
    score_niah,
    select_haystack,
    select_needle_index,
    slice_haystack,
)
from spark_benchmark.models import (
    BackendConfig,
    BackendKind,
    GenerationResult,
    InferenceMetrics,
    ModelConfig,
    SamplingConfig,
)
from spark_benchmark.reliability import fixture_path_for_suite_name


REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- #
# Test doubles                                                           #
# --------------------------------------------------------------------- #


class FakeBackend:
    """In-memory backend: a 'perfect' model unless told to be blind/raise."""

    def __init__(self, needles, *, answers=True, raise_on_ctx=None, mem=None, prefill_tokens=1000):
        self.needles = needles
        self.answers = answers  # bool or callable(num_ctx) -> bool
        self.raise_on_ctx = raise_on_ctx
        self.mem = mem
        self.prefill_tokens = prefill_tokens
        self.loaded = None
        self.timeout_s = 300.0
        self.prompts: list[str] = []
        self.num_ctx_seen: list[int | None] = []

    def load_model(self, mc):
        self.loaded = mc

    def unload(self):
        self.loaded = None

    def memory_snapshot(self):
        return self.mem

    def generate(self, prompt, params):
        self.prompts.append(prompt)
        self.num_ctx_seen.append(params.num_ctx)
        if self.raise_on_ctx is not None and params.num_ctx == self.raise_on_ctx:
            raise RuntimeError("simulated OOM")
        answers = self.answers(params.num_ctx) if callable(self.answers) else self.answers
        out = "I cannot find that in the document."
        if answers:
            for n in self.needles:
                if n.text in prompt:
                    out = f"Looking at the document, the answer is {n.expected_substring}."
                    break
        metrics = InferenceMetrics(prefill_tokens=self.prefill_tokens, prefill_time_s=0.5)
        return GenerationResult(prompt=prompt, output=out, metrics=metrics)


def _backend_config() -> BackendConfig:
    return BackendConfig(name=BackendKind.OLLAMA, entrypoint="test", version="test")


def _model(name="m1", ctx=131072) -> ModelConfig:
    return ModelConfig(
        name=name,
        family="test",
        revision="test:1",
        quantization="Q4",
        source="test",
        context_length=ctx,
    )


def _tiny_fixture(needles_per_cell=2) -> LongContextFixture:
    return LongContextFixture(
        name="lc_e2e",
        haystacks={"h": HaystackSpec(source_url="u", license="PD", text_file="h.txt")},
        needles=[
            Needle(id="n1", category="date", text="X launched on May 1, 2020.", question="When did X launch?", expected_substring="May 1, 2020"),
            Needle(id="n2", category="code", text="The code is AB-12-CD.", question="What is the code?", expected_substring="AB-12-CD"),
            Needle(id="n3", category="loc", text="The plant is in Brno.", question="Where is the plant?", expected_substring="Brno"),
        ],
        test_matrix=TestMatrix(
            context_lengths_tokens=[4096, 16384],
            depth_percentages=[0, 100],
            needles_per_cell=needles_per_cell,
            haystacks=["h"],
        ),
    )


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


# --------------------------------------------------------------------- #
# Prompt assembly                                                        #
# --------------------------------------------------------------------- #


def test_estimate_chars_for_tokens() -> None:
    assert estimate_chars_for_tokens(1000, 7.0) == 7000
    assert estimate_chars_for_tokens(0) >= 1


def test_slice_haystack_truncates_long_text() -> None:
    text = "abcdefghij" * 100  # 1000 chars
    assert slice_haystack(text, 250) == text[:250]


def test_slice_haystack_tiles_short_text() -> None:
    out = slice_haystack("abc", 10)
    assert len(out) == 10
    assert out == "abcabcabca"


def test_slice_haystack_empty_raises() -> None:
    _expect_raises(ValueError, lambda: slice_haystack("", 10))


def test_insert_needle_at_start() -> None:
    out = insert_needle("filler text here", "NEEDLE", 0)
    assert out.startswith("NEEDLE")
    assert "filler text here" in out


def test_insert_needle_at_end() -> None:
    out = insert_needle("filler text here", "NEEDLE", 100)
    assert out.rstrip().endswith("NEEDLE")


def test_insert_needle_in_middle_keeps_words_intact() -> None:
    haystack = "alpha beta gamma delta epsilon zeta eta theta"
    out = insert_needle(haystack, "NEEDLE", 50)
    assert "NEEDLE" in out
    # No word should be glued to the needle without a space.
    assert " NEEDLE " in out
    # All original words survive.
    for word in haystack.split():
        assert word in out


def test_cell_nonce_unique_per_rep_stable_across_runs() -> None:
    a0 = cell_nonce(4096, 0, 0)
    a1 = cell_nonce(4096, 0, 1)
    assert a0 != a1  # different repetitions -> different nonce (cache-busting)
    assert cell_nonce(4096, 0, 0) == a0  # stable across calls (reproducible)


def test_build_cell_prompt_contains_question_and_nonce() -> None:
    prompt = build_cell_prompt("DOC BODY", "What is X?", "abc-1")
    assert "abc-1" in prompt
    assert "What is X?" in prompt
    assert "DOC BODY" in prompt
    assert "DOCUMENT START" in prompt


# --------------------------------------------------------------------- #
# Haystack loading                                                       #
# --------------------------------------------------------------------- #


def test_load_haystack_texts_missing_file_raises_with_hint() -> None:
    fixture = _tiny_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_haystack_texts(fixture, tmp)
        except FileNotFoundError as exc:
            assert "fetch_haystacks.sh" in str(exc)
            return
    raise AssertionError("expected FileNotFoundError for missing haystack")


# --------------------------------------------------------------------- #
# Runner (end-to-end with a fake backend)                                #
# --------------------------------------------------------------------- #


def _run(fixture, backend, models, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        summary = run_long_context_suite(
            run_dir=run_dir,
            fixture=fixture,
            haystack_texts={"h": "lorem ipsum dolor sit amet " * 5000},
            backend=backend,
            backend_config=_backend_config(),
            model_configs=models,
            sampling=SamplingConfig(max_tokens=256),
            **kw,
        )
        results_text = (run_dir / "results.jsonl").read_text(encoding="utf-8")
        summary_md = (run_dir / "summary.md").read_text(encoding="utf-8")
        return summary, results_text, summary_md


def test_runner_perfect_model_passes_everything() -> None:
    fixture = _tiny_fixture(needles_per_cell=2)
    backend = FakeBackend(fixture.needles, answers=True)
    summary, results_text, _ = _run(fixture, backend, [_model()])
    # grid: 2 lengths * 2 depths * 2 reps = 8 cells
    assert summary["total_rows"] == 8
    model = summary["models"][0]
    assert model["pass_rate"] == 1.0
    assert model["skipped"] == 0
    assert model["errors"] == 0
    assert model["first_failure_length"] is None
    assert results_text.count("\n") == 8


def test_runner_sets_num_ctx_per_length() -> None:
    fixture = _tiny_fixture(needles_per_cell=1)
    backend = FakeBackend(fixture.needles, answers=True)
    _run(fixture, backend, [_model()])
    # Every generate call must carry an explicit num_ctx equal to a grid length.
    assert all(ctx in (4096, 16384) for ctx in backend.num_ctx_seen)
    assert 4096 in backend.num_ctx_seen
    assert 16384 in backend.num_ctx_seen


def test_runner_skips_unsupported_lengths() -> None:
    fixture = _tiny_fixture(needles_per_cell=1)
    backend = FakeBackend(fixture.needles, answers=True)
    # context_length 5000 -> 16384 cells are skipped_unsupported, 4096 run
    summary, _, _ = _run(fixture, backend, [_model(ctx=5000)])
    model = summary["models"][0]
    # 2 depths * 1 rep at 16384 = 2 skipped
    assert model["skipped"] == 2
    # backend only called for the 2 supported (4096) cells
    assert len(backend.prompts) == 2


def test_runner_records_error_on_backend_raise() -> None:
    fixture = _tiny_fixture(needles_per_cell=1)
    backend = FakeBackend(fixture.needles, answers=True, raise_on_ctx=16384)
    summary, results_text, _ = _run(fixture, backend, [_model()])
    model = summary["models"][0]
    assert model["errors"] == 2  # 2 depths * 1 rep at the failing length
    assert "simulated OOM" in results_text


def test_runner_blind_model_fails_and_reports_first_failure() -> None:
    fixture = _tiny_fixture(needles_per_cell=2)
    # model answers only up to 4096; goes blind at 16384
    backend = FakeBackend(fixture.needles, answers=lambda ctx: ctx <= 4096)
    summary, _, _ = _run(fixture, backend, [_model()])
    model = summary["models"][0]
    assert model["first_failure_length"] == 16384
    # 4096 cells pass, 16384 cells fail
    by_len = {(c["context_length"]): c["pass_rate"] for c in model["cells"]}
    assert by_len[4096] == 1.0
    assert by_len[16384] == 0.0


def test_runner_captures_memory_snapshot() -> None:
    fixture = _tiny_fixture(needles_per_cell=1)
    backend = FakeBackend(fixture.needles, answers=True, mem={"size_vram_mb": 4096.0})
    summary, results_text, _ = _run(fixture, backend, [_model()])
    assert "size_vram_mb" in results_text
    # peak vram surfaces into the per-cell summary
    cell = summary["models"][0]["cells"][0]
    assert cell["peak_vram_mb"] == 4096.0


def test_build_long_context_summary_counts_states() -> None:
    rows = [
        {"model": "m", "context_length": 4096, "depth_pct": 0, "status": "pass", "passed": True, "prefill_tokens_per_sec": 100},
        {"model": "m", "context_length": 4096, "depth_pct": 0, "status": "fail", "passed": False, "prefill_tokens_per_sec": 100},
        {"model": "m", "context_length": 16384, "depth_pct": 0, "status": "skipped_unsupported", "passed": False},
        {"model": "m", "context_length": 16384, "depth_pct": 100, "status": "error", "passed": False},
    ]
    fixture = _tiny_fixture(needles_per_cell=1)
    summary = build_long_context_summary(rows, fixture, _backend_config())
    model = summary["models"][0]
    assert model["total"] == 4
    assert model["skipped"] == 1
    assert model["errors"] == 1
    assert model["passes"] == 1
    # scored = total - skipped = 3; passes = 1
    assert model["pass_rate"] == round(1 / 3, 4)
