"""Long-context retrieval suite (single-needle NIAH).

This module is the v0.4.0 implementation of `long_context_retrieval`,
following the design in `docs/long-context-spec.md`. It is platform-
agnostic; the v1 configuration targets Spark only.

Layering note: this first slice ships the **fixture schema, loader, and
the pure deterministic plumbing** (needle/haystack selection, substring
scoring). The actual run loop — per-model tokenization, haystack
truncation, needle insertion, backend calls, three-state cell logic —
lands in the follow-up slice. Everything here is side-effect-free and
unit-testable without a backend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Needle(BaseModel):
    """A single fact hidden in the haystack, plus how to query/score it."""

    id: str
    category: str
    text: str
    question: str
    expected_substring: str

    @model_validator(mode="after")
    def _validate(self) -> "Needle":
        if not self.id.strip():
            raise ValueError("needle id must not be empty")
        if not self.expected_substring.strip():
            raise ValueError(f"needle {self.id!r} has empty expected_substring")
        if self.expected_substring not in self.text:
            raise ValueError(
                f"needle {self.id!r}: expected_substring is not contained in its own text "
                "(the scorer would never be able to pass)"
            )
        return self


class HaystackSpec(BaseModel):
    """Provenance + on-disk location for one filler corpus.

    The bytes themselves are fetched on demand (see
    ``scripts/fetch_haystacks.sh``) and are git-ignored; only this
    metadata ships in the fixture so the repo stays lean.
    """

    source_url: str
    license: str
    text_file: str
    sha256: str | None = None


class TestMatrix(BaseModel):
    context_lengths_tokens: list[int]
    depth_percentages: list[int]
    needles_per_cell: int
    haystacks: list[str]

    @model_validator(mode="after")
    def _validate(self) -> "TestMatrix":
        for field_name in ("context_lengths_tokens", "depth_percentages", "haystacks"):
            if not getattr(self, field_name):
                raise ValueError(f"test_matrix.{field_name} must not be empty")
        if self.needles_per_cell < 1:
            raise ValueError("test_matrix.needles_per_cell must be >= 1")
        for depth in self.depth_percentages:
            if not 0 <= depth <= 100:
                raise ValueError(f"depth_percentages must be in [0, 100], got {depth}")
        return self


class LongContextFixture(BaseModel):
    name: str
    category: str = "reliability"
    version: str = "0.4.0"
    description: str = ""
    notes: list[str] = Field(default_factory=list)
    haystacks: dict[str, HaystackSpec]
    needles: list[Needle]
    test_matrix: TestMatrix

    @model_validator(mode="after")
    def _validate(self) -> "LongContextFixture":
        if not self.haystacks:
            raise ValueError("fixture must define at least one haystack")
        if len(self.needles) < self.test_matrix.needles_per_cell:
            raise ValueError(
                f"fixture has {len(self.needles)} needles but needles_per_cell="
                f"{self.test_matrix.needles_per_cell}; a cell could not be filled without repeats"
            )
        missing = [h for h in self.test_matrix.haystacks if h not in self.haystacks]
        if missing:
            raise ValueError(f"test_matrix references undefined haystacks: {missing}")
        return self


def load_long_context_fixture(path: Path | str) -> LongContextFixture:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LongContextFixture.model_validate(payload)


def _stable_hash(*parts: Any) -> int:
    """Process-stable hash (Python's builtin ``hash`` is salted per run).

    Used to make needle/haystack selection reproducible across runs and
    machines: same inputs always yield the same task plan.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def select_needle_index(length: int, depth: int, repetition: int, n_needles: int) -> int:
    """Deterministically pick which needle a (length, depth, rep) cell uses."""
    if n_needles < 1:
        raise ValueError("n_needles must be >= 1")
    return _stable_hash(length, depth, repetition) % n_needles


def select_haystack(length: int, depth: int, haystacks: list[str]) -> str:
    """Deterministically rotate which haystack a (length, depth) cell uses."""
    if not haystacks:
        raise ValueError("haystacks must not be empty")
    return haystacks[_stable_hash(length, depth) % len(haystacks)]


def score_niah(response: str, expected: str) -> tuple[bool, dict[str, Any]]:
    """Case-insensitive substring match with whitespace normalisation.

    Deterministic, no LLM judge (see docs/long-context-spec.md). Returns
    ``(passed, details)``.
    """

    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    passed = norm(expected) in norm(response)
    return passed, {
        "matched": passed,
        "response_length": len(response),
        "expected": expected,
    }
