"""Hybrid score identity, filters, fallbacks, index integrity, and real demo ranking."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import indexing
from indexing import (
    acoustic_features,
    acoustic_query,
    build_index,
    load_demo_index,
    normalize,
    save_demo_index,
)
from retrieval import search
from schemas import SAMPLE_RATE, AudioSleuthError
from service import Collection


@pytest.mark.parametrize(
    "query,phrase,speaker",
    [
        ("heated argument about interest rates", "disagree", "SPEAKER_01"),
        ("speaker laughing about valuation", "laughing", "SPEAKER_01"),
        ("renewable energy and battery storage", "battery storage", "SPEAKER_01"),
    ],
)
def test_demo_queries_find_expected_moments(
    demo_collection: Collection, query: str, phrase: str, speaker: str
) -> None:
    hit = search(demo_collection.index, query, top_k=1)[0]
    assert phrase in hit.chunk.transcript.lower()
    assert hit.chunk.speaker_id == speaker


@pytest.mark.parametrize("alpha", [0.0, 0.35, 1.0])
def test_exact_hybrid_formula_and_sorted_results(demo_collection: Collection, alpha: float) -> None:
    hits = search(demo_collection.index, "heated interest rates", alpha=alpha)
    assert [hit.score for hit in hits] == sorted((h.score for h in hits), reverse=True)
    for hit in hits:
        assert hit.score == pytest.approx(alpha * hit.text_score + (1 - alpha) * hit.audio_score)


def test_filters_apply_before_topk(demo_collection: Collection) -> None:
    hits = search(demo_collection.index, "valuation", speaker="SPEAKER_00")
    assert hits and all(hit.chunk.speaker_id == "SPEAKER_00" for hit in hits)
    assert search(demo_collection.index, "valuation", speaker="absent") == []
    assert search(demo_collection.index, "valuation", min_score=1) == []
    copied = replace(
        demo_collection.index,
        chunks=[replace(chunk, confidence=0.2) for chunk in demo_collection.index.chunks],
    )
    assert search(copied, "valuation", min_confidence=0.5) == []


@pytest.mark.parametrize(
    "query,options",
    [
        ("", {}),
        ("x" * 1001, {}),
        ("hello", {"alpha": float("nan")}),
        ("hello", {"alpha": 2}),
        ("hello", {"top_k": 0}),
        ("hello", {"min_confidence": -1}),
    ],
)
def test_invalid_queries_rejected(
    demo_collection: Collection, query: str, options: dict[str, Any]
) -> None:
    with pytest.raises(AudioSleuthError):
        search(demo_collection.index, query, **options)


def test_signal_features_and_no_tone_query() -> None:
    silence = acoustic_features(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert silence[-1] == 1
    assert np.isfinite(silence).all()
    assert not acoustic_query("interest rates").any()
    assert acoustic_query("heated argument").any()
    np.testing.assert_array_equal(normalize(np.zeros((1, 4), dtype=np.float32)), 0)


def test_precomputed_roundtrip_and_integrity(demo_collection: Collection, tmp_path: Path) -> None:
    index = demo_collection.index
    save_demo_index(index, tmp_path)
    loaded = load_demo_index(index.chunks, tmp_path)
    original_hits = search(index, "laughing valuation")
    loaded_hits = search(loaded, "laughing valuation")
    assert [h.chunk.chunk_id for h in loaded_hits] == [h.chunk.chunk_id for h in original_hits]
    np.testing.assert_allclose([h.score for h in loaded_hits], [h.score for h in original_hits])
    manifest = json.loads((tmp_path / "demo_index.json").read_text())
    manifest["chunks"][0]["start_ms"] += 1
    (tmp_path / "demo_index.json").write_text(json.dumps(manifest))
    with pytest.raises(AudioSleuthError, match="provenance"):
        load_demo_index(index.chunks, tmp_path)


def test_missing_neural_backends_fall_back_visibly(
    demo_collection: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> None:
        raise ImportError("Optional dependency deliberately absent.")

    monkeypatch.setattr(indexing, "_sentence_model", unavailable)
    monkeypatch.setattr(indexing, "_clap_model", unavailable)
    index = build_index(demo_collection.index.chunks, neural=True)
    assert index.text_backend.startswith("TF-IDF")
    assert index.audio_backend.startswith("Signal")
    assert len(index.notices) == 2
    assert search(index, "heated interest rates")


def test_clap_windows_cover_tail_and_query_uses_same_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lengths: list[int] = []

    class FakeCLAP:
        def audio(self, samples: np.ndarray) -> np.ndarray:
            lengths.append(len(samples))
            return np.array([[1, 0]], dtype=np.float32)

    monkeypatch.setattr(indexing, "_clap_model", lambda: FakeCLAP())
    vector = indexing._clap_audio(np.ones(21 * SAMPLE_RATE, dtype=np.float32))
    assert lengths == [480_000, 480_000, 48_000]
    np.testing.assert_array_equal(vector, [1, 0])
