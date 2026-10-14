"""Regression cases discovered by rechecking upload and query edge conditions."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import indexing
from ingestion import _aligned_words
from retrieval import search
from schemas import AudioSleuthError
from service import Collection


def test_absent_word_score_keeps_aligned_times_with_zero_quality() -> None:
    words = _aligned_words(
        [
            {
                "start": 0,
                "end": 1,
                "text": "hello",
                "words": [{"word": "hello", "start": 0, "end": 1, "score": None}],
            }
        ],
        1,
    )
    assert len(words) == 1
    assert words[0].confidence == 0 and words[0].timing == "aligned"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_aligner_time_is_not_silently_clamped(bad: float) -> None:
    with pytest.raises(AudioSleuthError, match="finite"):
        _aligned_words(
            [
                {
                    "start": 0,
                    "end": 1,
                    "text": "hello",
                    "words": [{"word": "hello", "start": bad, "end": 1, "score": 1}],
                }
            ],
            1,
        )


def test_partial_unaligned_run_uses_next_complete_anchor() -> None:
    words = _aligned_words(
        [
            {
                "start": 0,
                "end": 4,
                "text": "one two three four",
                "words": [
                    {"word": "one", "start": 0, "end": 1},
                    {"word": "two"},
                    {"word": "three", "start": 1.5},
                    {"word": "four", "start": 3, "end": 4},
                ],
            }
        ],
        4,
    )
    assert [w.text for w in words] == ["one", "two", "three", "four"]
    assert words[1].end <= words[2].start
    assert words[2].end <= words[3].start
    assert all(w.timing == "estimated" for w in words[1:3])


def test_text_only_search_does_not_need_clap(
    demo_collection: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> None:
        raise ImportError("The zero-weight encoder should not be loaded.")

    monkeypatch.setattr(indexing, "_clap_model", unavailable)
    index = replace(demo_collection.index, audio_backend="CLAP")
    hits = search(index, "valuation", alpha=1)
    assert hits and all(h.audio_score == 0 for h in hits)


def test_audio_only_search_does_not_need_sentence_model(
    demo_collection: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> None:
        raise ImportError("The zero-weight encoder should not be loaded.")

    monkeypatch.setattr(indexing, "_sentence_model", unavailable)
    index = replace(demo_collection.index, text_backend="sentence-transformers")
    hits = search(index, "heated", alpha=0)
    assert hits and all(h.text_score == 0 for h in hits)


def test_unsupported_query_has_no_arbitrary_zero_evidence_matches(
    demo_collection: Collection,
) -> None:
    assert search(demo_collection.index, "zzqvnomatch", alpha=1) == []
    assert search(demo_collection.index, "interest rates", alpha=0) == []


def test_corrupt_idf_is_rejected_at_index_load(demo_collection: Collection, tmp_path: Path) -> None:
    indexing.save_demo_index(demo_collection.index, tmp_path)
    index = demo_collection.index
    np.savez_compressed(
        tmp_path / "demo_index.npz",
        text=index.text_vectors,
        audio=index.audio_vectors,
        idf=np.full(index.text_vectors.shape[1], np.nan),
    )
    with pytest.raises(AudioSleuthError, match="IDF"):
        indexing.load_demo_index(index.chunks, tmp_path)


def test_sentence_embeddings_cover_every_token_and_weight_the_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[int]] = []

    class Tensor:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def to(self, device: str) -> Tensor:
            return self

        def detach(self) -> Tensor:
            return self

        def cpu(self) -> Tensor:
            return self

        def numpy(self) -> np.ndarray:
            return self.value

    class Tokenizer:
        def num_special_tokens_to_add(self, pair: bool) -> int:
            return 2

        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [int(token) for token in text.split()]

        def prepare_for_model(self, tokens: list[int], **kwargs: Any) -> dict[str, list[int]]:
            seen.append(tokens)
            return {"input_ids": tokens}

        def pad(self, windows: list[dict[str, list[int]]], **kwargs: Any) -> dict[str, Tensor]:
            rows = [[int(9 not in w["input_ids"]), int(9 in w["input_ids"])] for w in windows]
            return {"input_ids": Tensor(np.array(rows, dtype=np.float32))}

    class Model:
        max_seq_length = 6
        device = "cpu"
        tokenizer = Tokenizer()

        def eval(self) -> None:
            return None

        def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
            return {"sentence_embedding": features["input_ids"]}

    original = indexing.importlib.import_module
    monkeypatch.setattr(indexing, "_sentence_model", lambda: Model())
    monkeypatch.setattr(
        indexing.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(inference_mode=nullcontext) if name == "torch" else original(name)
        ),
    )
    vector = indexing._sentence_embeddings(["1 2 3 4 5 6 7 8 9"])
    assert seen == [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    np.testing.assert_allclose(vector, indexing.normalize(np.array([[8, 1]], dtype=np.float32)))


def test_transformers_clap_adapter_uses_matching_audio_and_text_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    tensor = SimpleNamespace(
        detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: values))
    )

    def processor(**kwargs: Any) -> dict[str, object]:
        calls.append(kwargs)
        return {"features": "processed"}

    def encode(**kwargs: Any) -> SimpleNamespace:
        assert kwargs == {"features": "processed"}
        return tensor

    monkeypatch.setattr(
        indexing.importlib,
        "import_module",
        lambda name: SimpleNamespace(inference_mode=nullcontext),
    )
    backend = indexing.ClapEncoder(
        SimpleNamespace(get_audio_features=encode, get_text_features=encode), processor
    )
    np.testing.assert_array_equal(backend.audio(np.zeros(48000, dtype=np.float32)), values)
    np.testing.assert_array_equal(backend.text(["a calm voice"]), values)
    assert calls[0]["sampling_rate"] == 48000
    assert len(calls[0]["audio"]) == 48000
    assert calls[0]["return_tensors"] == "pt"
    assert calls[1]["text"] == ["a calm voice"]
