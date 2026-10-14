"""Audio decoding, diarization evidence, and timestamp failure cases."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import ingestion
from ingestion import _aligned_words, assign_speakers, ingest_upload, load_audio
from schemas import SAMPLE_RATE, AudioSleuthError, SpeakerInterval, Word


def test_maximum_overlap_and_deterministic_tie() -> None:
    intervals = [SpeakerInterval(0, 1.4, "A"), SpeakerInterval(1.4, 3, "B")]
    words = [Word("first", 0.2, 0.5), Word("cross", 1.3, 1.9), Word("tie", 1.2, 1.6)]
    assigned = assign_speakers(words, intervals)
    assert [w.speaker_id for w in assigned[:2]] == ["A", "B"]
    assert assigned[1].speaker_confidence == pytest.approx(5 / 6)
    tie = assign_speakers(
        [Word("tie", 0, 2)], [SpeakerInterval(0, 1, "Z"), SpeakerInterval(1, 2, "A")]
    )
    assert tie[0].speaker_id == "A"


def test_duplicate_intervals_are_unioned_not_double_counted() -> None:
    word = Word("hello", 0, 1)
    result = assign_speakers(
        [word],
        [SpeakerInterval(0, 0.6, "A"), SpeakerInterval(0, 0.6, "A"), SpeakerInterval(0.2, 1, "B")],
    )
    assert result[0].speaker_id == "B"
    assert result[0].speaker_confidence == pytest.approx(0.8)


def test_unknown_and_nearest_assignments_are_not_fake_confidence() -> None:
    result = assign_speakers(
        [Word("near", 1.1, 1.2), Word("far", 5, 6)], [SpeakerInterval(0, 1, "A")]
    )
    assert [(w.speaker_id, w.speaker_confidence) for w in result] == [("A", 0), ("UNKNOWN", 0)]
    assert assign_speakers([Word("no", 0, 1)], [])[0].speaker_id == "UNKNOWN"


@pytest.mark.parametrize(
    "start,end", [(0, 0), (1, 0), (-1, 1), (0, float("inf")), (float("nan"), 1)]
)
def test_invalid_word_boundaries_rejected(start: float, end: float) -> None:
    with pytest.raises(AudioSleuthError):
        Word("test", start, end)


def test_stereo_resampling(tmp_path: Path) -> None:
    rate = 48_000
    signal = np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32) * 0.25
    source = tmp_path / "stereo.wav"
    sf.write(source, np.stack([signal, signal], axis=1), rate)
    audio = load_audio(source)
    assert audio.shape == (SAMPLE_RATE,)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()


def test_mp3_decode(tmp_path: Path) -> None:
    if "MP3" not in sf.available_formats():
        pytest.skip("This platform's libsndfile cannot encode the MP3 test fixture.")
    source = tmp_path / "sample.mp3"
    sf.write(source, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, format="MP3")
    assert abs(len(load_audio(source)) - SAMPLE_RATE) < SAMPLE_RATE // 5


def test_invalid_files_and_short_audio(tmp_path: Path) -> None:
    with pytest.raises(AudioSleuthError, match="existing"):
        load_audio(tmp_path / "missing.wav")
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(100), SAMPLE_RATE)
    with pytest.raises(AudioSleuthError, match="100 milliseconds"):
        load_audio(path)


def test_unaligned_words_are_interpolated_and_labeled() -> None:
    segments: list[dict[str, Any]] = [
        {
            "start": 0,
            "end": 3,
            "text": "one missing three",
            "words": [
                {"word": "one", "start": 0, "end": 1, "score": 0.9},
                {"word": "missing"},
                {"word": "three", "start": 2, "end": 3, "score": 0.8},
            ],
        }
    ]
    result = _aligned_words(segments, 3)
    assert len(result) == 3
    assert result[1].start == 1 and result[1].end == 2
    assert result[1].timing == "estimated" and result[1].confidence == 0


def test_missing_whisperx_never_fabricates_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.wav"
    sf.write(path, np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    original = importlib.import_module

    def missing(name: str, package: str | None = None) -> Any:
        if name == "whisperx":
            raise ImportError("intentionally unavailable")
        return original(name, package)

    monkeypatch.setattr(ingestion.importlib, "import_module", missing)
    with pytest.raises(AudioSleuthError, match="was not indexed"):
        ingest_upload(path)


@pytest.mark.parametrize("with_token", [False, True])
def test_whisperx_cpu_and_diarization_adapter_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_token: bool
) -> None:
    path = tmp_path / "speech.wav"
    sf.write(path, np.ones(SAMPLE_RATE, dtype=np.float32) * 0.01, SAMPLE_RATE)
    calls: dict[str, Any] = {}
    segments = [
        {
            "start": 0,
            "end": 1,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0, "end": 0.5, "score": 0.9},
                {"word": "world", "start": 0.5, "end": 1, "score": 0.8},
            ],
        }
    ]

    class Model:
        def transcribe(self, samples: np.ndarray, batch_size: int) -> dict[str, Any]:
            assert samples.shape == (SAMPLE_RATE,) and batch_size == 1
            return {"language": "en", "segments": segments}

    def load_model(name: str, device: str, compute_type: str) -> Model:
        calls["asr"] = (device, compute_type)
        return Model()

    def load_align_model(language_code: str, device: str) -> tuple[str, dict[str, str]]:
        assert language_code == "en" and device == "cpu"
        return "aligner", {}

    def align(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"segments": segments}

    class Diarizer:
        def __init__(self, token: str, device: str) -> None:
            assert token == "test-only-token" and device == "cpu"
            calls["diarized"] = True

        def __call__(self, samples: np.ndarray) -> SimpleNamespace:
            return SimpleNamespace(
                itertuples=lambda: iter([SimpleNamespace(start=0, end=1, speaker="A")])
            )

    modules = {
        "whisperx": SimpleNamespace(
            load_model=load_model, load_align_model=load_align_model, align=align
        ),
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "whisperx.diarize": SimpleNamespace(DiarizationPipeline=Diarizer),
    }
    original = importlib.import_module
    monkeypatch.setattr(
        ingestion.importlib,
        "import_module",
        lambda name, package=None: modules[name] if name in modules else original(name, package),
    )
    if with_token:
        monkeypatch.setenv("HF_TOKEN", "test-only-token")
    else:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    result = ingest_upload(path)
    assert calls["asr"] == ("cpu", "int8")
    assert [word.text for word in result.words] == ["hello", "world"]
    assert all(w.speaker_id == ("A" if with_token else "UNKNOWN") for w in result.words)
    if not with_token:
        assert any("HF_TOKEN" in notice for notice in result.notices)
