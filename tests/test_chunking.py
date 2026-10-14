"""Exact waveform slices, duration caps, speaker turns, and word preservation."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chunking import chunk_recording
from schemas import SAMPLE_RATE, AudioSleuthError, Recording, Word
from service import Collection


def test_every_demo_clip_is_exact_source_slice(demo_collection: Collection) -> None:
    recording = demo_collection.recording
    all_text = []
    for chunk in demo_collection.index.chunks:
        audio, rate = sf.read(chunk.audio_path, dtype="float32")
        assert rate == SAMPLE_RATE
        assert len(audio) == (chunk.end_ms - chunk.start_ms) * 16
        np.testing.assert_array_equal(
            audio, recording.samples[chunk.start_ms * 16 : chunk.end_ms * 16]
        )
        assert chunk.end_ms - chunk.start_ms <= 120_000
        all_text.extend(chunk.transcript.split())
    assert all_text == [word.text for word in recording.words]


def test_long_monologue_is_split_without_dropped_words(tmp_path: Path) -> None:
    words = [
        Word(f"word{i}" + ("." if i % 50 == 49 else ""), i + 0.01, i + 0.9, speaker_id="A")
        for i in range(260)
    ]
    recording = Recording(
        np.zeros(261 * SAMPLE_RATE, dtype=np.float32),
        "test.wav",
        "b" * 64,
        words,
        [],
        "test",
    )
    chunks = chunk_recording(recording, tmp_path)
    assert len(chunks) >= 3
    assert all(chunk.end_ms - chunk.start_ms <= 120_000 for chunk in chunks)
    assert " ".join(chunk.transcript for chunk in chunks) == " ".join(w.text for w in words)


def test_turn_boundaries_and_silence_split(tmp_path: Path) -> None:
    words = [
        Word("one", 0, 1, speaker_id="A"),
        Word("two", 1, 2, speaker_id="B"),
        Word("three", 4, 5, speaker_id="B"),
    ]
    recording = Recording(
        np.zeros(SAMPLE_RATE * 6, dtype=np.float32), "x", "f" * 64, words, [], "test"
    )
    chunks = chunk_recording(recording, tmp_path)
    assert [c.speaker_id for c in chunks] == ["A", "B", "B"]
    assert [c.start_ms for c in chunks] == [0, 1000, 4000]


def test_invalid_chunk_and_word_bounds(tmp_path: Path) -> None:
    recording = Recording(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        "x",
        "f" * 64,
        [Word("outside", 0, 2)],
        [],
        "test",
    )
    with pytest.raises(AudioSleuthError, match="beyond"):
        chunk_recording(recording, tmp_path)
    with pytest.raises(ValueError):
        chunk_recording(recording, tmp_path, max_seconds=121)
