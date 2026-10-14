"""Shared domain objects. Scores describe evidence, not calibrated probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Samples = NDArray[np.float32]
Vectors = NDArray[np.float32]
SAMPLE_RATE = 16_000
MAX_SECONDS = 900
MAX_BYTES = 100 * 1024 * 1024


class AudioSleuthError(Exception):
    """An actionable ingestion, validation, model, or indexing failure."""


@dataclass(frozen=True)
class Word:
    """A word interval in source seconds, with explicit alignment provenance."""

    text: str
    start: float
    end: float
    confidence: float = 1.0
    speaker_id: str = "UNKNOWN"
    speaker_confidence: float = 0.0
    timing: Literal["aligned", "estimated", "synthetic"] = "aligned"

    def __post_init__(self) -> None:
        if not self.text.strip() or not all(math.isfinite(t) for t in (self.start, self.end)):
            raise AudioSleuthError("Words need text and finite timestamp boundaries.")
        if not 0 <= self.start < self.end:
            raise AudioSleuthError("Word intervals must have positive duration.")
        if not 0 <= self.confidence <= 1 or not 0 <= self.speaker_confidence <= 1:
            raise AudioSleuthError("Word quality scores must be finite and in [0, 1].")


@dataclass(frozen=True)
class SpeakerInterval:
    """A half-open diarization interval; overlaps are allowed."""

    start: float
    end: float
    speaker_id: str

    def __post_init__(self) -> None:
        if (
            not all(math.isfinite(t) for t in (self.start, self.end))
            or not 0 <= self.start < self.end
            or not self.speaker_id
        ):
            raise AudioSleuthError("Speaker intervals need a label and finite positive duration.")


@dataclass
class Recording:
    """Decoded canonical mono audio and words referencing the same time origin."""

    samples: Samples
    source_name: str
    source_sha256: str
    words: list[Word]
    intervals: list[SpeakerInterval]
    mode: str
    notices: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


@dataclass(frozen=True)
class Chunk:
    """One speaker's words and an exact sample slice of the canonical source."""

    chunk_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    transcript: str
    confidence: float
    speaker_confidence: float
    timing: str
    audio_path: str
    source_sha256: str

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000

    def payload(self) -> dict[str, str | int | float]:
        """Serialize the retrieval payload without embedding data or private paths."""
        return {
            "chunk_id": self.chunk_id,
            "speaker_id": self.speaker_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "transcript": self.transcript,
            "confidence": self.confidence,
            "speaker_confidence": self.speaker_confidence,
            "timing": self.timing,
            "audio_path": Path(self.audio_path).name,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class SearchHit:
    """Hybrid relevance and component cosine similarities for one chunk."""

    chunk: Chunk
    score: float
    text_score: float
    audio_score: float
