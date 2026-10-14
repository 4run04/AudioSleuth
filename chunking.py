"""Speaker-stratified turns, lexical topic boundaries, and exact snippet slicing."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.feature_extraction.text import TfidfVectorizer

from schemas import SAMPLE_RATE, AudioSleuthError, Chunk, Recording, Word


def _sentence_end(word: Word) -> bool:
    return bool(re.search(r"""[.!?]["')]*$""", word.text))


def _topic_shift(before: list[Word], after: list[Word]) -> bool:
    """A transparent lexical-cohesion heuristic, not a semantic neural classifier."""
    texts = [" ".join(w.text for w in group) for group in (before[-25:], after[:25])]
    if any(len(re.findall(r"\w+", text)) < 5 for text in texts):
        return False
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vectors = vectorizer.fit_transform(texts)
    except ValueError:
        return False  # All-stopword windows have no lexical evidence for a topic shift.
    return float((vectors[0] @ vectors[1].T).toarray()[0, 0]) < 0.12


def chunk_recording(
    recording: Recording, directory: Path, *, max_seconds: float = 120.0
) -> list[Chunk]:
    """Never cross speakers or exceed the hard duration cap; prefer natural breaks."""
    if not 1 <= max_seconds <= 120:
        raise ValueError("Chunk cap must lie between 1 and 120 seconds.")
    directory.mkdir(parents=True, exist_ok=True)
    turns: list[list[Word]] = []
    for word in recording.words:
        if word.end > recording.duration + 1e-6:
            raise AudioSleuthError("A word extends beyond the source audio.")
        if (
            not turns
            or turns[-1][-1].speaker_id != word.speaker_id
            or word.start - turns[-1][-1].end > 1.2
        ):
            turns.append([])
        turns[-1].append(word)
    groups: list[list[Word]] = []
    max_ms = int(max_seconds * 1000)
    for turn in turns:
        start = 0
        while start < len(turn):
            end, preferred = start, None
            while end < len(turn):
                span_ms = math.ceil(turn[end].end * 1000) - math.floor(turn[start].start * 1000)
                if span_ms > max_ms:
                    break
                end += 1
                if _sentence_end(turn[end - 1]):
                    preferred = end
                    if end < len(turn) and turn[end - 1].end - turn[start].start >= 8:
                        pause = turn[end].start - turn[end - 1].end
                        if pause >= 0.35 or _topic_shift(turn[start:end], turn[end:]):
                            break
            if end == start:
                raise AudioSleuthError("A single word exceeds the maximum chunk duration.")
            if end < len(turn) and end != preferred and preferred is not None:
                end = preferred
            groups.append(turn[start:end])
            start = end
    chunks: list[Chunk] = []
    for index, words in enumerate(groups):
        start_ms = max(0, math.floor(words[0].start * 1000))
        end_ms = min(int(recording.duration * 1000), math.ceil(max(w.end for w in words) * 1000))
        if end_ms <= start_ms:
            raise AudioSleuthError("An aligned chunk is shorter than one millisecond.")
        chunk_id = f"{recording.source_sha256[:12]}-{index:04d}"
        path = directory / f"{chunk_id}.wav"
        snippet = recording.samples[start_ms * 16 : end_ms * 16]
        sf.write(path, snippet, SAMPLE_RATE, subtype="FLOAT")
        weights = np.array([w.end - w.start for w in words])
        confidence = float(np.average([w.confidence for w in words], weights=weights))
        overlap = float(np.average([w.speaker_confidence for w in words], weights=weights))
        timing = (
            "estimated"
            if any(w.timing == "estimated" for w in words)
            else "synthetic"
            if any(w.timing == "synthetic" for w in words)
            else "aligned"
        )
        chunks.append(
            Chunk(
                chunk_id,
                words[0].speaker_id,
                start_ms,
                end_ms,
                " ".join(w.text for w in words),
                confidence,
                overlap,
                timing,
                str(path),
                recording.source_sha256,
            )
        )
    if not chunks:
        raise AudioSleuthError("No valid speech chunks could be created.")
    return chunks
