"""Cosine hybrid ranking with stable ties and pre-ranking metadata filters."""

from __future__ import annotations

import math

import numpy as np

from indexing import DualIndex
from schemas import AudioSleuthError, SearchHit


def search(
    index: DualIndex,
    query: str,
    *,
    alpha: float = 0.65,
    speaker: str | None = None,
    min_confidence: float = 0.0,
    min_score: float = -1.0,
    top_k: int = 5,
) -> list[SearchHit]:
    """Rank by alpha*cos(text) + (1-alpha)*cos(audio), not a probability.

    min_confidence filters the payload's word-alignment confidence. min_score
    filters hybrid relevance separately. Negative cosine values are retained.
    """
    if not query.strip() or len(query) > 1000:
        raise AudioSleuthError("Enter a query of 1 to 1,000 characters.")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise AudioSleuthError("Semantic weight must be between zero and one.")
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise AudioSleuthError("Alignment confidence must be between zero and one.")
    if not math.isfinite(min_score) or not -1 <= min_score <= 1 or not 1 <= top_k <= 20:
        raise AudioSleuthError("Use a score threshold in [-1, 1] and top_k in [1, 20].")
    candidates = [
        i
        for i, chunk in enumerate(index.chunks)
        if (speaker is None or chunk.speaker_id == speaker) and chunk.confidence >= min_confidence
    ]
    if not candidates:
        return []
    text, audio = index.encode_query(query, include_text=alpha > 0, include_audio=alpha < 1)
    if not np.any(text) and not np.any(audio):
        return []
    text_scores = np.clip(index.text_vectors @ text, -1, 1)
    audio_scores = np.clip(index.audio_vectors @ audio, -1, 1)
    scores = alpha * text_scores + (1 - alpha) * audio_scores
    ordered = sorted(candidates, key=lambda i: (-float(scores[i]), index.chunks[i].start_ms))
    return [
        SearchHit(index.chunks[i], float(scores[i]), float(text_scores[i]), float(audio_scores[i]))
        for i in ordered
        if scores[i] >= min_score
    ][:top_k]
