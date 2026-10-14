"""Assemble SAPI-generated speech and persist the portable, precomputed demo index."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chunking import chunk_recording  # noqa: E402
from indexing import build_index, save_demo_index  # noqa: E402
from ingestion import ASSETS, file_sha256, load_demo  # noqa: E402
from schemas import SAMPLE_RATE  # noqa: E402


def main() -> None:
    """Only regeneration needs Windows; all emitted assets are platform-independent."""
    generated = ASSETS / "_generated"
    events = json.loads((generated / "events.json").read_text(encoding="utf-8-sig"))
    pieces = [np.zeros(SAMPLE_RATE // 4, dtype=np.float32)]
    position = len(pieces[0])
    words: list[dict[str, object]] = []
    intervals: list[dict[str, object]] = []
    for turn in events:
        samples, rate = sf.read(generated / turn["audio"], dtype="float32")
        if rate != SAMPLE_RATE:
            raise ValueError("Synthetic speech must be 16 kHz mono.")
        samples = np.clip(samples * turn["gain"], -0.99, 0.99).astype(np.float32)
        start = position / SAMPLE_RATE
        duration = len(samples) / SAMPLE_RATE
        boundaries = turn["words"]
        if not boundaries:
            raise ValueError("The speech synthesizer emitted no word boundaries.")
        for index, event in enumerate(boundaries):
            left = float(event["Start"])
            right = (
                float(boundaries[index + 1]["Start"]) if index + 1 < len(boundaries) else duration
            )
            if right <= left:
                raise ValueError("Synthetic word boundaries must be strictly increasing.")
            words.append(
                {
                    "text": event["Text"],
                    "start": start + left,
                    "end": start + right,
                    "confidence": 1.0,
                    "timing": "synthetic",
                }
            )
        intervals.append(
            {
                "start": start,
                "end": start + duration,
                "speaker_id": turn["speaker"],
            }
        )
        silence = np.zeros(int(SAMPLE_RATE * 0.65), dtype=np.float32)
        pieces.extend((samples, silence))
        position += len(samples) + len(silence)
    path = ASSETS / "demo.wav"
    sf.write(path, np.concatenate(pieces), SAMPLE_RATE, subtype="PCM_16")
    manifest = {
        "source_sha256": file_sha256(path),
        "license_note": "Original AudioSleuth script; locally generated synthetic voices.",
        "word_timing": "TTS start events with next-event word-end estimates.",
        "words": words,
        "intervals": intervals,
    }
    (ASSETS / "demo.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    recording = load_demo()
    with tempfile.TemporaryDirectory(prefix="audiosleuth-build-") as temporary:
        chunks = chunk_recording(recording, Path(temporary))
        index = build_index(chunks)
        save_demo_index(index, ASSETS)
    print(f"Bundled {recording.duration:.3f}s, {len(words)} words, {len(chunks)} indexed chunks.")


if __name__ == "__main__":
    main()
