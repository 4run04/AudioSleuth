"""Exercise real local ML adapters; never treat an offline fallback as a neural success."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from httpx import HTTPError  # noqa: E402

from indexing import _clap_audio, _clap_model, _sentence_embeddings, normalize  # noqa: E402
from ingestion import load_demo  # noqa: E402
from retrieval import search  # noqa: E402
from schemas import SAMPLE_RATE, AudioSleuthError  # noqa: E402
from service import create_collection, dispose  # noqa: E402


def check_text() -> dict[str, object]:
    """Require genuine sentence-model embeddings, including a long transcript."""
    values = _sentence_embeddings(
        [
            "A heated disagreement about interest rates.",
            "Interest rates were discussed. " * 150 + "Battery storage became the final topic.",
        ]
    )
    if values.shape[0] != 2 or values.shape[1] < 100:
        raise AudioSleuthError("Unexpected sentence-model output shape.")
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
    return {"backend": "sentence-transformers", "shape": list(values.shape), "status": "passed"}


def check_audio() -> dict[str, object]:
    """Require CLAP audio and text to inhabit the same finite vector space."""
    audio = _clap_audio(load_demo().samples[: SAMPLE_RATE * 12])
    text = normalize(
        np.asarray(
            _clap_model().text(["A person speaking calmly."]),
            dtype=np.float32,
        )
    )[0]
    if audio.shape != text.shape or len(audio) < 100:
        raise AudioSleuthError("CLAP audio/text embedding dimensions differ.")
    return {
        "backend": "CLAP",
        "dimensions": len(audio),
        "cosine": float(audio @ text),
        "status": "passed",
    }


def check_asr() -> dict[str, object]:
    """Exercise real upload ASR, alignment, chunking, indexing, and retrieval."""
    with tempfile.TemporaryDirectory(prefix="audiosleuth-asr-check-") as temporary:
        path = Path(temporary) / "speech.wav"
        sf.write(path, load_demo().samples[: SAMPLE_RATE * 12], SAMPLE_RATE)
        collection = create_collection(upload=path)
    try:
        recording = collection.recording
        if not recording.words or recording.mode.startswith("Bundled"):
            raise AudioSleuthError("Real ASR did not produce aligned words.")
        hits = search(collection.index, recording.words[0].text, alpha=1)
        if not hits:
            raise AudioSleuthError("The real ASR output could not be retrieved.")
        return {
            "backend": recording.mode,
            "words": len(recording.words),
            "chunks": len(collection.index.chunks),
            "speakers": sorted({word.speaker_id for word in recording.words}),
            "notices": recording.notices,
            "status": "passed",
        }
    finally:
        dispose(collection)


def main() -> int:
    """Model weights may download; failures return nonzero with a named backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["text", "audio", "asr", "all"], default="text")
    parser.add_argument("--asr-model", default="tiny.en", help="Small English smoke-test model.")
    args = parser.parse_args()
    os.environ["AUDIOSLEUTH_ASR_MODEL"] = args.asr_model
    checks = {"text": check_text, "audio": check_audio, "asr": check_asr}
    selected = checks if args.backend == "all" else {args.backend: checks[args.backend]}
    failed = False
    for name, check in selected.items():
        try:
            print(json.dumps(check()), flush=True)
        except (
            AudioSleuthError,
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            HTTPError,
            AssertionError,
        ) as exc:
            failed = True
            print(
                json.dumps(
                    {
                        "backend": name,
                        "status": "failed",
                        "exception": type(exc).__name__,
                        "action": "Check dependencies, model access, network and decoder setup. "
                        "This failure is not counted as a working neural backend.",
                    }
                ),
                flush=True,
            )
            print(f"{name} diagnostic: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
