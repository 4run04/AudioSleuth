"""Bounded audio decoding, optional WhisperX alignment, and overlap diarization."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from httpx import HTTPError
from scipy.signal import resample_poly

from schemas import (
    MAX_BYTES,
    MAX_SECONDS,
    SAMPLE_RATE,
    AudioSleuthError,
    Recording,
    Samples,
    SpeakerInterval,
    Word,
)

LOGGER = logging.getLogger(__name__)
ASSETS = Path(__file__).parent / "assets"


def file_sha256(path: Path) -> str:
    """Hash original encoded source bytes, not a mutable path or transcript."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio(path: Path) -> Samples:
    """Decode WAV/MP3 and resample to 16 kHz mono, enforcing resource limits."""
    if not path.is_file() or path.suffix.lower() not in {".wav", ".mp3"}:
        raise AudioSleuthError("Upload an existing .wav or .mp3 file.")
    if path.stat().st_size > MAX_BYTES:
        raise AudioSleuthError("The audio exceeds the 100 MiB upload limit.")
    try:
        with sf.SoundFile(path) as stream:
            rate = stream.samplerate
            if stream.frames / rate > MAX_SECONDS:
                raise AudioSleuthError("Audio must be at most 15 minutes long.")
            if stream.channels > 8:
                raise AudioSleuthError("Audio with more than eight channels is not supported.")
            data = stream.read(dtype="float32", always_2d=True).mean(axis=1)
        factor = math.gcd(rate, SAMPLE_RATE)
        samples = np.asarray(
            resample_poly(data, SAMPLE_RATE // factor, rate // factor), dtype=np.float32
        )
    except sf.LibsndfileError as exc:
        LOGGER.warning("libsndfile could not decode %s: %s", path.name, exc)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise AudioSleuthError(
                "This audio could not be decoded. Install FFmpeg or convert it to PCM WAV."
            ) from exc
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-t",
                    str(MAX_SECONDS + 1),
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.SubprocessError, OSError) as decode_error:
            raise AudioSleuthError(
                "FFmpeg could not decode the audio within 120 seconds."
            ) from decode_error
        samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(samples) < SAMPLE_RATE // 10:
        raise AudioSleuthError("Audio must contain at least 100 milliseconds.")
    if len(samples) > MAX_SECONDS * SAMPLE_RATE:
        raise AudioSleuthError("Audio must be at most 15 minutes long.")
    if not np.isfinite(samples).all():
        raise AudioSleuthError("Decoded audio contains nonfinite samples.")
    return np.clip(samples, -1, 1).astype(np.float32)


def assign_speakers(
    words: list[Word], intervals: list[SpeakerInterval], *, nearest_limit: float = 0.25
) -> list[Word]:
    """Maximize unioned temporal overlap per speaker for every word.

    Unknown regions stay UNKNOWN. A nearest-turn assignment is allowed only
    within 250 ms and carries zero overlap confidence, never invented certainty.
    Simultaneous speakers are reduced to one label with deterministic tie-breaking.
    """
    if nearest_limit < 0:
        raise ValueError("Nearest-speaker distance cannot be negative.")
    for interval in intervals:
        if not 0 <= interval.start < interval.end or not interval.speaker_id:
            raise AudioSleuthError("Diarization contains an invalid interval.")
    assigned: list[Word] = []
    for word in words:
        if not 0 <= word.start < word.end or not word.text.strip():
            raise AudioSleuthError("Every word must have text and a positive, finite interval.")
        pieces: dict[str, list[tuple[float, float]]] = {}
        for interval in intervals:
            start, end = max(word.start, interval.start), min(word.end, interval.end)
            if end > start:
                pieces.setdefault(interval.speaker_id, []).append((start, end))
        overlaps: dict[str, float] = {}
        for speaker, spans in pieces.items():
            total, right = 0.0, -1.0
            for start, end in sorted(spans):
                total += max(0.0, end - max(start, right))
                right = max(right, end)
            overlaps[speaker] = total
        if overlaps:
            speaker = min(overlaps, key=lambda label: (-overlaps[label], label))
            confidence = overlaps[speaker] / (word.end - word.start)
        elif intervals:
            nearest = min(
                intervals,
                key=lambda i: (max(i.start - word.end, word.start - i.end, 0), i.speaker_id),
            )
            distance = max(nearest.start - word.end, word.start - nearest.end, 0)
            speaker = nearest.speaker_id if distance <= nearest_limit else "UNKNOWN"
            confidence = 0.0
        else:
            speaker, confidence = "UNKNOWN", 0.0
        assigned.append(replace(word, speaker_id=speaker, speaker_confidence=min(confidence, 1.0)))
    return assigned


def _aligned_words(segments: list[dict[str, Any]], duration: float) -> list[Word]:
    """Keep aligner word times; interpolate only explicitly unaligned tokens."""
    if not math.isfinite(duration) or duration <= 0:
        raise AudioSleuthError("The recording duration must be finite and positive.")

    def finite_time(value: Any) -> float:
        timestamp = float(value)
        if not math.isfinite(timestamp):
            raise AudioSleuthError("The aligner returned a nonfinite timestamp.")
        return timestamp

    words: list[Word] = []
    for segment in segments:
        tokens = segment.get("words") or [{"word": t} for t in segment["text"].split()]
        left = max(0.0, finite_time(segment["start"]))
        stop = min(duration, finite_time(segment["end"]))
        for token in tokens:
            for boundary in ("start", "end"):
                if token.get(boundary) is not None:
                    finite_time(token[boundary])
        index = 0
        while index < len(tokens):
            item = tokens[index]
            if item.get("start") is not None and item.get("end") is not None:
                start, end = max(0.0, float(item["start"])), min(duration, float(item["end"]))
                if end > start:
                    words.append(
                        Word(
                            str(item["word"]).strip(),
                            start,
                            end,
                            0.0 if item.get("score") is None else float(item["score"]),
                        )
                    )
                    left = end
                    index += 1
                    continue
            last = index + 1
            while last < len(tokens) and (
                tokens[last].get("start") is None or tokens[last].get("end") is None
            ):
                last += 1
            right = min(stop, float(tokens[last]["start"])) if last < len(tokens) else stop
            if right <= left:
                raise AudioSleuthError(
                    "The aligner produced words without recoverable timing. Try clearer audio."
                )
            step = (right - left) / (last - index)
            for offset, token in enumerate(tokens[index:last]):
                words.append(
                    Word(
                        str(token["word"]).strip(),
                        left + offset * step,
                        left + (offset + 1) * step,
                        0.0,
                        timing="estimated",
                    )
                )
            left, index = right, last
    return sorted(words, key=lambda word: (word.start, word.end))


def ingest_upload(path: Path) -> Recording:
    """Transcribe real uploads locally; never substitute synthetic words.

    CPU int8 is supported. Missing diarization credentials yield UNKNOWN
    speakers with a visible warning; ASR failure asks the user to load the demo.
    """
    samples = load_audio(path)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    try:
        whisperx = importlib.import_module("whisperx")
        torch = importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError) as exc:
        raise AudioSleuthError(
            "Real transcription needs the optional ML dependencies. Install requirements-ml.txt "
            "and FFmpeg, or click Load Pre-indexed Demo Audio. Your upload was not indexed."
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    notices: list[str] = []
    try:
        model = whisperx.load_model(
            os.getenv("AUDIOSLEUTH_ASR_MODEL", "small"),
            device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        transcript = model.transcribe(samples, batch_size=8 if device == "cuda" else 1)
        del model
        if not transcript.get("segments"):
            raise AudioSleuthError("No speech was detected in the upload.")
        aligner, metadata = whisperx.load_align_model(
            language_code=transcript["language"], device=device
        )
        aligned = whisperx.align(
            transcript["segments"],
            aligner,
            metadata,
            samples,
            device,
            return_char_alignments=False,
        )
        words = _aligned_words(aligned["segments"], len(samples) / SAMPLE_RATE)
        del aligner
    except AudioSleuthError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError, HTTPError) as exc:
        LOGGER.warning("ASR/alignment backend failed (%s).", type(exc).__name__)
        raise AudioSleuthError(
            "WhisperX transcription/alignment failed. Check model downloads, FFmpeg, and the "
            "optional environment. The bundled demo remains available; "
            "no upload transcript was invented."
        ) from exc
    if not words:
        raise AudioSleuthError("No aligned words were found.")
    intervals: list[SpeakerInterval] = []
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            diarize = importlib.import_module("whisperx.diarize")
            pipeline = diarize.DiarizationPipeline(token=token, device=device)
            frame = pipeline(samples)
            intervals = [
                SpeakerInterval(float(row.start), float(row.end), str(row.speaker))
                for row in frame.itertuples()
            ]
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            HTTPError,
        ) as exc:
            LOGGER.warning("Diarization unavailable (%s).", type(exc).__name__)
            notices.append(
                "Diarization failed. Speakers are UNKNOWN; no identities were fabricated."
            )
    else:
        notices.append("HF_TOKEN is missing: pyannote is disabled and all speakers are UNKNOWN.")
    if any(w.timing == "estimated" for w in words):
        notices.append(
            "Some unaligned words have interpolated times and confidence 0; "
            "millisecond display precision is not millisecond ASR accuracy."
        )
    return Recording(
        samples,
        path.name,
        file_sha256(path),
        assign_speakers(words, intervals),
        intervals,
        f"WhisperX / {device}",
        notices,
    )


def load_demo() -> Recording:
    """Load bundled original synthetic speech and synthesizer word-boundary events."""
    path = ASSETS / "demo.wav"
    try:
        metadata = json.loads((ASSETS / "demo.json").read_text(encoding="utf-8"))
        digest = file_sha256(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioSleuthError("Bundled demo assets are missing or damaged.") from exc
    if digest != metadata["source_sha256"]:
        raise AudioSleuthError("Demo audio hash does not match its provenance manifest.")
    samples = load_audio(path)
    words = [Word(**item) for item in metadata["words"]]
    intervals = [SpeakerInterval(**item) for item in metadata["intervals"]]
    return Recording(
        samples,
        "Market Signals / synthetic two-speaker podcast",
        digest,
        assign_speakers(words, intervals),
        intervals,
        "Bundled synthetic speech",
        [
            "Original scripted speech synthesized offline with two voices; not a real podcast.",
            "Word starts come from TTS boundary events; word ends are next-boundary estimates. "
            "Diarization labels are scripted ground truth, not pyannote predictions.",
        ],
    )
