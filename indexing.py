"""Immutable NumPy dual index with optional neural models and honest offline fallbacks."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from httpx import HTTPError
from scipy.signal import resample_poly
from sklearn.feature_extraction.text import TfidfVectorizer

from schemas import SAMPLE_RATE, AudioSleuthError, Chunk, Samples, Vectors

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

LOGGER = logging.getLogger(__name__)
MODEL_LOCK = threading.Lock()
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAP_MODEL = "laion/clap-htsat-unfused"
INDEX_VERSION = 1
ACOUSTIC_AXES = (
    "loud",
    "quiet",
    "bright",
    "dark",
    "dynamic",
    "steady",
    "noisy",
    "tonal",
    "silence",
)


def normalize(matrix: Vectors) -> Vectors:
    """Normalize each row, preserving zero vectors without NaNs."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise AudioSleuthError("An encoder produced malformed or nonfinite vectors.")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


@lru_cache(maxsize=1)
def _sentence_model() -> Any:
    module = importlib.import_module("sentence_transformers")
    return module.SentenceTransformer(TEXT_MODEL, device="cpu")


def _sentence_embeddings(texts: list[str]) -> Vectors:
    """Pool every token window instead of truncating long monologues at MiniLM's limit."""
    model = _sentence_model()
    tokenizer = model.tokenizer
    budget = int(model.max_seq_length) - tokenizer.num_special_tokens_to_add(pair=False)
    if budget <= 0:
        raise AudioSleuthError("The sentence model has no usable token capacity.")
    windows: list[Any] = []
    owners: list[int] = []
    lengths: list[int] = []
    for owner, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        if not tokens:
            raise AudioSleuthError("A transcript contains no encodable text.")
        for start in range(0, len(tokens), budget):
            window = tokens[start : start + budget]
            windows.append(
                tokenizer.prepare_for_model(
                    window, add_special_tokens=True, truncation=False, return_attention_mask=True
                )
            )
            owners.append(owner)
            lengths.append(len(window))
    torch = importlib.import_module("torch")
    batches: list[Vectors] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), 32):
            features = tokenizer.pad(windows[start : start + 32], padding=True, return_tensors="pt")
            features = {key: value.to(model.device) for key, value in features.items()}
            encoded = model(features)["sentence_embedding"].detach().cpu().numpy()
            batches.append(normalize(np.asarray(encoded, dtype=np.float32)))
    vectors = np.concatenate(batches)
    owner_array = np.asarray(owners)
    weight_array = np.asarray(lengths)
    pooled = [
        np.average(
            vectors[owner_array == owner], axis=0, weights=weight_array[owner_array == owner]
        )
        for owner in range(len(texts))
    ]
    return normalize(np.asarray(pooled, dtype=np.float32))


@dataclass
class ClapEncoder:
    """LAION weights through Transformers, compatible with WhisperX's NumPy 2 stack."""

    model: Any
    processor: Any

    def audio(self, samples_48k: Samples) -> Vectors:
        """Encode a single already-resampled waveform on the CPU."""
        inputs = self.processor(audio=samples_48k, sampling_rate=48_000, return_tensors="pt")
        torch = importlib.import_module("torch")
        with torch.inference_mode():
            result = self.model.get_audio_features(**inputs)
        return np.asarray(result.detach().cpu().numpy(), dtype=np.float32)

    def text(self, texts: list[str]) -> Vectors:
        """Encode tone descriptions in the same learned projection space."""
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        torch = importlib.import_module("torch")
        with torch.inference_mode():
            result = self.model.get_text_features(**inputs)
        return np.asarray(result.detach().cpu().numpy(), dtype=np.float32)


@lru_cache(maxsize=1)
def _clap_model() -> ClapEncoder:
    module = importlib.import_module("transformers")
    model = module.ClapModel.from_pretrained(CLAP_MODEL).eval()
    processor = module.ClapProcessor.from_pretrained(CLAP_MODEL)
    return ClapEncoder(model, processor)


def acoustic_features(samples: Samples) -> Vectors:
    """Nine interpretable signal descriptors, NOT learned emotion embeddings.

    Absolute RMS retains vocal intensity; spectral brightness and modulation
    expose simple timbre/activity cues. No transcript or speaker ID is used.
    """
    if len(samples) == 0 or not np.isfinite(samples).all():
        raise AudioSleuthError("Cannot embed an empty or invalid audio snippet.")
    frame_size, hop = 400, 160
    padded = np.pad(samples, (0, max(0, frame_size - len(samples))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_size)[::hop]
    energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    active = energy > max(0.002, float(energy.max()) * 0.08)
    if not active.any():
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    voiced = frames[active][:: max(1, int(active.sum()) // 200)]
    rms = float(np.sqrt(np.mean(samples**2)))
    loud = float(np.clip((20 * np.log10(max(rms, 1e-8)) + 45) / 35, 0, 1))
    spectrum = np.abs(np.fft.rfft(voiced * np.hanning(frame_size), axis=1))
    frequencies = np.fft.rfftfreq(frame_size, 1 / SAMPLE_RATE)
    centroid = np.mean((spectrum @ frequencies) / np.maximum(spectrum.sum(axis=1), 1e-12))
    bright = float(np.clip(centroid / 3500, 0, 1))
    dynamic = float(np.clip(np.std(energy) / max(float(np.mean(energy)), 1e-6), 0, 1))
    crossing = float(np.mean(np.signbit(voiced[:, 1:]) != np.signbit(voiced[:, :-1])))
    noisy = float(np.clip(crossing / 0.3, 0, 1))
    return np.array(
        [loud, 1 - loud, bright, 1 - bright, dynamic, 1 - dynamic, noisy, 1 - noisy, 0],
        dtype=np.float32,
    )


def acoustic_query(query: str) -> Vectors:
    """Map explicit tone words to descriptor axes; absent tone yields a zero vector."""
    terms = set(re.findall(r"[a-z]+", query.lower()))
    cues = (
        {"loud", "heated", "argument", "angry", "shouting", "intense", "energetic"},
        {"quiet", "calm", "soft", "whisper", "whispering", "gentle"},
        {"bright", "excited", "laugh", "laughing", "laughter", "cheerful"},
        {"dark", "deep", "low", "somber"},
        {"dynamic", "animated", "heated", "laughing", "laughter", "excited"},
        {"steady", "measured", "calm", "monotone", "serious"},
        {"noisy", "rough", "raspy", "breathy"},
        {"clear", "smooth", "tonal"},
        {"silence", "silent"},
    )
    return np.array([float(bool(terms & cue)) for cue in cues], dtype=np.float32)


def _clap_audio(samples: Samples) -> Vectors:
    """Pool all 10-second windows; never silently discard a long chunk's tail."""
    embeddings: list[Vectors] = []
    model = _clap_model()
    for offset in range(0, len(samples), SAMPLE_RATE * 10):
        window = samples[offset : offset + SAMPLE_RATE * 10]
        audio_48k = np.asarray(resample_poly(window, 3, 1), dtype=np.float32)
        values = model.audio(audio_48k)
        embeddings.append(normalize(np.asarray(values, dtype=np.float32))[0])
    return normalize(np.mean(np.stack(embeddings), axis=0, keepdims=True))[0]


@dataclass
class DualIndex:
    """Session-local payloads and normalized row-aligned text/audio vector matrices."""

    chunks: list[Chunk]
    text_vectors: Vectors
    audio_vectors: Vectors
    text_backend: str
    audio_backend: str
    vectorizer: TfidfVectorizer | None = None
    notices: list[str] = field(default_factory=list)

    def encode_query(
        self, query: str, *, include_text: bool = True, include_audio: bool = True
    ) -> tuple[Vectors, Vectors]:
        """Use matching encoders, skipping modalities with zero retrieval weight."""
        try:
            with MODEL_LOCK:
                if not include_text:
                    text = np.zeros((1, self.text_vectors.shape[1]), dtype=np.float32)
                elif self.text_backend == "sentence-transformers":
                    text = _sentence_embeddings([query])
                elif self.vectorizer is not None:
                    text = self.vectorizer.transform([query]).toarray().astype(np.float32)
                else:
                    raise AudioSleuthError("The lexical vectorizer is missing.")
                if not include_audio:
                    audio = np.zeros((1, self.audio_vectors.shape[1]), dtype=np.float32)
                elif self.audio_backend == "CLAP":
                    audio = np.asarray(
                        _clap_model().text([query]),
                        dtype=np.float32,
                    )
                else:
                    audio = acoustic_query(query)[None, :]
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, HTTPError) as exc:
            raise AudioSleuthError(
                "The indexed encoder is unavailable. Re-index in offline mode; "
                "query vectors cannot be mixed with a different model's index."
            ) from exc
        if text.shape != (1, self.text_vectors.shape[1]) or audio.shape != (
            1,
            self.audio_vectors.shape[1],
        ):
            raise AudioSleuthError("Query embedding dimensions do not match the stored index.")
        return normalize(text)[0], normalize(audio)[0]


def build_index(chunks: list[Chunk], *, neural: bool = False) -> DualIndex:
    """Build neural vectors if explicitly requested, with observable per-modality fallback."""
    if not chunks:
        raise AudioSleuthError("Cannot index an empty collection.")
    transcripts = [chunk.transcript for chunk in chunks]
    vectorizer: TfidfVectorizer | None = None
    text_backend, audio_backend = "TF-IDF (offline lexical)", "Signal descriptors (offline)"
    notices: list[str] = []
    text_vectors: Vectors | None = None
    audio_vectors: Vectors | None = None
    if neural:
        with MODEL_LOCK:
            try:
                text_vectors = _sentence_embeddings(transcripts)
                text_backend = "sentence-transformers"
            except (ImportError, OSError, RuntimeError, ValueError, TypeError, HTTPError) as exc:
                LOGGER.warning("Sentence embedding fallback (%s).", type(exc).__name__)
                notices.append(
                    "Sentence-transformers was unavailable; text search uses lexical TF-IDF, "
                    "not neural semantic embeddings."
                )
            try:
                rows = []
                for chunk in chunks:
                    audio, rate = sf.read(chunk.audio_path, dtype="float32")
                    if rate != SAMPLE_RATE:
                        raise AudioSleuthError("CLAP input is not canonical 16 kHz audio.")
                    rows.append(_clap_audio(audio))
                audio_vectors = normalize(np.stack(rows))
                audio_backend = "CLAP"
            except (ImportError, OSError, RuntimeError, ValueError, TypeError, HTTPError) as exc:
                LOGGER.warning("CLAP fallback (%s).", type(exc).__name__)
                notices.append(
                    "CLAP was unavailable; acoustic search uses signal descriptors and a "
                    "tone-word mapping, not zero-shot emotion recognition."
                )
    if text_vectors is None:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
            max_features=8192,
        )
        try:
            text_vectors = vectorizer.fit_transform(transcripts).toarray().astype(np.float32)
        except ValueError as exc:
            raise AudioSleuthError("Transcripts contain no searchable words.") from exc
    if audio_vectors is None:
        audio_vectors = normalize(
            np.stack(
                [
                    acoustic_features(sf.read(chunk.audio_path, dtype="float32")[0])
                    for chunk in chunks
                ]
            )
        )
    if not neural:
        notices.append(
            "Offline mode: TF-IDF + measured audio descriptors. "
            "No neural semantic, CLAP, or emotion-accuracy claim is made."
        )
    return DualIndex(
        chunks,
        normalize(text_vectors),
        audio_vectors,
        text_backend,
        audio_backend,
        vectorizer,
        notices,
    )


def save_demo_index(index: DualIndex, directory: Path) -> None:
    """Write a pickle-free offline index with verifiable ordered chunk identities."""
    if index.vectorizer is None or index.audio_backend == "CLAP":
        raise AudioSleuthError("Only the offline demo index is bundled.")
    manifest = {
        "version": INDEX_VERSION,
        "chunks": [chunk.payload() for chunk in index.chunks],
        "vocabulary": {key: int(value) for key, value in index.vectorizer.vocabulary_.items()},
        "text_backend": index.text_backend,
        "audio_backend": index.audio_backend,
    }
    np.savez_compressed(
        directory / "demo_index.npz",
        text=index.text_vectors,
        audio=index.audio_vectors,
        idf=index.vectorizer.idf_,
    )
    (directory / "demo_index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_demo_index(chunks: list[Chunk], directory: Path) -> DualIndex:
    """Load precomputed vectors, refusing mismatched audio, boundaries, or text."""
    try:
        manifest = json.loads((directory / "demo_index.json").read_text(encoding="utf-8"))
        if manifest["version"] != INDEX_VERSION:
            raise AudioSleuthError("Unsupported bundled index version.")
        if manifest["chunks"] != [chunk.payload() for chunk in chunks]:
            raise AudioSleuthError("Bundled index and source provenance do not match.")
        with np.load(directory / "demo_index.npz", allow_pickle=False) as archive:
            text = np.asarray(archive["text"], dtype=np.float32)
            audio = np.asarray(archive["audio"], dtype=np.float32)
            idf = archive["idf"].copy()
        if text.shape != (len(chunks), len(manifest["vocabulary"])):
            raise AudioSleuthError("Bundled text index has an invalid shape.")
        if audio.shape != (len(chunks), len(ACOUSTIC_AXES)):
            raise AudioSleuthError("Bundled audio index has an invalid shape.")
        if idf.shape != (text.shape[1],) or not np.isfinite(idf).all() or np.any(idf <= 0):
            raise AudioSleuthError(
                "Bundled IDF weights must be finite, positive, and vocabulary-sized."
            )
        if sorted(manifest["vocabulary"].values()) != list(range(text.shape[1])):
            raise AudioSleuthError("Bundled vocabulary indices are not contiguous and unique.")
        vectorizer = TfidfVectorizer(
            vocabulary=manifest["vocabulary"],
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        vectorizer.idf_ = idf
        return DualIndex(
            chunks,
            normalize(text),
            normalize(audio),
            manifest["text_backend"],
            manifest["audio_backend"],
            vectorizer,
            ["Loaded bundled precomputed vectors: lexical TF-IDF + audio signal descriptors."],
        )
    except (OSError, KeyError, ValueError) as exc:
        raise AudioSleuthError(
            "The bundled index is missing or corrupt; rebuild the demo assets."
        ) from exc
