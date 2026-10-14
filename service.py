"""Session-local collection lifecycle and an auditable provenance manifest."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from chunking import chunk_recording
from indexing import DualIndex, build_index, load_demo_index
from ingestion import ASSETS, ingest_upload, load_demo
from schemas import SAMPLE_RATE, Recording

LOGGER = logging.getLogger(__name__)
CACHE_ROOT = Path(tempfile.gettempdir()) / "audiosleuth-collections"


@dataclass
class Collection:
    """One browser session's index and generated files, never a shared global corpus."""

    recording: Recording
    index: DualIndex
    directory: Path
    source_path: Path
    manifest_path: Path


def _remove_workspace(directory: Path) -> None:
    resolved = directory.resolve()
    if resolved.parent != CACHE_ROOT.resolve() or not resolved.name.startswith("collection-"):
        raise ValueError("Refusing to remove a directory not owned by this collection.")
    if resolved.exists():
        shutil.rmtree(resolved)


def dispose(collection: Collection | None) -> None:
    """Delete only the named collection workspace on expiry or explicit replacement."""
    if collection is not None:
        _remove_workspace(collection.directory)


def _build_collection(
    recording: Recording, *, neural: bool, precomputed: bool = False
) -> Collection:
    """Build in a fresh owned workspace and remove partial files after any failure."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="collection-", dir=CACHE_ROOT))
    completed = False
    try:
        chunks = chunk_recording(recording, directory)
        index = (
            load_demo_index(chunks, ASSETS)
            if precomputed and not neural
            else build_index(chunks, neural=neural)
        )
        source_path = directory / "source.wav"
        sf.write(source_path, recording.samples, SAMPLE_RATE, subtype="FLOAT")
        manifest_path = directory / "provenance.json"
        manifest = {
            "source_name": recording.source_name,
            "source_sha256": recording.source_sha256,
            "sample_rate": SAMPLE_RATE,
            "mode": recording.mode,
            "text_backend": index.text_backend,
            "audio_backend": index.audio_backend,
            "score_meaning": "Cosine hybrid relevance, not calibrated probability.",
            "notices": recording.notices + index.notices,
            "chunks": [chunk.payload() for chunk in chunks],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        collection = Collection(recording, index, directory, source_path, manifest_path)
        completed = True
        return collection
    finally:
        if not completed:
            _remove_workspace(directory)


def create_collection(*, upload: Path | None = None, neural: bool = False) -> Collection:
    """Construct atomically; a failed ingestion never replaces the previous collection."""
    recording = load_demo() if upload is None else ingest_upload(upload)
    return _build_collection(recording, neural=neural, precomputed=upload is None)


def reindex_collection(current: Collection, *, neural: bool) -> Collection:
    """Change encoders without retranscribing or relabeling the source recording."""
    return _build_collection(current.recording, neural=neural)
