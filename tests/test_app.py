"""Dashboard callbacks, HTML safety, isolated state, and cleanup."""

from pathlib import Path

import gradio as gr
import pytest

from app import (
    backend_status,
    build_app,
    clear_callback,
    highlight,
    load_demo_callback,
    reindex_callback,
    search_callback,
    timestamp,
)
from service import CACHE_ROOT, Collection, create_collection, dispose


def test_html_highlights_escape_untrusted_text() -> None:
    output = highlight('<script>alert("rates")</script> rates & taxes', "rates <script>")
    assert "<script>" not in output
    assert "&lt;" in output and "&amp;" in output
    assert "<mark>rates</mark>" in output
    assert timestamp(61_237) == "01:01.237"


def test_dashboard_constructs_without_neural_dependencies() -> None:
    app = build_app()
    assert len(app.blocks) > 30


def test_search_callback_clip_outputs_and_empty_state(demo_collection: Collection) -> None:
    outputs = search_callback("heated rates", 0.65, "All speakers", 0, 0, demo_collection)
    assert len(outputs) == 16
    assert "moments found" in outputs[0]
    assert Path(outputs[3]).is_file()
    with pytest.raises(gr.Error):
        search_callback("test", 0.5, "All speakers", 0, 0, None)
    empty = search_callback("rates", 0.5, "missing", 0, 0, demo_collection)
    assert "No matches" in empty[0]
    assert all(empty[i] is None for i in (3, 6, 9, 12, 15))


def test_collections_are_independent_and_cleanup_is_scoped() -> None:
    first, second = create_collection(), create_collection()
    try:
        assert first.directory != second.directory
        assert first.index is not second.index
        first.index.chunks.clear()
        assert len(second.index.chunks) > 0
        dispose(first)
        assert not first.directory.exists()
        assert second.directory.exists()
    finally:
        dispose(first)
        dispose(second)


def test_demo_button_replaces_only_its_own_previous_collection() -> None:
    old = create_collection()
    outputs = load_demo_callback(old)
    new = outputs[0]
    try:
        assert isinstance(new, Collection)
        assert len(outputs) == 23
        assert not old.directory.exists()
        assert new.directory.parent == CACHE_ROOT
    finally:
        dispose(new)


def test_reindex_preserves_transcript_source_and_speakers() -> None:
    old = create_collection()
    words = old.recording.words.copy()
    source_hash = old.recording.source_sha256
    result = reindex_callback(False, old)
    current = result[0]
    try:
        assert len(result) == 23
        assert current.recording.words == words
        assert current.recording.source_sha256 == source_hash
        assert not old.directory.exists()
        assert current.directory.exists()
        assert len(current.index.chunks) == 13
    finally:
        dispose(current)


def test_clear_resets_players_upload_and_only_current_collection() -> None:
    current, other = create_collection(), create_collection()
    try:
        result = clear_callback(current)
        assert len(result) == 24
        assert result[0] is None and result[-1] is None
        assert result[3] == [] and result[4] is None and result[6] is None
        assert not current.directory.exists()
        assert other.directory.exists()
        assert all(result[i] is None for i in (10, 13, 16, 19, 22))
        assert len(clear_callback(None)) == 24
    finally:
        dispose(current)
        dispose(other)


def test_backend_status_never_displays_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token-not-for-display")
    status = backend_status()
    assert "Configured" in status
    assert "test-token-not-for-display" not in status
