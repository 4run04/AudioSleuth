"""AudioSleuth: local, session-isolated multimodal podcast retrieval dashboard."""

from __future__ import annotations

import argparse
import html
import importlib.util
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from indexing import acoustic_query  # noqa: E402
from retrieval import search  # noqa: E402
from schemas import AudioSleuthError, SearchHit  # noqa: E402
from service import Collection, create_collection, dispose, reindex_collection  # noqa: E402

ROOT = Path(__file__).parent
MAX_RESULTS = 5


def backend_status() -> str:
    """Report installation prerequisites without loading models or exposing credentials."""
    rows = [
        "| Component | Local prerequisite |",
        "|---|---|",
    ]
    for name, module in (
        ("WhisperX ASR", "whisperx"),
        ("Sentence-transformers", "sentence_transformers"),
        ("LAION-CLAP via Transformers", "transformers"),
    ):
        installed = importlib.util.find_spec(module) is not None
        rows.append(f"| {name} | {'Installed' if installed else 'Not installed'} |")
    rows.append(f"| Diarization token | {'Configured' if os.getenv('HF_TOKEN') else 'Missing'} |")
    rows.append(f"| FFmpeg executable | {'Found' if shutil.which('ffmpeg') else 'Not found'} |")
    return "\n".join(rows) + (
        "\n\nInstalled does not mean model weights are cached or imports are compatible. "
        "The bundled offline demo needs none of these optional prerequisites."
    )


def timestamp(milliseconds: int) -> str:
    """Format the precise indexed time boundary without rounding to whole seconds."""
    minutes, remaining = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remaining, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def highlight(transcript: str, query: str) -> str:
    """Escape all untrusted text before adding bounded query-word highlight markup."""
    terms = set(re.findall(r"\w{3,}", query, flags=re.UNICODE))
    if not terms:
        return html.escape(transcript)
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(terms)) + r")\b", re.I)
    result: list[str] = []
    last = 0
    for match in pattern.finditer(transcript):
        result.extend(
            (
                html.escape(transcript[last : match.start()]),
                f"<mark>{html.escape(match.group())}</mark>",
            )
        )
        last = match.end()
    result.append(html.escape(transcript[last:]))
    return "".join(result)


def hit_card(hit: SearchHit, query: str, rank: int, *, alpha: float = 0.65) -> str:
    """Display relevance separately from timing confidence and speaker overlap."""
    chunk = hit.chunk
    color = "mint" if chunk.speaker_id.endswith("00") else "violet"
    text_score = f"{hit.text_score:.3f}" if alpha > 0 else "not evaluated"
    audio_score = f"{hit.audio_score:.3f}" if alpha < 1 else "not evaluated"
    return f"""
    <article class="hit">
      <div class="hit-top">
        <span class="rank">MATCH {rank:02d}</span>
        <span class="speaker {color}">{html.escape(chunk.speaker_id)}</span>
        <span class="timecode">{timestamp(chunk.start_ms)} &rarr; {timestamp(chunk.end_ms)}</span>
        <span class="score">{hit.score:.3f}<small> relevance</small></span>
      </div>
      <p class="quote">{highlight(chunk.transcript, query)}</p>
      <div class="hit-meta">
        <span>Text cosine <b>{text_score}</b></span>
        <span>Audio cosine <b>{audio_score}</b></span>
        <span>Alignment quality <b>{chunk.confidence:.2f}</b></span>
        <span>Speaker overlap <b>{chunk.speaker_confidence:.2f}</b></span>
        <span>{html.escape(chunk.timing)} timing</span>
      </div>
      <div class="provenance">SOURCE SHA-256 {chunk.source_sha256[:16]}...
        &nbsp; / &nbsp; {html.escape(chunk.chunk_id)}
        &nbsp; / &nbsp; {(chunk.end_ms - chunk.start_ms) / 1000:.3f}s clip</div>
    </article>"""


def _empty_results() -> list[Any]:
    return [
        component for _ in range(MAX_RESULTS) for component in (gr.update(visible=False), "", None)
    ]


def _collection_outputs(collection: Collection, elapsed: float) -> list[Any]:
    recording, index = collection.recording, collection.index
    speakers = sorted({chunk.speaker_id for chunk in index.chunks})
    stats = f"""
    <div class="metrics">
      <div><strong>{recording.duration / 60:.1f}<small> min</small></strong>
      <span>Source audio</span></div>
      <div><strong>{len(speakers)}</strong><span>Speaker labels</span></div>
      <div><strong>{len(index.chunks)}</strong><span>Retrievable moments</span></div>
      <div><strong>{elapsed:.2f}<small> s</small></strong><span>Index load / build</span></div>
    </div>"""
    notices = "".join(
        f"<li>{html.escape(notice)}</li>" for notice in recording.notices + index.notices
    )
    status = f"""
    <div class="status-ready"><b>INDEX READY</b> &nbsp; {html.escape(recording.source_name)}
    <p>{html.escape(recording.mode)} &middot; {html.escape(index.text_backend)}
    &middot; {html.escape(index.audio_backend)}</p>
    <details open><summary>Evidence &amp; limitations</summary><ul>{notices}</ul>
    <p>Times address exact source slices; alignment accuracy is not guaranteed to a millisecond.
    Relevance is not a probability. Synthetic alignment quality 1.0 means scripted ground truth.</p>
    </details></div>"""
    table = [
        [
            chunk.speaker_id,
            timestamp(chunk.start_ms),
            timestamp(chunk.end_ms),
            round(chunk.confidence, 3),
            chunk.timing,
            chunk.transcript,
        ]
        for chunk in index.chunks
    ]
    return [
        collection,
        stats,
        status,
        table,
        str(collection.source_path),
        gr.update(choices=["All speakers"] + speakers, value="All speakers"),
        str(collection.manifest_path),
        "<div class='search-summary'>Index ready. Search for a topic, a tone, or both.</div>",
        *_empty_results(),
    ]


def load_demo_callback(current: Collection | None) -> list[Any]:
    """The one-click demo never loads model weights, accesses a token, or downloads data."""
    start = time.perf_counter()
    try:
        collection = create_collection()
    except AudioSleuthError as exc:
        raise gr.Error(str(exc)) from exc
    outputs = _collection_outputs(collection, time.perf_counter() - start)
    dispose(current)
    return outputs


def process_callback(upload: str | None, neural: bool, current: Collection | None) -> list[Any]:
    """Preserve previous results on failure; a missing backend never fabricates uploads."""
    if upload is None:
        raise gr.Error("Choose a WAV or MP3 file, or use the pre-indexed demo.")
    start = time.perf_counter()
    try:
        collection = create_collection(upload=Path(upload), neural=neural)
    except AudioSleuthError as exc:
        raise gr.Error(str(exc)) from exc
    outputs = _collection_outputs(collection, time.perf_counter() - start)
    dispose(current)
    return outputs


def reindex_callback(neural: bool, current: Collection | None) -> list[Any]:
    """Let the demo exercise real neural encoders without running ASR again."""
    if current is None:
        raise gr.Error("Load a recording before re-indexing.")
    start = time.perf_counter()
    try:
        collection = reindex_collection(current, neural=neural)
    except AudioSleuthError as exc:
        raise gr.Error(str(exc)) from exc
    outputs = _collection_outputs(collection, time.perf_counter() - start)
    dispose(current)
    return outputs


def clear_callback(current: Collection | None) -> list[Any]:
    """Clear this session's index and files; Gradio's media cache expires separately."""
    dispose(current)
    return [
        None,
        "<div class='empty-banner'>Workspace cleared. Load a recording to begin.</div>",
        "",
        [],
        None,
        gr.update(choices=["All speakers"], value="All speakers"),
        None,
        "<div class='search-summary'>No active index.</div>",
        *_empty_results(),
        None,
    ]


def search_callback(
    query: str,
    alpha: float,
    speaker: str,
    confidence: float,
    minimum: float,
    collection: Collection | None,
) -> list[Any]:
    """Return actual clip files to Gradio Audio, not a full-source seek suggestion."""
    if collection is None:
        raise gr.Error("Load the demo or index an upload before searching.")
    start = time.perf_counter()
    try:
        hits = search(
            collection.index,
            query,
            alpha=alpha,
            speaker=None if speaker == "All speakers" else speaker,
            min_confidence=confidence,
            min_score=minimum,
            top_k=MAX_RESULTS,
        )
    except AudioSleuthError as exc:
        raise gr.Error(str(exc)) from exc
    took = (time.perf_counter() - start) * 1000
    summary = (
        f"<div class='search-summary'><b>{len(hits)} moments found</b> in {took:.1f} ms"
        f" &nbsp; &middot; &nbsp; Text weight {alpha:.0%} / Audio weight {1 - alpha:.0%}"
        "<br><small>Sorted by hybrid cosine relevance, not a calibrated confidence probability."
        " Clip players start at 0:00 within each exact source interval.</small></div>"
    )
    if not hits:
        summary += (
            "<p>No matches have usable query evidence or meet these filters. "
            "Try another query, speaker, weight, or threshold.</p>"
        )
    if alpha < 1 and collection.index.audio_backend != "CLAP" and not acoustic_query(query).any():
        summary += (
            "<p>Offline audio search found no supported tone words in this query. "
            "Audio similarity is zero; try loud, calm, excited, or use text weight 1.</p>"
        )
    outputs: list[Any] = [summary]
    for position in range(MAX_RESULTS):
        if position < len(hits):
            hit = hits[position]
            outputs.extend(
                (
                    gr.update(visible=True),
                    hit_card(hit, query, position + 1, alpha=alpha),
                    hit.chunk.audio_path,
                )
            )
        else:
            outputs.extend((gr.update(visible=False), "", None))
    return outputs


def build_app() -> gr.Blocks:
    """Build two tabs with per-browser state and bounded model concurrency."""
    css = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    theme = gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=["Inter", "Segoe UI", "Arial", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    )
    with gr.Blocks(
        title="AudioSleuth | Find the moment. Hear the evidence.",
        theme=theme,
        css=css,
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as dashboard:
        state = gr.State(value=None, time_to_live=3600, delete_callback=dispose)
        gr.HTML("""
        <header class="hero">
          <div class="brand"><span class="brand-mark">a<span>s</span></span> AudioSleuth
            <span class="edition">RESEARCH STUDIO</span></div>
          <div class="eyebrow">MULTIMODAL AUDIO INTELLIGENCE</div>
          <h1>Find the moment.<br><em>Hear the evidence.</em></h1>
          <p>Search beyond the transcript. Follow the speaker, the tone, and the exact
          audio behind every answer.</p>
          <div class="hero-tags"><span>Speaker-aware</span><span>Dual-vector retrieval</span>
          <span>Source-linked playback</span>
          <span class="local">Local-first / offline demo</span></div>
          <div class="signal" aria-hidden="true">
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
        </header>""")
        metrics = gr.HTML(
            "<div class='empty-banner'>Your listening workspace is ready. "
            "Load the bundled podcast to start instantly, without a GPU or account.</div>"
        )
        with gr.Tabs():
            with gr.Tab("01  Ingest & index", id="ingest"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5, min_width=300):
                        gr.Markdown(
                            "### Bring a conversation\nWAV or MP3 · up to 15 minutes · 100 MiB"
                        )
                        upload = gr.File(
                            label="Podcast audio", file_types=[".wav", ".mp3"], type="filepath"
                        )
                        neural = gr.Checkbox(
                            label="Use neural text + CLAP embeddings (optional ML installation)",
                            value=False,
                        )
                        process = gr.Button("Process & Index", variant="primary", size="lg")
                        gr.Markdown(
                            "Uploads require WhisperX. "
                            "Pyannote needs `HF_TOKEN` and model consent. "
                            "Missing diarization leaves **UNKNOWN** speakers, "
                            "not guessed identities."
                        )
                    with gr.Column(scale=4, min_width=300):
                        gr.HTML("""
                        <section class="demo-panel"><div class="eyebrow">NO SETUP REQUIRED</div>
                        <h2>Market Signals</h2>
                        <p>A two-voice synthetic podcast about interest rates,
                        startup valuations, and clean energy.
                        Real playable speech, bundled timestamps,
                        and precomputed vectors.</p>
                        <div class="demo-pills"><span>2 voices</span><span>Original script</span>
                        <span>Offline-ready</span></div>
                        <p class="fine">Synthetic speech + scripted speaker labels.
                        Offline search uses TF-IDF and signal descriptors,
                        not neural emotion detection.</p>
                        </section>""")
                        demo = gr.Button(
                            "Load Pre-indexed Demo Audio", size="lg", variant="secondary"
                        )
                with gr.Row():
                    reindex = gr.Button("Re-index current recording", variant="secondary")
                    clear = gr.Button("Clear workspace")
                gr.Markdown(
                    "Re-index uses the embedding checkbox above and keeps existing words/speakers. "
                    "The one-click demo always starts offline. "
                    "Clear removes the active collection; "
                    "cached media expires separately."
                )
                with gr.Accordion("Local backend readiness", open=False):
                    readiness = gr.Markdown(backend_status())
                    refresh = gr.Button("Refresh backend status", size="sm")
                    refresh.click(backend_status, outputs=[readiness], api_name="backend_status")
                status = gr.HTML("")
                source = gr.Audio(
                    label="Source recording / waveform", interactive=False, type="filepath"
                )
                with gr.Accordion("Diarized speaker timeline & provenance", open=True):
                    timeline = gr.Dataframe(
                        headers=["Speaker", "Start", "End", "Alignment", "Timing", "Transcript"],
                        datatype=["str", "str", "str", "number", "str", "str"],
                        interactive=False,
                        wrap=True,
                        label="Indexed speaker turns",
                    )
                    manifest = gr.File(label="Download provenance manifest", interactive=False)
            with gr.Tab("02  Search & listen", id="search"):
                gr.Markdown(
                    "### Every result comes with receipts\n"
                    "Find what was said, how it sounded, and who said it."
                )
                with gr.Row():
                    query = gr.Textbox(
                        label="Search the conversation",
                        scale=6,
                        placeholder="heated argument about interest rates",
                    )
                    submit = gr.Button("Find moments", variant="primary", scale=1)
                gr.Examples(
                    examples=[
                        ["heated argument about interest rates"],
                        ["speaker laughing about valuation"],
                        ["calm explanation of monetary policy"],
                        ["renewable energy and battery storage"],
                    ],
                    inputs=query,
                )
                with gr.Row():
                    alpha = gr.Slider(
                        0,
                        1,
                        value=0.65,
                        step=0.05,
                        label="Semantic weight alpha",
                        info="0 = audio / tone only · 1 = transcript only",
                    )
                    speaker = gr.Dropdown(
                        choices=["All speakers"], value="All speakers", label="Speaker"
                    )
                with gr.Accordion("Evidence filters & scoring", open=False):
                    with gr.Row():
                        confidence = gr.Slider(
                            0, 1, value=0, step=0.05, label="Minimum alignment quality"
                        )
                        minimum = gr.Slider(
                            -1, 1, value=0, step=0.05, label="Minimum hybrid relevance"
                        )
                    gr.Markdown(
                        "`score = alpha * cosine(text) + (1-alpha) * cosine(audio)`\n\n"
                        "Cosines are not probabilities. Offline tone mapping is a heuristic: "
                        "it cannot verify laughter, anger, identity, or psychological state."
                    )
                summary = gr.HTML(
                    "<div class='empty-banner'>Load an index in the first tab to begin.</div>"
                )
                result_outputs: list[Any] = []
                for position in range(MAX_RESULTS):
                    with gr.Group(visible=False, elem_classes=["result-card"]) as group:
                        card = gr.HTML("")
                        player = gr.Audio(
                            label=f"Match {position + 1} / exact audio excerpt",
                            interactive=False,
                            type="filepath",
                            waveform_options=gr.WaveformOptions(
                                waveform_color="#8296a7",
                                waveform_progress_color="#16a394",
                            ),
                        )
                    result_outputs.extend((group, card, player))
        gr.HTML("""
        <footer>AudioSleuth &nbsp; / &nbsp; Evidence before inference.
        <span>Local research demo. Synthetic fixtures are labeled. No analytics.</span></footer>""")
        index_outputs = [
            state,
            metrics,
            status,
            timeline,
            source,
            speaker,
            manifest,
            summary,
            *result_outputs,
        ]
        demo.click(
            load_demo_callback,
            inputs=[state],
            outputs=index_outputs,
            api_name="load_demo",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        process.click(
            process_callback,
            inputs=[upload, neural, state],
            outputs=index_outputs,
            api_name="process_audio",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        reindex.click(
            reindex_callback,
            inputs=[neural, state],
            outputs=index_outputs,
            api_name="reindex",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        clear.click(
            clear_callback,
            inputs=[state],
            outputs=index_outputs + [upload],
            api_name="clear_workspace",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        search_inputs = [query, alpha, speaker, confidence, minimum, state]
        submit.click(
            search_callback,
            inputs=search_inputs,
            outputs=[summary, *result_outputs],
            api_name="search",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        query.submit(
            search_callback,
            inputs=search_inputs,
            outputs=[summary, *result_outputs],
            api_name=False,
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
    return dashboard


def main() -> None:
    """Bind only loopback by default; never create a public sharing tunnel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_app().queue(max_size=16, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        max_file_size="100mb",
        show_error=False,
    )


if __name__ == "__main__":
    main()
