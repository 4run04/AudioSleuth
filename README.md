# AudioSleuth

**Zero-Shot Multi-Speaker Podcast Retrieval & Audio Provenance System**

A complete local demonstration with original, bundled two-voice speech,
speaker-aware chunking, a NumPy dual-vector index, hybrid retrieval, and
waveform playback of exact source slices.

**Start here:** launch [app.py](app.py), click **Load Pre-indexed Demo Audio**,
then open **Search & listen**. No GPU, token, model download, FFmpeg, or online
speech service is required for this path.

## What is implemented, and what is actually being demonstrated?

| Stage | Optional neural path | Always-available offline demonstration |
|---|---|---|
| ASR and alignment | Local WhisperX ASR and its language-specific forced aligner | Bundled original speech with TTS word-boundary events |
| Diarization | WhisperX's pyannote pipeline, with an authorized Hugging Face token | Two scripted, explicitly labeled speaker tracks |
| Dynamic chunks | Speaker changes, silence, punctuation, lexical topic shifts, 120-second cap | The same chunking code |
| Text vectors | `sentence-transformers/all-MiniLM-L6-v2`, on CPU | Word/bigram TF-IDF vectors |
| Audio vectors | LAION-CLAP weights via Transformers, 48 kHz windows pooled across the entire chunk | Nine measured signal descriptors |
| Query-to-audio comparison | CLAP's text encoder in the shared text/audio space | Explicit tone-word-to-descriptor mapping |
| Index | Session-local, normalized NumPy vector matrices | Same; bundled precomputed vectors load immediately |
| Provenance player | Timestamped source WAV excerpts with waveform controls | Same |

**The offline mode is not zero-shot neural semantic search or an emotion
classifier.** Its TF-IDF and acoustic descriptor backends are clearly shown
in the UI. For example, "laughing" is a query cue for brightness/dynamics in
the fallback, not an acoustic verification that someone laughed. The demo
contains synthesized "ha ha" speech, not recorded natural human laughter.
CLAP is also a general audio/text model, not a calibrated psychological
state detector.

The optional model adapters are implemented against the upstream APIs linked
below. Real MiniLM, LAION-CLAP, and WhisperX CPU inference have now been
exercised in the separate `.venv-ml` environment, including full neural
indexing of the bundled recording. Authenticated speaker diarization remains
unverified because no `HF_TOKEN` with model access is configured. The
lightweight `.venv` remains usable independently.

This is a **working, production-oriented local demonstration**, not a claim
of audited production readiness, benchmarked recognition accuracy, or a
multi-tenant hosted service. Deployment requirements are listed below.

## Quick start

### Windows PowerShell

From this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:7860**.

The delivered environment already exists, so on this machine:

```powershell
cd C:\dev\AudioSleuth
.\.venv\Scripts\python.exe app.py
```

If the app is already running, open its URL instead of starting a second copy.
Use `python app.py --port 7861` to intentionally run another instance.
Stop a foreground server with Ctrl+C.

The prepared neural environment can run the same dashboard with the actual
ML backends installed:

```powershell
.\.venv-ml\Scripts\python.exe app.py
```

Use the neural-embedding checkbox followed by **Re-index current recording**
to test real MiniLM + CLAP on the bundled recording. The initial
**Load Pre-indexed Demo Audio** button intentionally stays instantaneous and
offline in either environment.

### Linux / macOS / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

The source targets Python **3.10+**; pip selects compatible dependency releases.
Python **3.11** is recommended for optional speech-model compatibility.
The offline application was tested on Windows with Python 3.11.15 and Gradio
5.50.0. Linux/macOS/WSL are supported by the portable code paths but were not
separately tested here. Gradio is constrained to 5.x because 6.x changes
several UI APIs.

The first dependency installation needs internet access. The bundled demo
needs no network afterward. The app disables Gradio analytics, uses system
fonts, binds only to loopback, and never creates a public Gradio sharing tunnel.

## Presentation walkthrough

1. In **Ingest & index**, click **Load Pre-indexed Demo Audio**.
2. See the two speaker labels, 13 chunks, source waveform, and timeline.
   Loading the index took approximately **0.20 seconds** in a browser run;
   timings depend on hardware and cache state.
3. Open **Search & listen**.
4. Search **heated argument about interest rates**.
5. The top example result is `SPEAKER_01`, from **00:15.545 to 00:26.420**.
   Its actual playable excerpt is **10.875 seconds**, not the whole podcast.
6. Move `alpha` toward 1 for transcript relevance or toward 0 for audio cues.
   Click **Find moments** to apply updated weights and filters.
7. Filter `SPEAKER_00` to retrieve only that speaker's chunks.
8. Try:
   - `speaker laughing about valuation`
   - `calm explanation of monetary policy`
   - `renewable energy and battery storage`
9. Download the provenance manifest from the first tab.

Use **Local backend readiness** to inspect installed prerequisites without
downloading weights or displaying tokens. **Re-index current recording** uses
the embedding checkbox and retains the existing words and speaker labels;
this allows the bundled recording to exercise real neural embeddings without
rerunning ASR. **Clear workspace** removes the active index and its owned
source/snippet files, resets results, and clears the upload field. Gradio's
separate media cache remains subject to its expiry policy.

Results display speaker, transcript with escaped query highlights, start/end
timestamps, combined relevance, component cosines, alignment quality,
speaker-overlap evidence, timing method, and source SHA-256.

The recording is **126.055 seconds** of original scripted synthetic speech,
with **277 timed words**, generated locally using two Windows system voices.
It is not a copyrighted podcast excerpt, a voice clone, or a real interview.
Runtime playback works on any supported platform; Windows is needed only to
regenerate these particular bundled TTS assets.

## Architecture and file layout

```text
AudioSleuth/
|-- app.py                    # Gradio dashboard, callbacks, safe highlights
|-- ingestion.py              # WAV/MP3 decoding, WhisperX, speaker assignment
|-- chunking.py               # Dynamic turns and exact audio snippets
|-- indexing.py               # Neural/fallback encoders and NumPy dual index
|-- retrieval.py              # Hybrid scoring and metadata filters
|-- service.py                # Atomic session-local collection lifecycle
|-- schemas.py                # Typed words, intervals, chunks, search hits
|-- requirements.txt          # Lightweight runtime
|-- requirements-ml.txt       # Optional neural stack
|-- requirements-dev.txt      # Tests and static checks
|-- pyproject.toml
|-- static/
|   `-- style.css
|-- assets/
|   |-- script.json           # Original script and voice settings
|   |-- demo.wav              # Bundled playable two-speaker audio
|   |-- demo.json             # Word timings, intervals, source hash
|   |-- demo_index.json       # Payloads, vocabulary, index version
|   `-- demo_index.npz        # Precomputed vectors and IDF; no pickle
|-- tools/
|   |-- synthesize_demo.ps1   # Optional offline Windows TTS regeneration
|   |-- build_demo.py         # Audio/manifest/index assembly
|   `-- check_backends.py     # Real-model smoke tests, never fallback successes
`-- tests/
    |-- conftest.py
    |-- test_ingestion.py
    |-- test_chunking.py
    |-- test_retrieval.py
    |-- test_app.py
    `-- test_recheck.py
```

### Data flow

```text
WAV / MP3
   |
bounded decode -> 16 kHz mono -> original-source SHA-256
   |
WhisperX ASR -> forced alignment -> words with timing provenance
   |                                     |
pyannote speaker intervals --------------> maximum temporal-overlap assignment
                                         |
                             speaker-stratified dynamic chunks
                                         |
                +------------------------+------------------------+
                |                                                 |
        transcript embeddings                               audio embeddings
                |                                                 |
                +---------------- NumPy dual index ---------------+
                                         |
                             metadata filters + hybrid cosine
                                         |
                              ranked timestamped excerpts
                                         |
                             waveform player + JSON provenance
```

The demo enters the same pipeline after ASR/diarization, using verified
bundled fixtures, and loads precomputed text/audio vectors.

## Pipeline details

### 1. Audio loading and word-level speaker assignment

- WAV and MP3 are decoded with `soundfile`; FFmpeg is a fallback for codecs
  unsupported by the installed libsndfile.
- Downmix channels and polyphase-resample to 16 kHz mono.
- Reject files over 100 MiB, recordings over 15 minutes, empty/very short audio,
  nonfinite samples, and unsupported file extensions.
- CPU WhisperX uses `int8`; CUDA uses `float16`. Model name defaults to `small`
  and can be changed with `AUDIOSLEUTH_ASR_MODEL`.
- Forced alignment provides word timestamps. Tokens that cannot be aligned
  are explicitly interpolated where possible, marked **estimated**, and
  assigned alignment quality zero. Impossible timings produce an error.

For word interval \(W=[s_w,e_w)\), choose the speaker with greatest overlap:

\[
\hat{p}(w)=\arg\max_p
\left|W\cap\bigcup_{i:\,p_i=p}[s_i,e_i)\right|.
\]

Unioning intervals prevents duplicate tracks from inflating overlap.
Ties are resolved deterministically by speaker label. Nearest-turn assignment
is allowed only within 250 ms and has **zero overlap confidence**.
Otherwise the word is `UNKNOWN`. Simultaneous speech is reduced to one label;
this system does not separate overlapping voices or guarantee that a snippet
contains only one audible speaker.

### 2. Speaker-stratified dynamic chunking

- Contiguous same-speaker words form turns.
- Speaker changes and silences longer than 1.2 seconds start new turns.
- After at least eight seconds, punctuation with a 350 ms pause or a strong
  lexical-cohesion change can start a new chunk.
- The topic-shift test uses adjacent transcript windows and TF-IDF similarity.
  It is a transparent lexical heuristic, not an unvalidated neural semantic
  change-point detector.
- Long monologues split preferentially at punctuation, otherwise at a word
  boundary. A hard cap of **120,000 milliseconds** is enforced.
- No fixed token-count slicing is used. All words are preserved in order.

### 3. Dual-vector indexing

**Neural text:** sentence-transformers MiniLM produces normalized vectors.
Long transcripts are split at the tokenizer's actual context capacity.
Every token window is embedded and token-count-weighted before normalization,
so the end of a 120-second monologue is not silently discarded by MiniLM.

**Neural audio:** Transformers loads `laion/clap-htsat-unfused`, the actual
LAION model weights. CLAP consumes 48 kHz waveform windows. Every window
of a long chunk is embedded; normalized window vectors are mean-pooled and
renormalized. This avoids silently indexing only the first ten seconds.
The query is encoded with the **same CLAP model's text encoder**.

**Offline text:** normalized unigram/bigram TF-IDF with a bounded vocabulary.
This is lexical matching; it cannot promise synonym or paraphrase retrieval.

**Offline audio:** measured loudness, quietness, brightness, darkness,
modulation/dynamics, steadiness, noisiness, tonality, and silence. Query
tone words map explicitly to those axes. If there are no supported tone
words, the acoustic query vector is zero and the UI explains this.

NumPy matrices are an exact cosine index, not approximate nearest-neighbor
search. This is appropriate for bounded, single-recording collections.
The index stores row-aligned payloads and validates the bundled source hash,
chunk order, text, boundaries, and dimensions before loading precomputed vectors.
No untrusted pickle is loaded.

Payload example:

```json
{
  "chunk_id": "bf94c59f951d-0002",
  "speaker_id": "SPEAKER_01",
  "start_ms": 15545,
  "end_ms": 26420,
  "start_sec": 15.545,
  "end_sec": 26.42,
  "transcript": "I completely disagree about interest rates! ...",
  "confidence": 1.0,
  "speaker_confidence": 1.0,
  "timing": "synthetic",
  "audio_path": "bf94c59f951d-0002.wav",
  "source_sha256": "the full 64-character hash is included in the actual manifest"
}
```

The example above abbreviates only the documentation display; actual payloads
contain the full transcript and full hash. Runtime payloads point to real
clip files. The exported manifest uses basenames rather than leaking server
filesystem paths.

### 4. Retrieval and score meaning

All vectors are L2-normalized:

\[
\mathrm{score}(q,c)=
\alpha\,(t_q^\top t_c)+(1-\alpha)\,(a_q^\top a_c).
\]

- `alpha=1`: transcript-only ranking.
- `alpha=0`: acoustic-only ranking.
- A zero-weight modality is not invoked. Text-only search does not require a
  working acoustic encoder, and audio-only search does not require MiniLM.
- Queries with no usable evidence in either selected modality return no
  matches instead of arbitrary all-zero ranked chunks.
- Speaker and minimum **alignment quality** filters apply before top-k.
- Minimum **hybrid relevance** is a separate filter.
- Cosines can be negative; the implementation does not pretend that they
  are probabilities or silently remap them to percentages.
- Query embeddings never silently switch models after indexing. If the
  original encoder becomes unavailable, the app asks for re-indexing.
- Ties are stable by source start time.

### 5. What "exact millisecond provenance" means

At 16 kHz, one millisecond contains exactly 16 samples. The snippet saved for
`[start_ms, end_ms)` contains:

```python
recording.samples[start_ms * 16 : end_ms * 16]
```

Start times round outward down; end times round outward up, bounded to the
last full millisecond of available audio. Snippets use float WAV so tests
can verify **sample-for-sample equality** against the normalized source.
The UI plays this actual excerpt rather than seeking approximately into
the whole source.

This guarantees precise indexing/playback coordinates, **not** millisecond
accuracy of recognition or forced alignment. Demo word starts come from
TTS boundary events; word ends use the next event and include inter-word
pauses. MP3 decoding/encoder delay and resampling conventions also affect
the relationship to another player's timeline.

A SHA-256 identifies source bytes and detects accidental fixture mismatch.
It is not a digital signature, authenticity certificate, watermark, or
proof that a recording has never been edited.

## Optional real-audio / neural setup

Use a separate environment if you need to preserve a lightweight presentation
installation. Heavy ML dependencies have tighter Python/platform constraints.

```bash
python3.11 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
```

The optional stack uses **Transformers CLAP**, not the legacy `laion-clap`
Python package. This resolves a real dependency conflict: WhisperX 3.8.6
requires NumPy 2.1+, whereas `laion-clap` 1.1.7 requires NumPy below 2.
Older WhisperX 3.7 also requires NumPy 2. Do not install the legacy CLAP
package into this environment. Transformers is constrained to its compatible
4.x API.

Install FFmpeg for formats that libsndfile cannot decode and for optional
file-based decoder integrations, using your platform package manager:

```bash
# Debian / Ubuntu / WSL
sudo apt-get update
sudo apt-get install ffmpeg

# macOS with Homebrew
brew install ffmpeg
```

On Windows, install FFmpeg and ensure `ffmpeg` is on PATH. For CUDA, install
the PyTorch/CUDA stack recommended by WhisperX and PyTorch for your platform;
the app does not assume that a CUDA runtime is present.

For diarization:

1. Accept the model's conditions at
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Create an appropriately scoped Hugging Face read token.
3. Set it in the process environment; never commit it:

```bash
export HF_TOKEN="your-read-token"
python app.py
```

```powershell
$env:HF_TOKEN = "your-read-token"
python app.py
```

Upload a WAV/MP3 and click **Process & Index**. Check the neural-embedding
checkbox if you want MiniLM and CLAP instead of the offline encoders.
Initial model downloads and transcription can take minutes.

To test the actual adapters rather than the fallback path:

```bash
python tools/check_backends.py --backend text
python tools/check_backends.py --backend audio
python tools/check_backends.py --backend asr --asr-model tiny.en
```

These commands may download weights and return nonzero on failure. The ASR
check transcribes a short excerpt of the original synthetic speech; it does
not read the bundled transcript as a substitute. Authenticated diarization
still needs your token and acceptance of the model's access conditions.

All actual transcription/embedding inference happens locally. Model
downloads contact the respective model hosts; the program does not send
your uploaded audio to a hosted transcription API.

### Explicit fallback behavior

| Condition | Behavior |
|---|---|
| No CUDA | WhisperX attempts local CPU int8 inference |
| WhisperX absent or alignment fails | Visible error; upload is not indexed; demo button remains available |
| Missing token / diarization failure | Real transcript retained; speaker labels are UNKNOWN with a notice |
| MiniLM unavailable | Visible TF-IDF fallback, with matching lexical query encoder |
| CLAP unavailable | Visible signal-descriptor fallback, with matching tone-query mapper |
| No recognized offline tone words | Audio similarity is zero; UI explains how to change the query/weight |
| Invalid audio or input | Actionable error, no fabricated output |
| Failed replacement upload | Existing collection remains available |
| Invalid/mismatched bundled assets | Explicit integrity error, not an empty success |

The app deliberately does **not** attach the sample podcast transcript to
an unrelated uploaded file. Such a fallback would fabricate provenance.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m mypy
python -m ruff check .
python -m ruff format --check .
```

The delivered test suite exercises:

- Stereo resampling and WAV/MP3 loading.
- Maximum speaker overlap, deterministic ties, duplicated intervals, and unknown speakers.
- Invalid/nonfinite timestamps and visibly estimated word times.
- Speaker changes, silences, long-monologue caps, and preservation of every word.
- Exact equality between all demo snippet samples and their source slices.
- Actual demo retrieval for interest rates, valuation, and battery storage.
- The hybrid-score formula at alpha endpoints and intermediate weights.
- Speaker/confidence filtering, empty results, and invalid queries.
- Explicit missing-model fallback without fabricated transcript data.
- Whole-chunk CLAP window coverage with a controlled adapter fixture.
- Pickle-free precomputed-index round trips and provenance tamper detection.
- Dashboard creation, escaped highlights, independent collections, and scoped cleanup.
- Missing word scores, invalid aligner timestamps, and partial timing anchors.
- Zero-weight encoder independence and unsupported-query empty results.
- Full-token neural text pooling and the Transformers CLAP adapter contract.
- Corrupt IDF rejection, re-indexing, and explicit workspace clearing.

Verified result: **55 tests passed**, mypy passed for all seven runtime Python
modules, and Ruff lint/format checks passed. Pylance reported no syntax errors
in the dashboard entry point. WAV and MP3 fixtures both ran on the test machine.

Browser verification additionally exercised demo loading, search, speaker
filtering, and successful playback of the top 10.875-second excerpt.
HTTP client verification downloaded that exact-duration clip and confirmed
that a second session could not search the first session's index.
Updated HTTP checks also exercised re-indexing, clearing, and empty-evidence queries.

### Real neural integration results

The separate `.venv-ml` installed successfully and `pip check` reported no
broken requirements. Tested versions: WhisperX 3.8.6, Transformers 4.57.6,
sentence-transformers 5.7.0, PyTorch 2.8.0, pyannote.audio 4.0.7, NumPy 2.4.6.

| Check | Observed result |
|---|---|
| MiniLM real inference, including a long transcript | Normalized `(2, 384)` output; passed |
| LAION-CLAP real audio/text inference | Matching 512-dimensional vectors; passed |
| WhisperX `tiny.en` + forced alignment on CPU | 25 words from a 12-second speech excerpt; passed |
| Real upload through ASR, chunking, indexing, and retrieval | One searchable chunk, with UNKNOWN speaker and an explicit missing-token notice; passed |
| Entire demo indexed without either fallback | Text `(13, 384)`, audio `(13, 512)`; about 29.95 seconds |
| Neural interest-rate query | First result is the disagreement at 15.545 seconds |
| Neural renewable-energy query | First result includes battery storage at 75.434 seconds |
| Authenticated speaker diarization | Not run: token/model authorization required |

The live dashboard was also switched to the neural environment, re-indexed
through its UI with MiniLM + CLAP, and successfully played the top neural
result's exact 10.875-second clip.

These are integration checks, not accuracy benchmarks. In the real neural
run, the laughing/valuation query ranked a serious valuation discussion
first at the default weight. This is a genuine limitation of the model/demo,
not hidden or corrected by injecting transcript labels into audio vectors.
The offline lexical test's expected ranking is not evidence of CLAP's
laughter-recognition accuracy.

On this Windows machine, pyannote emits a TorchCodec/FFmpeg warning. The
WhisperX smoke test nevertheless completed because AudioSleuth supplies
decoded in-memory audio. Do not infer from that success that arbitrary
TorchCodec file decoding works; install an appropriate FFmpeg shared-library
runtime if using that separate path.

Gradio 5.x emits deprecation warnings about its upcoming 6.x APIs. These
warnings are not hidden and do not prevent the pinned-major application
from running.

## Session data, privacy, and deployment limits

- Each browser session has its own collection and index. Model objects may
  be reused, but transcripts and result matrices are not a global corpus.
- A bounded queue serializes ingestion/search callbacks; large model work
  does not spawn unlimited concurrent jobs.
- Generated source/snippet/manifest files live in uniquely named collection
  folders under the OS temporary directory.
- Successful replacement deletes only that session's previous collection.
  Session-state expiry is one hour. Gradio media caches have a one-hour
  age-based cleanup policy.
- Abrupt process termination can leave temporary files. These settings are
  not a legal retention guarantee, secure erasure mechanism, or substitute
  for operational storage policies.
- The default server is loopback-only and unauthenticated for local use.
  A hostile process or administrator on the same machine is outside the
  privacy boundary. Media URLs are not a substitute for authorization.
- Query/transcript HTML is escaped; tokens are read from the environment,
  and backend failure logs avoid printing token values.
- Hugging Face telemetry and pyannote metrics default to disabled in the
  neural inference paths, in addition to Gradio analytics being disabled.

Before hosting for real users, add authenticated per-user storage and media
authorization, TLS, quotas, a sandboxed media decoder, worker isolation,
job cancellation, robust persistence and deletion policies, observability,
dependency/security maintenance, and load testing. Evaluate WER, diarization
error rate, alignment error, retrieval nDCG/Recall@k, and acoustic-retrieval
accuracy on consented representative recordings. Do not deploy this local
demo publicly merely by changing the bind address.

## Regenerating the bundled demonstration

This is optional; the finished WAV, manifest, and index are already included.
On Windows with the David and Zira system voices:

```powershell
powershell.exe -NoProfile -File tools\synthesize_demo.ps1
.\.venv\Scripts\python.exe tools\build_demo.py
```

The script collects real TTS word-start events and assembles original speech
turns, silence, controlled gain, source hashes, and precomputed vectors.
Changing the script changes the source hash and requires rebuilding the index.
Intermediate turn files are written to `assets\_generated`; they are not
needed for runtime playback.

## Verified upstream API references

- [WhisperX usage and forced alignment](https://github.com/m-bain/whisperX#python-usage)
- [WhisperX pyannote wrapper](https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py)
- [Transformers CLAP API and LAION weights](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/clap)
- [Sentence-transformers](https://www.sbert.net/)
- [Gradio Audio](https://www.gradio.app/docs/gradio/audio)

Model weights and upstream libraries retain their own licenses and access
conditions. No third-party model weights or real podcast recordings are
redistributed in this project.
