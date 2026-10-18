"""SceneFinder — AI YouTube Episode & Scene Finder (Streamlit edition).

Paste a YouTube playlist URL -> searchable episode database.
Describe a scene or quote dialogue -> episode, timestamp, clickable link.
Runs free on Streamlit Community Cloud (no credit card needed).
"""
import sys
import time
from pathlib import Path

# repo root (parent of streamlit/) holds the reusable backend package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
# tiny Whisper fits the free 1GB container for bulk playlist backfill;
# uploads use "base" for better quality (see _whisper_model below).
os.environ.setdefault("WHISPER_MODEL", "tiny")
os.environ.setdefault("SEEDS_PATH",
                      str(Path(__file__).resolve().parent / "seeds.json"))

import streamlit as st

from backend import db, ingest
from backend import search as search_mod
from backend import embeddings as emb_mod

# ---------------------------------------------------------------- seed transcripts
# Episodes whose YouTube captions are missing/broken get a human-verified
# transcript bundled with the app, so they stay searchable forever.
SEEDS_PATH = Path(__file__).resolve().parent / "seeds.json"

def _load_seeds():
    try:
        import json
        if SEEDS_PATH.exists():
            return json.loads(SEEDS_PATH.read_text())
    except Exception as e:
        print(f"[seeds] load failed: {e}")
    return {}

def _seeds_hash():
    import hashlib
    try:
        return hashlib.md5(SEEDS_PATH.read_bytes()).hexdigest()[:12]
    except Exception:
        return "none"

def _marker_path(pid):
    from backend.config import DATA_DIR
    return DATA_DIR / f".seeds_applied_{pid}.json"

def _read_marker(pid):
    import json
    try:
        p = _marker_path(pid)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {}

def _write_marker(pid, h, seeded_yids):
    import json
    try:
        _marker_path(pid).write_text(json.dumps({"hash": h, "videos": sorted(seeded_yids)}))
    except Exception as e:
        print(f"[seeds] marker write failed: {e}")

def _embed_segments_for_videos(pid, video_ids):
    """Embed only the segments of the given videos (used after seeding)."""
    if not emb_mod.available():
        return
    batch, seg_ids = [], []
    for v in db.get_videos(pid):
        if v["id"] not in video_ids:
            continue
        for s in db.get_segments(v["id"]):
            if len(s["text"]) < 3:
                continue
            seg_ids.append(s["id"])
            batch.append(s["text"][:500])
    import numpy as np
    B = 64
    for i in range(0, len(batch), B):
        try:
            vecs = emb_mod.embed(batch[i:i + B])
        except Exception as e:
            print(f"[seeds] embedding batch failed: {e}")
            break
        pairs = [(seg_ids[i + j], int(vecs.shape[1]),
                  vecs[j].astype(np.float32).tobytes())
                 for j in range(vecs.shape[0])]
        db.save_embeddings(pairs)

def apply_seed_transcripts(pid):
    """Fill in transcripts for videos that have none, from bundled seeds.
    Version-aware: when seeds.json changes, previously-seeded videos get
    re-seeded (never clobbers real YouTube captions). Returns count applied."""
    seeds = _load_seeds()
    if not seeds:
        return 0
    h = _seeds_hash()
    marker = _read_marker(pid)
    if marker.get("hash") == h:
        return 0  # already up to date
    prev_seeded = set(marker.get("videos") or [])
    # Migration: the first seed rollout wrote no marker. Detect videos that
    # were seeded by it (their segments contain seed text) so updates apply.
    seed_texts = {str(s[2]) for segs in seeds.values() for s in segs}
    if not marker:
        for v in db.get_videos(pid):
            yid = v.get("youtube_id")
            if yid in seeds and v.get("has_transcript"):
                try:
                    cur = {s["text"] for s in db.get_segments(v["id"])}
                except Exception:
                    cur = set()
                if cur & seed_texts:
                    prev_seeded.add(yid)
    seeded, seeded_yids = [], set()
    for v in db.get_videos(pid):
        yid = v.get("youtube_id")
        if yid not in seeds:
            continue
        # seed only videos without real captions, or ones we seeded before
        if v.get("has_transcript") and yid not in prev_seeded:
            continue
        segs = [(float(s[0]), float(s[1]), str(s[2])) for s in seeds[yid]]
        db.add_segments(v["id"], segs)  # replaces existing segments
        db.mark_transcript(v["id"], True)
        seeded.append(v["id"])
        seeded_yids.add(yid)
        print(f"[seeds] seeded transcript for {yid} ({len(segs)} segments)")
    if seeded:
        _embed_segments_for_videos(pid, set(seeded))
        search_mod.invalidate_cache(pid)
    _write_marker(pid, h, seeded_yids | (prev_seeded & set(seeds)))
    return len(seeded)

# ---------------------------------------------------------------- video uploads
# Users can upload any video file -> transcribed on-device -> searchable,
# jumping to the exact second with the built-in player.
def _uploads_dir():
    from backend.config import DATA_DIR
    d = DATA_DIR / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _upload_index_path():
    return _uploads_dir() / "index.json"

def _upload_video_path(youtube_id):
    """Return the local file path for an uploaded video, or None."""
    import json
    try:
        idx = json.loads(_upload_index_path().read_text())
        p = Path(idx.get(youtube_id, ""))
        if p.exists():
            return p
    except Exception:
        pass
    return None

def _register_upload(youtube_id, path):
    import json
    ip = _upload_index_path()
    try:
        idx = json.loads(ip.read_text()) if ip.exists() else {}
    except Exception:
        idx = {}
    idx[youtube_id] = str(path)
    ip.write_text(json.dumps(idx))

@st.cache_resource
def _whisper_model():
    from faster_whisper import WhisperModel
    # "base" fits the free 1GB container; "small" is better if RAM allows.
    return WhisperModel("base", device="cpu", compute_type="int8")


def _transcribe_upload(vpath):
    """Transcribe an uploaded file, working around the faster-whisper/PyAV
    metadata_errors incompatibility (old faster-whisper + new PyAV)."""
    model = _whisper_model()
    try:
        segments, _info = model.transcribe(str(vpath), beam_size=5)
    except TypeError:
        import av
        _orig = av.open

        def _patched(*a, **k):
            k.pop("metadata_errors", None)
            return _orig(*a, **k)
        av.open = _patched
        try:
            segments, _info = model.transcribe(str(vpath), beam_size=5)
        finally:
            av.open = _orig
    return [(round(s.start, 1), round(s.end, 1), s.text.strip())
            for s in segments if s.text.strip()]

def _is_upload_video(r):
    return str(r.get("youtube_id") or "").startswith("upload-")

st.set_page_config(
    page_title="SceneFinder — AI YouTube Scene Finder",
    page_icon="🎬",
    layout="centered",
)

# ---------------------------------------------------------------- setup
db.init_db()

st.title("🎬 SceneFinder")
st.caption("AI YouTube Episode & Scene Finder — paste a playlist, find any scene.")

# ---------------------------------------------------------------- session state
if "active_pid" not in st.session_state:
    st.session_state.active_pid = None
if "ingest_jid" not in st.session_state:
    st.session_state.ingest_jid = None

# ---------------------------------------------------------------- playlist input
with st.expander("➕ Analyze a new playlist", expanded=st.session_state.active_pid is None):
    url = st.text_input(
        "YouTube playlist URL",
        placeholder="https://www.youtube.com/playlist?list=...",
        key="playlist_url",
    )
    if st.button("🔍 Analyze playlist", type="primary", disabled=not (url or "").strip()):
        with st.spinner("Starting analysis…"):
            pid = db.create_playlist(url.strip())
            jid = ingest.start_playlist_job(pid, url.strip())
            st.session_state.active_pid = pid
            st.session_state.ingest_jid = jid
        st.rerun()

# ---------------------------------------------------------------- ingest progress
def poll_job(jid):
    """Render live progress for a running ingest job. Returns the job dict."""
    bar = st.progress(0.0, text="Starting…")
    status_box = st.empty()
    while True:
        job = db.get_job(jid)
        if not job:
            break
        stage, cur, tot = job.get("stage") or "", job.get("current") or 0, job.get("total") or 1
        frac = min(1.0, max(0.0, cur / max(1, tot)))
        bar.progress(frac, text=f"{stage} — {cur}/{tot}")
        status = job.get("status")
        if status in ("done", "error"):
            bar.empty()
            status_box.empty()
            return job
        if status == "error":
            status_box.error(job.get("error") or "Analysis failed.")
        time.sleep(2)
    return job

pid = st.session_state.active_pid
jid = st.session_state.ingest_jid
if jid:
    job = db.get_job(jid)
    if job and job.get("status") not in ("done", "error"):
        st.info("Analyzing playlist — this takes a few minutes for long playlists. You can leave and come back.")
        job = poll_job(jid)
        if job and job.get("status") == "done":
            with st.spinner("Filling in verified transcripts for episodes missing captions…"):
                n = apply_seed_transcripts(pid)
                if n:
                    st.success(f"Added verified transcripts for {n} episode(s) missing captions.")
        st.rerun()
    elif job and job.get("status") == "error":
        st.error(f"Analysis failed: {job.get('error')}")
        st.session_state.ingest_jid = None

# ---------------------------------------------------------------- playlist picker
playlists = db.list_playlists()
ready = [p for p in playlists if p.get("status") == "ready"]
if not ready:
    st.info("👆 Paste a YouTube playlist URL above to build your searchable episode database.")
    st.stop()

options = {f"{p['title'] or p['url'][:40]} ({p.get('video_count', 0)} videos)": p["id"] for p in ready}
labels = list(options.keys())
default_idx = 0
if pid in options.values():
    default_idx = labels.index(next(k for k, v in options.items() if v == pid))
choice = st.selectbox("📚 Playlist", labels, index=default_idx)
pid = options[choice]
st.session_state.active_pid = pid

pl = db.get_playlist(pid)
# Backfill verified transcripts for episodes missing captions (cheap no-op
# when everything already has transcripts).
_n_seeded = apply_seed_transcripts(pid)
if _n_seeded:
    st.success(f"✅ Added verified transcripts for {_n_seeded} episode(s) missing captions.")
vids = db.get_videos(pid)
with_cap = sum(1 for v in vids if v.get("has_transcript"))
st.caption(f"{len(vids)} episodes • {with_cap} with searchable captions • {db.segment_count(pid):,} transcript segments")

# ---- transcribe episodes that have no captions (runs on the server)
_missing = [v for v in vids if not v.get("has_transcript")]
if _missing:
    _tx = st.session_state.get("tx_thread")
    _tx_running = _tx is not None and _tx.is_alive()
    if _tx_running:
        st.info("🎙️ Transcribing episodes without captions in the background… "
                "you can keep searching; newly transcribed episodes become searchable automatically. "
                "This takes a while (39 episodes).")
    else:
        if _tx is not None:
            # thread finished (or died); clear handle
            st.session_state["tx_thread"] = None
        if st.button(f"🎙️ Transcribe {len(_missing)} episode(s) without captions",
                     key="tx_btn",
                     help="Uses AI transcription on the server so every episode becomes searchable. Takes a while; runs in the background."):
            import threading
            from backend import ingest as _ingest

            def _run():
                try:
                    n = _ingest.transcribe_missing_episodes(pid)
                    print(f"[app] background transcription done: {n} episodes")
                except Exception as e:
                    print(f"[app] background transcription failed: {e}")

            th = threading.Thread(target=_run, daemon=True)
            st.session_state["tx_thread"] = th
            th.start()
            st.success("✅ Transcription started in the background! You can keep using the site.")
            st.info("🎙️ Transcribing episodes without captions… newly transcribed episodes become searchable automatically.")

# ---------------------------------------------------------------- search
query = st.text_input(
    "Describe the scene",
    placeholder="e.g. the scene where Suleiman Shah dies …",
    key="scene_query",
)
go = st.button("🔎 Find scene", type="primary")

if go and (query or "").strip():
    with st.spinner("Searching episodes… (first search downloads the AI model, ~1 min)"):
        res = search_mod.search(pid, query.strip())
    st.caption(f"Search method: {res.get('method')}")
    if res.get("note"):
        st.warning(res["note"])

    results = res.get("results") or []
    for i, r in enumerate(results, 1):
        with st.container(border=True):
            st.subheader(f"#{i} — Episode {r['episode']}: {r['title']}")
            cols = st.columns([1, 1, 1])
            cols[0].metric("Timestamp", r["timestamp_hms"])
            cols[1].metric("Confidence", r["confidence"])
            if _is_upload_video(r):
                vpath = _upload_video_path(r["youtube_id"])
                if vpath:
                    cols[2].caption("▶ Plays below at the exact second")
                else:
                    cols[2].caption("Video file not found on server")
            else:
                cols[2].link_button("▶ Watch scene", r["url_at"])
            if r.get("reasons"):
                st.write("**Why this matches:**")
                for reason in r["reasons"]:
                    st.write(f"- {reason}")
            if r.get("matched_lines"):
                with st.expander("Matched dialogue"):
                    for line in r["matched_lines"]:
                        st.write(f"> {line}")
            if _is_upload_video(r):
                vpath = _upload_video_path(r["youtube_id"])
                if vpath:
                    st.video(str(vpath), start_time=int(r["timestamp"]))

    alts = res.get("alternatives") or []
    if alts:
        with st.expander(f"Other possible matches ({len(alts)})"):
            for r in alts:
                if _is_upload_video(r):
                    st.markdown(
                        f"- **{r['title']}** @ `{r['timestamp_hms']}` "
                        f"({r['confidence']})"
                    )
                else:
                    st.markdown(
                        f"- **Episode {r['episode']}** @ `{r['timestamp_hms']}` — "
                        f"[{r['title']}]({r['url_at']}) ({r['confidence']})"
                    )
elif go:
    st.warning("Please describe the scene first.")

st.divider()

# ---------------------------------------------------------------- upload & analyze your own video
st.header("🎞 Or analyze your own video")
st.caption("Upload any video — it gets fully transcribed, becomes searchable, "
           "and plays back at the exact second of the scene.")
up = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mkv", "webm", "mov", "mp3", "wav", "m4a"],
    key="vid_up",
)
if st.button("🔍 Analyze uploaded video", type="primary", key="analyze_up",
             disabled=up is None):
    if up is None:
        st.warning("Please choose a video file first.")
        st.stop()
    udir = _uploads_dir()
    suffix = Path(up.name).suffix or ".mp4"
    # create the playlist first so we can name the file after it
    new_pid = db.create_playlist(f"upload:{up.name}", title=f"📁 {up.name}")
    yid = f"upload-{new_pid}"
    vpath = udir / f"{new_pid}{suffix}"
    with open(vpath, "wb") as f:
        f.write(up.getbuffer())
    _register_upload(yid, vpath)
    try:
        with st.spinner("Transcribing video… long videos can take a while, please keep this page open."):
            segs = _transcribe_upload(vpath)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.stop()
    if not segs:
        st.warning("No speech detected in this video.")
        st.stop()
    db.add_videos(new_pid, [{"youtube_id": yid, "title": up.name, "position": 1,
                             "url": "", "thumbnail": None}])
    v = db.get_videos(new_pid)[0]
    db.add_segments(v["id"], segs)
    db.mark_transcript(v["id"], True)
    db.update_playlist(new_pid, status="ready", video_count=1)
    with st.spinner("Indexing for search…"):
        _embed_segments_for_videos(new_pid, {v["id"]})
    search_mod.invalidate_cache(new_pid)
    st.session_state.active_pid = new_pid
    st.success(f"✅ Analyzed {len(segs)} segments from **{up.name}**. Describe a scene above to jump to it!")
    st.rerun()

st.caption("Note: analyzed data is kept while the app is running; very long playlists take a few minutes to analyze.")
