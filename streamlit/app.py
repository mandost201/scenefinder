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
            cols[2].link_button("▶ Watch scene", r["url_at"])
            if r.get("reasons"):
                st.write("**Why this matches:**")
                for reason in r["reasons"]:
                    st.write(f"- {reason}")
            if r.get("matched_lines"):
                with st.expander("Matched dialogue"):
                    for line in r["matched_lines"]:
                        st.write(f"> {line}")

    alts = res.get("alternatives") or []
    if alts:
        with st.expander(f"Other possible matches ({len(alts)})"):
            for r in alts:
                st.markdown(
                    f"- **Episode {r['episode']}** @ `{r['timestamp_hms']}` — "
                    f"[{r['title']}]({r['url_at']}) ({r['confidence']})"
                )
elif go:
    st.warning("Please describe the scene first.")

st.divider()
st.caption("Note: analyzed data is kept while the app is running; very long playlists take a few minutes to analyze.")
