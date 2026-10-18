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
