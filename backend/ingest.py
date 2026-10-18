"""Background playlist ingestion: metadata -> transcripts -> embeddings."""
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import db
from . import embeddings
from . import search as search_mod
from .config import MAX_VIDEOS_PER_PLAYLIST, REQUEST_DELAY, TRANSCRIPT_WORKERS
from .youtube import fetch_playlist, fetch_transcript


def run_playlist_job(jid: str, pid: str, url: str):
    def progress(stage, current, total):
        db.update_job(jid, status="running", stage=stage,
                      current=current, total=total)

    try:
        progress("Reading playlist", 0, 1)
        title, videos = fetch_playlist(url)
        videos = videos[:MAX_VIDEOS_PER_PLAYLIST]
        db.update_playlist(pid, title=title, video_count=len(videos),
                           status="ingesting")
        db.add_videos(pid, videos)
        stored = db.get_videos(pid)
        total = len(stored)

        # ---- transcripts (parallel, polite)
        progress("Downloading captions", 0, total)
        done = [0]

        def one(v):
            try:
                segs = fetch_transcript(v["youtube_id"])
                if segs:
                    db.add_segments(v["id"], segs)
                    db.mark_transcript(v["id"], True)
                else:
                    db.mark_transcript(v["id"], False)
            except Exception as e:
                print(f"[ingest] transcript failed for {v['youtube_id']}: {e}")
                db.mark_transcript(v["id"], False)
            finally:
                time.sleep(REQUEST_DELAY)
                done[0] += 1
                progress("Downloading captions", done[0], total)

        with ThreadPoolExecutor(max_workers=TRANSCRIPT_WORKERS) as ex:
            list(ex.map(one, stored))

        # ---- embeddings
        progress("Indexing for semantic search", 0, 1)
        _index_embeddings(pid)

        with_caption = sum(1 for v in db.get_videos(pid) if v["has_transcript"])
        db.update_playlist(pid, status="ready", analyzed_at=time.time())
        search_mod.invalidate_cache(pid)
        db.update_job(jid, status="done", stage="done", current=total, total=total)
        print(f"[ingest] playlist {pid}: {total} videos, {with_caption} with captions")
    except Exception as e:
        traceback.print_exc()
        db.update_playlist(pid, status="error")
        db.update_job(jid, status="error", error=str(e))


def _index_embeddings(pid: str):
    if not embeddings.available():
        print("[ingest] embeddings unavailable - keyword search only")
        return
    dim = embeddings.dim() or 384
    batch, seg_ids = [], []
    for v in db.get_videos(pid):
        for s in db.get_segments(v["id"]):
            # embed short windows (merge tiny segments implicitly by batching text)
            if len(s["text"]) < 3:
                continue
            seg_ids.append(s["id"])
            batch.append(s["text"][:500])
    B = 64
    for i in range(0, len(batch), B):
        try:
            vecs = embeddings.embed(batch[i:i + B])
        except Exception as e:
            print(f"[ingest] embedding batch failed: {e}")
            break
        pairs = [(seg_ids[i + j], int(vecs.shape[1]),
                  vecs[j].astype(np.float32).tobytes())
                 for j in range(vecs.shape[0])]
        db.save_embeddings(pairs)


def start_playlist_job(pid: str, url: str) -> str:
    jid = db.create_job("playlist", pid)
    t = threading.Thread(target=run_playlist_job, args=(jid, pid, url),
                         daemon=True)
    t.start()
    return jid
