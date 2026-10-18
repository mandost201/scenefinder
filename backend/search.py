"""Scene search over an analyzed playlist.

Two honest signals, blended:
  * keyword  - SQLite FTS5 BM25 over transcript segments (exact dialogue)
  * semantic - cosine similarity of query vs segment embeddings (meaning)

Results map to episode (playlist position), timestamp, and a deep link.
Confidence is derived from the actual scores and labelled plainly.
"""
import numpy as np

from . import db
from . import embeddings
from .vtt import format_hms
from .youtube import watch_url_at

# in-memory cache of playlist embeddings: pid -> dict
_emb_cache = {}


def _playlist_embeddings(pid):
    if pid in _emb_cache:
        return _emb_cache[pid]
    rows = db.load_embeddings(pid)
    cache = {"seg_ids": [], "video_ids": [], "starts": [], "matrix": None}
    if rows:
        mat = np.stack([np.frombuffer(blob, dtype=np.float32)
                        for _, _, _, _, blob in rows])
        cache = {"seg_ids": [r[0] for r in rows],
                 "video_ids": [r[1] for r in rows],
                 "starts": [r[2] for r in rows],
                 "matrix": mat}
    _emb_cache[pid] = cache
    return cache


def invalidate_cache(pid):
    _emb_cache.pop(pid, None)


def _videos_by_id(pid):
    return {v["id"]: v for v in db.get_videos(pid)}


def _confidence(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _nearby_lines(video_id, center_start, n=2):
    segs = db.get_segments(video_id)
    if not segs:
        return []
    idx = min(range(len(segs)), key=lambda i: abs(segs[i]["start"] - center_start))
    lo, hi = max(0, idx - n), min(len(segs), idx + n + 1)
    return [{"start": s["start"], "hms": format_hms(s["start"]), "text": s["text"]}
            for s in segs[lo:hi]]


def _build_result(video, start, score, method, matched_lines, reasons):
    return {
        "video_id": video["id"],
        "youtube_id": video["youtube_id"],
        "title": video["title"],
        "episode": video["position"],
        "timestamp": round(start, 1),
        "timestamp_hms": format_hms(start),
        "url": video["url"],
        "url_at": watch_url_at(video["youtube_id"], start),
        "thumbnail": video.get("thumbnail"),
        "confidence": _confidence(score),
        "score": round(float(score), 3),
        "method": method,
        "reasons": reasons,
        "matched_lines": matched_lines,
    }


def keyword_search(pid, query, top_k=8):
    hits = db.fts_search(pid, query, limit=300)
    if not hits:
        return [], "keyword"
    videos = _videos_by_id(pid)
    # best segment per video (fts rank is negative BM25: more negative = better)
    best = {}
    for h in hits:
        vid = h["video_id"]
        key = -h["rank"]
        if vid not in best or key > best[vid][0]:
            best[vid] = (key, h)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    top_score = ranked[0][1][0] if ranked else 1.0
    results = []
    for vid, (key, h) in ranked[:top_k]:
        video = videos.get(vid)
        if not video:
            continue
        score = min(1.0, key / max(top_score, 1e-9))
        # boost exact-phrase hits
        phrase = query.strip().lower()
        if phrase and phrase in h["text"].lower():
            score = min(1.0, score + 0.15)
        lines = _nearby_lines(vid, h["start"])
        results.append(_build_result(
            video, h["start"], score, "keyword", lines,
            [f"Dialogue match: \"{h['text'][:120]}\""]))
    return results, "keyword"


def semantic_search(pid, query, top_k=8):
    if not embeddings.available():
        return [], "semantic-unavailable"
    cache = _playlist_embeddings(pid)
    if cache["matrix"] is None:
        return [], "semantic-empty"
    q = embeddings.embed([query])[0]
    sims = cache["matrix"] @ q  # cosine (normalized)
    order = np.argsort(-sims)
    videos = _videos_by_id(pid)
    seen, results = set(), []
    for idx in order:
        if sims[idx] < 0.32:
            break
        vid = int(cache["video_ids"][idx])
        if vid in seen:
            continue
        seen.add(vid)
        video = videos.get(vid)
        if not video:
            continue
        start = float(cache["starts"][idx])
        lines = _nearby_lines(vid, start)
        results.append(_build_result(
            video, start, float(sims[idx]), "semantic", lines,
            ["Meaning/semantic match to your description"]))
        if len(results) >= top_k:
            break
    return results, "semantic"


def search(pid, query, mode="auto", top_k=8):
    """Combined search. Returns dict with results, alternatives, method info."""
    query = (query or "").strip()
    if not query:
        return {"results": [], "alternatives": [], "method": "none",
                "note": "Empty query."}
    kw, kw_method = keyword_search(pid, query, top_k=top_k)
    sem, sem_method = semantic_search(pid, query, top_k=top_k)

    merged = {}
    for r in kw:
        merged[r["video_id"]] = dict(r, score=r["score"] * 0.55,
                                     method="keyword")
    for r in sem:
        if r["video_id"] in merged:
            m = merged[r["video_id"]]
            m["score"] = min(1.0, m["score"] + r["score"] * 0.45)
            m["method"] = "keyword+semantic"
            if r["score"] > 0.5:
                m["reasons"].append("Also a strong semantic match to the description")
        else:
            merged[r["video_id"]] = dict(r, score=r["score"] * 0.9)
    ranked = sorted(merged.values(), key=lambda r: r["score"], reverse=True)
    for r in ranked:
        r["confidence"] = _confidence(r["score"])
        r["score"] = round(r["score"], 3)

    results = ranked[:3]
    alternatives = ranked[3:3 + max(0, top_k - 3)]

    methods = []
    if kw:
        methods.append("dialogue keyword match")
    if sem and sem_method == "semantic":
        methods.append("semantic meaning match")
    if not methods:
        if sem_method == "semantic-unavailable":
            methods.append("keyword only (semantic model unavailable)")
        elif sem_method == "semantic-empty":
            methods.append("keyword only (no embeddings indexed)")
    method_note = " + ".join(methods) if methods else "no matches"

    note = None
    if not ranked:
        note = ("No matching scenes found in this playlist's transcripts. "
                "Try fewer/different words, or check that the videos have "
                "captions (videos without captions can't be searched).")
    return {"results": results, "alternatives": alternatives,
            "method": method_note, "note": note,
            "semantic_available": embeddings.available()}
