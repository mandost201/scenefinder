---
title: SceneFinder — AI YouTube Scene Finder
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Paste a YouTube playlist, find any scene by description, clip or screenshot
---

# 🎬 SceneFinder — AI YouTube Episode & Scene Finder

Paste a YouTube **playlist URL** → it builds a searchable episode database. Then:

- **Option A** — upload a clip or screenshots to find the exact scene
- **Option B** — describe the scene or quote dialogue

Results show episode title/number, timestamp (HH:MM:SS), a clickable timestamped
YouTube link, confidence score with reasons, and alternative matches.

## How it works

1. **Ingest** — playlist metadata + captions are fetched with `yt-dlp`
   (no API key needed; optional `YOUTUBE_API_KEY` raises quotas).
2. **Index** — captions are chunked and embedded with
   `sentence-transformers/all-MiniLM-L6-v2` (fastembed, ONNX, CPU-friendly)
   into SQLite + FTS5.
3. **Search** — hybrid keyword (BM25) + semantic search, blended and ranked.
4. **Clip analysis** — audio extracted with ffmpeg, transcribed with
   faster-whisper (`small`), then fuzzy-aligned against episode transcripts.
5. **Screenshot analysis** — OCR (tesseract) + perceptual hashing, labelled
   as hint-only when weak.

## Notes

- Runs fully on CPU on the free tier.
- The database is ephemeral: analysed playlists are forgotten when the Space
  restarts or sleeps. Re-paste the playlist to re-analyse.
- Respects YouTube rate limits; very long playlists are processed in the
  background with a progress bar.
