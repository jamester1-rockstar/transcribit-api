#!/usr/bin/env python3
"""
server.py — Transcribit backend for Railway (decoupled architecture)
====================================================================
The browser (served from Vercel) POSTs an audio file here over HTTPS.
We process it in a private temp directory, read the outputs into memory,
base64-encode them, return JSON, and ALWAYS delete the temp files in a
finally: block — so nothing remains on the server after the request.

Endpoints
---------
  GET  /health            readiness probe
  POST /transcribe        multipart audio -> {midi_b64, pdf_b64, lyrics_b64, stats}

Run locally:   uvicorn server:app --host 0.0.0.0 --port 8000
Run on Railway: see Procfile (uses $PORT)

SonicRockstar Records (c) 2025
"""

import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

APP_DIR = Path(__file__).parent
MAX_SECONDS = int(os.environ.get("TRANSCRIBE_TIMEOUT", "600"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "60"))

# Lock CORS down to your site in production via ALLOWED_ORIGINS env
# (comma-separated). Defaults to "*" for first-run/testing.
_origins = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED = ["*"] if _origins.strip() == "*" else [o.strip() for o in _origins.split(",") if o.strip()]

app = FastAPI(title="Transcribit API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "Transcribit API", "version": "1.0.0"}


def _run(cmd, timeout):
    """Run a module, capture combined output, raise on failure."""
    p = subprocess.run(cmd, cwd=str(APP_DIR), capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stdout + "\n" + p.stderr)[-800:])
    return p.stdout


def _pipeline(work: Path, audio_path: Path, vocal_path, params):
    """Synchronous pipeline run inside a worker thread. Returns stats + file paths."""
    stem = audio_path.stem
    midi = work / f"{stem}.mid"

    out = _run([sys.executable, str(APP_DIR / "transcriber_v2.py"), str(audio_path), str(midi),
                "--instrument", params["instrument"], "--bpm", str(params["bpm"]),
                "--time-sig", params["timeSig"], "--onset-delta", str(params["onsetDelta"]),
                "--confidence", str(params["confidence"])], MAX_SECONDS)
    notes = 0
    m = re.search(r"(\d+)\s+notes after quantisation", out)
    if m:
        notes = int(m.group(1))
    m = re.search(r"Using:\s+([\d.]+)\s+BPM", out)
    bpm_used = m.group(1) if m else params["bpm"]

    final_midi = midi
    lyrics_txt = None
    words = 0
    if params["includeLyrics"]:
        final_midi = work / f"{stem}_lyrics.mid"
        cmd = [sys.executable, str(APP_DIR / "add_lyrics.py"), str(midi), str(final_midi),
               "--model", params["whisperModel"]]
        cmd += (["--vocal-stem", str(vocal_path)] if vocal_path else ["--audio", str(audio_path)])
        if params["knownLyrics"].strip():
            kl = work / "known.txt"
            kl.write_text(params["knownLyrics"], encoding="utf-8")
            cmd += ["--known-lyrics", str(kl)]
        try:
            o2 = _run(cmd, MAX_SECONDS)
            m = re.search(r"(\d+)\s+words embedded", o2)
            if m:
                words = int(m.group(1))
            lt = final_midi.with_suffix(".lyrics.txt")
            lyrics_txt = lt if lt.exists() else None
        except Exception:
            final_midi = midi  # keep no-lyrics MIDI if lyric step fails

    pdf = None
    if params["includePdf"]:
        pdf = work / f"{stem}_leadsheet.pdf"
        pcmd = [sys.executable, str(APP_DIR / "pdf_export.py"), str(final_midi), str(pdf),
                "--copyright-holder", params["copyrightHolder"], "--year", str(params["year"])]
        if params["title"]:    pcmd += ["--title", params["title"]]
        if params["composer"]: pcmd += ["--composer", params["composer"]]
        _run(pcmd, 120)

    return {
        "notes": notes, "bpm": bpm_used, "timeSig": params["timeSig"], "words": words,
        "midi": final_midi, "lyrics": lyrics_txt, "pdf": pdf, "stem": stem,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    vocal: UploadFile = File(None),
    instrument: str = Form("guitar"),
    bpm: str = Form("auto"),
    timeSig: str = Form("4/4"),
    onsetDelta: float = Form(0.03),
    confidence: float = Form(0.25),
    whisperModel: str = Form("small"),
    includeLyrics: bool = Form(True),
    includePdf: bool = Form(True),
    knownLyrics: str = Form(""),
    title: str = Form(""),
    composer: str = Form(""),
    copyrightHolder: str = Form("SonicRockstar Records"),
    year: int = Form(2025),
):
    # private per-request workspace
    work = Path(tempfile.mkdtemp(prefix="transcribit_"))
    audio_name = Path(audio.filename or "audio.wav").name
    audio_path = work / (audio_name if audio_name else "audio.wav")
    vocal_path = None
    try:
        data = await audio.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit")
        audio_path.write_bytes(data)

        if vocal is not None:
            vname = Path(vocal.filename or "vocal.wav").name
            vocal_path = work / (vname if vname else "vocal.wav")
            vocal_path.write_bytes(await vocal.read())

        params = dict(instrument=instrument, bpm=bpm, timeSig=timeSig,
                      onsetDelta=onsetDelta, confidence=confidence, whisperModel=whisperModel,
                      includeLyrics=includeLyrics, includePdf=includePdf, knownLyrics=knownLyrics,
                      title=title, composer=composer, copyrightHolder=copyrightHolder, year=year)

        loop = asyncio.get_event_loop()
        r = await loop.run_in_executor(None, lambda: _pipeline(work, audio_path, vocal_path, params))

        def b64(p):
            return base64.b64encode(Path(p).read_bytes()).decode() if p and Path(p).exists() else None

        return {
            "success": True,
            "notes": r["notes"], "bpm": r["bpm"], "timeSig": r["timeSig"], "words": r["words"],
            "midiName": f"{r['stem']}_transcribit.mid",
            "pdfName": f"{r['stem']}_leadsheet.pdf",
            "lyricsName": f"{r['stem']}.lyrics.txt",
            "midi_b64": b64(r["midi"]),
            "pdf_b64": b64(r["pdf"]),
            "lyrics_b64": b64(r["lyrics"]),
        }

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Processing timed out — try a shorter clip or a faster Whisper model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # ALWAYS remove the workspace — audio, MIDI, PDF, everything.
        shutil.rmtree(work, ignore_errors=True)
