# Transcribit API (Railway backend)

FastAPI service that turns uploaded audio into MIDI + time-aligned lyrics +
a copyright-ready lead-sheet PDF. Audio is processed in a temp dir and deleted
immediately after the response (verified). Pairs with the Vercel frontend.

## Endpoints
- `GET  /health`      readiness probe
- `POST /transcribe`  multipart audio → JSON with base64 midi/pdf/lyrics

## Deploy to Railway (copy-paste)
```bash
# 1) from this folder, create the repo
git init && git add . && git commit -m "Transcribit API"
git branch -M main
# 2) make an empty repo on github.com, then:
git remote add origin https://github.com/<you>/transcribit-api.git
git push -u origin main
```
- railway.app → New Project → Deploy from GitHub → pick `transcribit-api`.
- Railway reads `nixpacks.toml` (installs **ffmpeg**) and `Procfile`.
- Set Variables (Railway → Variables):
  - `ALLOWED_ORIGINS=https://transcribe.sonicrockstar.com`
  - optional: `MAX_UPLOAD_MB=60`, `TRANSCRIBE_TIMEOUT=600`
- Copy the public URL it gives you, then open `<url>/health` — expect
  `{"status":"ok",...}`.

## Then wire the frontend
In the Vercel site's `app.html`, set:
`const API_BASE = "https://<your-railway-url>";`

## Local test
```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
# POST a file:
curl -F "audio=@song.wav" -F "instrument=guitar" -F "bpm=94" \
     -F "includeLyrics=false" http://127.0.0.1:8000/transcribe
```

## Notes that affect cost / first run
- `openai-whisper` pulls in PyTorch → large image, slow first build.
- First lyrics request downloads the Whisper model (~30–60s cold start), then
  it's cached while the instance stays warm.
- Whisper `small` wants ~1.5–2 GB RAM. Size the Railway plan accordingly, or
  drop `openai-whisper` from requirements to launch MIDI+PDF only (much lighter).
- Requests are synchronous (hold the connection for the whole job). Fine at low
  traffic; move to a job-queue + polling when concurrency grows.
