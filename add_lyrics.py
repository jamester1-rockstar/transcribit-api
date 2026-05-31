#!/usr/bin/env python3
"""
add_lyrics.py — Transcribe vocals and embed lyrics into a MIDI file (v3)
=======================================================================
What changed vs the old version, and why it matters
---------------------------------------------------
1. VOCAL STEM input (--vocal-stem). Whisper runs on the isolated vocal, not
   the full mix. Cleaner words (fewer mishears like "blonde-sided") and the
   instrumental intro is silent, so no hallucinated intro lyrics.

2. KNOWN-LYRICS mode (--known-lyrics file.txt). You paste the correct lyrics.
   Whisper is used ONLY for *timing*; your exact words are mapped onto those
   timings by sequence alignment (difflib), with interpolation where Whisper
   misheard, split, or dropped a word. Result: correct words at real sung time.

3. TIME-BASED placement. The old code snapped every word to the nearest GUITAR
   note onset and allowed only one word per onset — so where the guitar played
   sparsely (bridge, final chorus) four or five words piled onto a single note.
   This version places each word at the tick of its actual sung time. Words are
   only *gently* snapped to a nearby note for clean MuseScore attachment, and
   never collapsed onto an already-occupied tick. No more pile-ups.

Usage
-----
  # Best quality — clean vocal stem + your exact lyrics:
  python3 add_lyrics.py Blindsided_guitar_or_mix.mid out.mid \
      --vocal-stem Blindsided_vocal.mp3 --known-lyrics Blindsided.txt --model small

  # Auto (Whisper words) on a vocal stem:
  python3 add_lyrics.py song.mid out.mid --vocal-stem vocal.mp3 --model small

  # Legacy (full mix, Whisper words) — still works:
  python3 add_lyrics.py song.mid out.mid --audio mix.wav --model base

Whisper models: tiny | base | small | medium | large  (downloaded on first run)

Requirements
------------
  pip install openai-whisper librosa mido soundfile
"""

import argparse
import difflib
import re
import sys
import time
import warnings
from pathlib import Path

import mido

warnings.filterwarnings("ignore")

WHISPER_SR = 16000
# Section markers / structural words to drop from pasted lyrics
_SECTION_WORDS = {
    "verse", "chorus", "prechorus", "pre", "bridge", "final", "intro", "outro",
    "shift", "empowerment", "rises", "refrain", "hook", "tag",
}


# ══════════════════════════════════════════════════════
# KNOWN-LYRICS PARSING
# ══════════════════════════════════════════════════════
def load_known_lyrics(path):
    """Parse pasted lyrics into an ordered list of word tokens.

    Strips markdown (### , *), bracketed [sections], (parentheticals), digits,
    and a small set of structural words. Tolerant of missing spaces around
    punctuation (e.g. 'was,No' -> ['was', 'No']).
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    # remove bracketed and parenthetical annotations
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    # remove markdown markers
    text = text.replace("#", " ").replace("*", " ")
    # tokenize words (letters + internal apostrophes/hyphens)
    raw = re.findall(r"[A-Za-z]+(?:['\u2019\-][A-Za-z]+)*", text)
    words = []
    for w in raw:
        norm = re.sub(r"[^a-z]", "", w.lower())
        if norm in _SECTION_WORDS:
            continue
        words.append(w)
    return words


# ══════════════════════════════════════════════════════
# WHISPER (timing backbone) — isolated so the rest is testable without it
# ══════════════════════════════════════════════════════
def whisper_words(audio_path, model_name="small", language="en"):
    """Return [{'word','start','end'}, ...] with word-level timestamps."""
    import librosa
    import whisper

    print(f"[1/4] Loading audio for Whisper: {Path(audio_path).name}")
    y, _ = librosa.load(str(audio_path), sr=WHISPER_SR, mono=True)
    audio_duration = len(y) / WHISPER_SR

    print(f"[2/4] Transcribing with Whisper '{model_name}' (word timestamps) …")
    t0 = time.time()
    model = whisper.load_model(model_name)
    lang = None if (not language or language.lower() == "auto") else language
    result = model.transcribe(y, word_timestamps=True, language=lang, fp16=False)

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            tok = w["word"].strip()
            if tok:
                words.append({"word": tok, "start": float(w["start"]), "end": float(w["end"])})
    print(f"      {len(words)} words in {time.time()-t0:.1f}s | first sung word at "
          f"{words[0]['start']:.2f}s" if words else "      no words detected")
    return words, result.get("text", ""), audio_duration


# ══════════════════════════════════════════════════════
# MAP KNOWN WORDS → WHISPER TIMINGS  (correct words, real timing)
# ══════════════════════════════════════════════════════
def _norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def map_known_to_timings(known_words, whisper, audio_duration=None):
    """Align known words to whisper word timings, robustly.

    whisper: list of {'word','start','end'}.
    Returns list of (known_word, time_seconds).

    Whisper supplies the timing backbone via sequence alignment. BUT if Whisper
    returns too few usable anchors (common when run on a near-silent stem, or
    when it only catches a couple of words), interpolating between those sparse
    anchors collapses the whole lyric into the first seconds. So we guard:
    if anchors don't span enough of the song, fall back to spreading the known
    words evenly across the audio duration.
    """
    kn = [_norm(w) for w in known_words]
    wn = [_norm(w["word"]) for w in whisper]
    wt = [w["start"] for w in whisper]
    n = len(known_words)

    # Duration we should span. Prefer the real audio length; else the last
    # whisper timestamp; else a nominal spacing.
    span_end = audio_duration or (wt[-1] if wt else None)

    # Collect anchored (known_index -> time) pairs from 'equal' blocks only.
    anchors = []
    sm = difflib.SequenceMatcher(a=kn, b=wn, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                anchors.append((i1 + k, wt[j1 + k]))

    # Decide whether anchors are trustworthy:
    #  - need a handful of them, AND
    #  - they must span a real portion of the timeline (not all bunched at t≈0)
    anchor_span = (anchors[-1][1] - anchors[0][1]) if len(anchors) >= 2 else 0.0
    enough = (
        len(anchors) >= max(8, int(0.05 * n)) and
        (span_end is None or anchor_span >= 0.4 * span_end)
    )

    if not enough:
        # FALLBACK: even distribution across the song. Honest and never collapses.
        total = span_end if (span_end and span_end > 5) else max(1.0, n * 0.45)
        start = 0.0
        return [(known_words[i], start + (total - start) * (i / max(1, n - 1)))
                for i in range(n)]

    # Trustworthy anchors: fill the rest by interpolation between them.
    times = [None] * n
    for idx, t in anchors:
        times[idx] = t
    _fill_none_by_interpolation(times, span_end)
    # enforce non-decreasing
    for i in range(1, n):
        if times[i] < times[i - 1]:
            times[i] = times[i - 1]
    return list(zip(known_words, times))


def _fill_none_by_interpolation(times, span_end=None):
    """Linearly interpolate None entries from nearest known neighbours (in place)."""
    n = len(times)
    first = next((i for i, t in enumerate(times) if t is not None), None)
    if first is None:
        end = span_end or float(n)
        for i in range(n):
            times[i] = end * (i / max(1, n - 1))
        return
    # leading words before first anchor: back off at a modest pace, floor 0
    for i in range(first):
        times[i] = max(0.0, times[first] - (first - i) * 0.3)
    i = first
    last = next((k for k in range(n - 1, -1, -1) if times[k] is not None), first)
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        left = times[i - 1]
        if j < n:
            right = times[j]
            steps = j - (i - 1)
            for k in range(i, j):
                times[k] = left + (right - left) * ((k - (i - 1)) / steps)
        else:
            # trailing words after last anchor: spread toward song end if known
            if span_end and span_end > left:
                steps = n - (i - 1)
                for k in range(i, n):
                    times[k] = left + (span_end - left) * ((k - (i - 1)) / steps)
            else:
                for k in range(i, n):
                    times[k] = left + (k - (i - 1)) * 0.3
        i = j


# ══════════════════════════════════════════════════════
# TIME-BASED PLACEMENT (no onset pile-ups)
# ══════════════════════════════════════════════════════
def place_lyrics_by_time(midi_path, words_with_time, snap_tol_beats=0.4):
    """Place each (word, time) at the tick of its sung time.

    Words are gently snapped to a nearby note onset (within snap_tol_beats) for
    clean MuseScore attachment, but never onto a tick already taken by another
    word — so distinct words stay distinct. Returns (mid, aligned, tpb, tempo).
    """
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat or 480

    tempo = 500000
    for tr in mid.tracks:
        for m in tr:
            if m.type == "set_tempo":
                tempo = m.tempo
                break

    sec_per_beat = tempo / 1_000_000.0

    def sec_to_tick(s):
        return int(round((s / sec_per_beat) * tpb))

    # collect note onset ticks for gentle snapping
    onset_ticks = []
    for tr in mid.tracks:
        ab = 0
        for m in tr:
            ab += m.time
            if m.type == "note_on" and m.velocity > 0:
                onset_ticks.append(ab)
    onset_ticks = sorted(set(onset_ticks))
    snap_tol = int(snap_tol_beats * tpb)

    import bisect
    aligned = []
    taken = set()
    last_tick = -1
    for word, t in words_with_time:
        tick = sec_to_tick(t)
        # gentle snap to nearest onset if within tolerance and free
        if onset_ticks:
            idx = bisect.bisect_left(onset_ticks, tick)
            for cand in (onset_ticks[idx] if idx < len(onset_ticks) else None,
                         onset_ticks[idx - 1] if idx > 0 else None):
                if cand is not None and abs(cand - tick) <= snap_tol and cand not in taken:
                    tick = cand
                    break
        # Preserve lyric order: a later word can never land before an earlier
        # one. If snapping/timing would place it at or before the previous word,
        # push it just after. This stops near-simultaneous words from swapping
        # (e.g. "hand your in mine" -> "hand in your mine").
        if tick <= last_tick:
            tick = last_tick + 1
        # never collapse onto an occupied tick
        while tick in taken:
            tick += 1
        taken.add(tick)
        last_tick = tick
        aligned.append((tick, word))

    aligned.sort(key=lambda x: x[0])
    return mid, aligned, tpb, tempo


# ══════════════════════════════════════════════════════
# EMBED + EXPORT
# ══════════════════════════════════════════════════════
def embed_lyrics(mid, aligned, tpb, tempo, output_path):
    print("[4/4] Embedding lyrics into MIDI …")
    if not aligned:
        print("  ⚠ no aligned words — saving MIDI unchanged")
        mid.save(str(output_path))
        return

    new_tracks = []
    for ti, track in enumerate(mid.tracks):
        msgs = []
        ab = 0
        for m in track:
            ab += m.time
            msgs.append((ab, m.copy(time=0)))
        if ti == 0:  # put lyric meta events on the first track
            for tick, word in aligned:
                msgs.append((tick, mido.MetaMessage("lyrics", text=word, time=0)))
        msgs.sort(key=lambda x: x[0])
        nt = mido.MidiTrack()
        prev = 0
        for ab, m in msgs:
            nt.append(m.copy(time=ab - prev))
            prev = ab
        new_tracks.append(nt)

    out = mido.MidiFile(type=mid.type, ticks_per_beat=tpb)
    out.tracks.extend(new_tracks)
    out.save(str(output_path))
    print(f"      ✓ {output_path}  ({len(aligned)} words embedded)")

    txt = Path(output_path).with_suffix(".lyrics.txt")
    sec_per_beat = tempo / 1_000_000.0
    with open(txt, "w", encoding="utf-8") as f:
        f.write("LYRICS (time-aligned)\n=====================\n\n")
        for tick, word in aligned:
            f.write(f"[{(tick / tpb) * sec_per_beat:7.2f}s]  {word}\n")
    print(f"      ✓ {txt}")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Add time-aligned lyrics to a MIDI file (v3)")
    ap.add_argument("midi", help="MIDI to add lyrics to (the transcribed part)")
    ap.add_argument("output", help="output MIDI with lyrics embedded")
    ap.add_argument("--vocal-stem", help="isolated vocal audio (best) for Whisper")
    ap.add_argument("--audio", help="full-mix audio (fallback if no vocal stem)")
    ap.add_argument("--known-lyrics", help="text file of correct lyrics (uses Whisper only for timing)")
    ap.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large"])
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    audio = args.vocal_stem or args.audio
    if not audio:
        print("ERROR: provide --vocal-stem (preferred) or --audio")
        sys.exit(1)
    if not Path(args.midi).exists():
        print(f"ERROR: MIDI not found: {args.midi}")
        sys.exit(1)

    words, _, audio_duration = whisper_words(audio, args.model, args.language)

    print("[3/4] Aligning lyrics …")
    if args.known_lyrics:
        known = load_known_lyrics(args.known_lyrics)
        print(f"      known lyrics: {len(known)} words | whisper: {len(words)} words")
        if not known:
            print("  ⚠ known-lyrics file had no usable words.")
            sys.exit(1)
        # span fallback: prefer audio duration, else MIDI length
        span = audio_duration
        if not span or span < 5:
            try:
                span = mido.MidiFile(str(args.midi)).length
            except Exception:
                span = None
        words_with_time = map_known_to_timings(known, words, audio_duration=span)
        if not words:
            print("  ℹ Whisper found no vocal — distributing your pasted lyrics "
                  "evenly across the song (open in MuseScore to fine-tune).")
    else:
        if not words:
            print("  ⚠ Whisper found no words and no known lyrics were provided — "
                  "is the audio an instrumental or a near-silent stem?")
            sys.exit(1)
        words_with_time = [(w["word"], w["start"]) for w in words]

    mid, aligned, tpb, tempo = place_lyrics_by_time(args.midi, words_with_time)
    embed_lyrics(mid, aligned, tpb, tempo, Path(args.output))


if __name__ == "__main__":
    main()
