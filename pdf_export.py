#!/usr/bin/env python3
"""
pdf_export.py — Transcribit Stage 8: PDF Lead Sheet Export
==========================================================
Turns a transcribed MIDI file (with embedded lyric events) into a
copyright-ready PDF lead sheet: title block, key / tempo / meter,
copyright line, and a chords-over-lyrics chart laid out by phrase.

Design note
-----------
Input is the MIDI produced by Stage 1-7 (+ embedded lyrics), NOT the audio,
so this module is fully decoupled from transcriber_v2.py. That keeps it
testable on its own and re-runnable without re-transcribing.

Pure ReportLab cannot *engrave* a five-line staff to professional quality.
What it CAN do — and what songwriters actually deposit for copyright — is a
clean chords-over-lyrics lead sheet with full credits and metadata. If you
later want engraved melody notation, route the MIDI through MusicXML
(music21) into MuseScore/LilyPond; the metadata layer here is reusable.

ASCII accidentals (F#m, Bb) are used on purpose: ReportLab's base fonts have
no glyph for U+266F / U+266D, which would render as black boxes.

SonicRockstar Records (c) 2025
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import mido
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth


# ── MUSIC THEORY HELPERS ───────────────────────────────────────
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (major / minor)
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Interval-set -> chord-quality suffix, longest match wins
CHORD_TEMPLATES = [
    (frozenset({0, 4, 7, 11}), "maj7"),
    (frozenset({0, 3, 7, 10}), "m7"),
    (frozenset({0, 4, 7, 10}), "7"),
    (frozenset({0, 4, 8}), "aug"),
    (frozenset({0, 3, 6}), "dim"),
    (frozenset({0, 5, 7}), "sus4"),
    (frozenset({0, 2, 7}), "sus2"),
    (frozenset({0, 4, 7}), ""),     # major triad
    (frozenset({0, 3, 7}), "m"),    # minor triad
]


def detect_key(pitch_class_weights):
    """Krumhansl-Schmuckler key finding from duration-weighted pitch classes."""
    if not any(pitch_class_weights):
        return "C", "major"

    def correlate(profile, rotation):
        rotated = [profile[(i - rotation) % 12] for i in range(12)]
        n = 12
        mp, mw = sum(rotated) / n, sum(pitch_class_weights) / n
        num = sum((rotated[i] - mp) * (pitch_class_weights[i] - mw) for i in range(n))
        den = (sum((rotated[i] - mp) ** 2 for i in range(n)) *
               sum((pitch_class_weights[i] - mw) ** 2 for i in range(n))) ** 0.5
        return num / den if den else 0.0

    best_score, best_key, best_mode = -2.0, "C", "major"
    for root in range(12):
        for profile, mode in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")):
            score = correlate(profile, root)
            if score > best_score:
                best_score, best_key, best_mode = score, PITCH_NAMES[root], mode
    return best_key, best_mode


def name_chord(pitch_classes):
    """Best-effort triad/7th label from a set of sounding pitch classes."""
    pcs = set(pitch_classes)
    if len(pcs) < 3:
        return None
    best = None  # (template_size, root, suffix)
    for root in pcs:
        intervals = frozenset((pc - root) % 12 for pc in pcs)
        for template, suffix in CHORD_TEMPLATES:
            if template.issubset(intervals):
                size = len(template)
                if best is None or size > best[0]:
                    best = (size, root, suffix)
    if best is None:
        return None
    _, root, suffix = best
    return f"{PITCH_NAMES[root]}{suffix}"


# ── MIDI PARSING ───────────────────────────────────────────────
class Note:
    __slots__ = ("pitch", "start", "end")

    def __init__(self, pitch, start, end):
        self.pitch, self.start, self.end = pitch, start, end


def parse_midi(midi_path):
    """Return (notes, lyrics, tempo_bpm, time_sig, duration_sec) in seconds."""
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat or 480

    tempo = 500000  # default 120 bpm in microseconds/beat
    numerator, denominator = 4, 4
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "time_signature":
                numerator, denominator = msg.numerator, msg.denominator

    sec_per_tick = (tempo / 1_000_000.0) / tpb

    notes, lyrics = [], []
    for track in mid.tracks:
        abs_tick = 0
        active = defaultdict(list)  # pitch -> [start_tick, ...]
        for msg in track:
            abs_tick += msg.time
            t = abs_tick * sec_per_tick
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note].append(abs_tick)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if active.get(msg.note):
                    start = active[msg.note].pop(0)
                    notes.append(Note(msg.note, start * sec_per_tick, abs_tick * sec_per_tick))
            elif msg.type == "lyrics":
                word = (msg.text or "").strip()
                if word:
                    lyrics.append((t, word))

    notes.sort(key=lambda n: n.start)
    lyrics.sort(key=lambda x: x[0])
    duration = max((n.end for n in notes), default=0.0)
    duration = max(duration, max((t for t, _ in lyrics), default=0.0))
    bpm = round(60_000_000 / tempo)
    return notes, lyrics, bpm, f"{numerator}/{denominator}", duration


def chords_by_time(notes, window=0.18):
    """Cluster note onsets that start within `window` seconds, label as chords.

    Returns [(onset_sec, 'Gmaj7'), ...]. Monophonic input yields few/no chords,
    which is correct — a single melodic line has no harmony to name.
    """
    if not notes:
        return []
    clusters, current, start = [], [], notes[0].start
    for n in notes:
        if n.start - start <= window:
            current.append(n)
        else:
            clusters.append((start, current))
            current, start = [n], n.start
    clusters.append((start, current))

    out = []
    for onset, group in clusters:
        # include notes still sustaining at this onset for fuller harmony
        sounding = {n.pitch % 12 for n in notes if n.start <= onset + window and n.end > onset}
        label = name_chord(sounding)
        if label:
            out.append((onset, label))
    # de-dup consecutive identical chords
    deduped = []
    for onset, label in out:
        if not deduped or deduped[-1][1] != label:
            deduped.append((onset, label))
    return deduped


def group_lyric_lines(lyrics, line_gap=1.1, section_gap=2.6):
    """Split flat (time, word) list into phrase lines and section breaks.

    Yields list of lines, where a line is dict{words:[(t,w)], section_break:bool}.
    """
    lines, current = [], []
    prev_t = None
    for t, w in lyrics:
        if prev_t is not None and (t - prev_t) > line_gap and current:
            section = (t - prev_t) > section_gap
            lines.append({"words": current, "section_break": section})
            current = []
        current.append((t, w))
        prev_t = t
    if current:
        lines.append({"words": current, "section_break": False})
    return lines


# ── PDF LAYOUT ─────────────────────────────────────────────────
INK = (0.08, 0.07, 0.09)
GOLD = (0.66, 0.52, 0.18)
MUTED = (0.42, 0.40, 0.46)
HAIR = (0.80, 0.78, 0.82)


def _fmt_dur(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def build_lead_sheet(midi_path, out_pdf, *, title=None, artist=None,
                     composer=None, copyright_holder="SonicRockstar Records",
                     year=2025, key=None, include_chords=True):
    notes, lyrics, bpm, time_sig, duration = parse_midi(midi_path)

    if key is None:
        weights = [0.0] * 12
        for n in notes:
            weights[n.pitch % 12] += max(n.end - n.start, 0.05)
        root, mode = detect_key(weights)
        key = f"{root}{'' if mode == 'major' else 'm'}"

    if title is None:
        title = Path(midi_path).stem.replace("_transcribit", "").replace("_", " ").title()

    chords = chords_by_time(notes) if include_chords else []
    lines = group_lyric_lines(lyrics)

    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    W, H = letter
    LM, RM, TM, BM = 0.9 * inch, 0.9 * inch, 0.85 * inch, 0.85 * inch
    content_w = W - LM - RM

    def footer(page_no):
        c.setFont("Helvetica", 7.5)
        c.setFillColorRGB(*MUTED)
        c.drawString(LM, BM - 22, f"(c) {year} {copyright_holder}. All rights reserved.")
        c.drawCentredString(W / 2, BM - 22, "Generated by Transcribit")
        c.drawRightString(W - RM, BM - 22, f"Page {page_no}")

    # ── HEADER ──────────────────────────────────────────────
    y = H - TM
    c.setFillColorRGB(*INK)
    c.setFont("Times-Bold", 26)
    c.drawString(LM, y - 18, title)

    subtitle_bits = []
    if artist:
        subtitle_bits.append(artist)
    credit = f"Music & Lyrics by {composer}" if composer else "Music & Lyrics"
    c.setFont("Times-Italic", 12)
    c.setFillColorRGB(*MUTED)
    sub_y = y - 36
    if subtitle_bits:
        c.drawString(LM, sub_y, " — ".join(subtitle_bits))
        sub_y -= 15
    c.drawString(LM, sub_y, credit)

    # meta box (right aligned)
    meta = [("KEY", key), ("TEMPO", f"{bpm} BPM"), ("METER", time_sig), ("LENGTH", _fmt_dur(duration))]
    c.setFont("Helvetica", 8)
    mx = W - RM
    my = y - 4
    for label, val in meta:
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 7)
        c.drawRightString(mx, my, label)
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 11)
        c.drawRightString(mx, my - 12, str(val))
        my -= 30

    # rule under header
    rule_y = min(sub_y, my) - 10
    c.setStrokeColorRGB(*GOLD)
    c.setLineWidth(1.2)
    c.line(LM, rule_y, W - RM, rule_y)

    # ── BODY: chords over lyrics ────────────────────────────
    page_no = 1
    y = rule_y - 30
    LYRIC_FONT, LYRIC_SIZE = "Helvetica", 11
    CHORD_FONT, CHORD_SIZE = "Helvetica-Bold", 10
    LINE_H, ROW_GAP = 16, 12
    SPACE = stringWidth(" ", LYRIC_FONT, LYRIC_SIZE)

    def chord_for(t):
        """Nearest chord onset within 0.6s of a word's time, else None."""
        if not chords:
            return None
        best, best_d = None, 0.6
        for onset, label in chords:
            d = abs(onset - t)
            if d < best_d:
                best, best_d = label, d
        return best

    last_chord_drawn = [None]  # only print a chord symbol when it changes

    def new_page():
        nonlocal page_no, y
        footer(page_no)
        c.showPage()
        page_no += 1
        y = H - TM
        c.setFillColorRGB(*INK)

    if not lines:
        # No lyrics — emit a melody/harmony summary instead of an empty sheet
        c.setFont("Helvetica-Oblique", 11)
        c.setFillColorRGB(*MUTED)
        c.drawString(LM, y, "Instrumental — no lyrics embedded.")
        y -= 22
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica", 10)
        if notes:
            lo, hi = min(n.pitch for n in notes), max(n.pitch for n in notes)
            c.drawString(LM, y, f"Notes: {len(notes)}   Range: "
                                f"{PITCH_NAMES[lo % 12]}{lo // 12 - 1} – "
                                f"{PITCH_NAMES[hi % 12]}{hi // 12 - 1}")
            y -= 18
        if chords:
            c.setFont("Helvetica-Bold", 11)
            chord_line = "   ".join(lbl for _, lbl in chords[:60])
            for chunk in _wrap(chord_line, "Helvetica-Bold", 11, content_w):
                if y < BM + 24:
                    new_page()
                c.drawString(LM, y, chunk)
                y -= 18
    else:
        for line in lines:
            if line["section_break"]:
                y -= ROW_GAP  # extra air between sections

            # measure full wrapped layout for this phrase
            if y - (LINE_H + ROW_GAP) < BM + 10:
                new_page()

            chord_y = y
            lyric_y = y - 12
            x = LM
            for t, w in line["words"]:
                token = w if w in ",.!?;:" else w + " "
                w_width = stringWidth(token, LYRIC_FONT, LYRIC_SIZE)
                if x + w_width > W - RM:
                    # wrap: drop two rows, reset x
                    y -= (LINE_H + ROW_GAP)
                    if y - (LINE_H + ROW_GAP) < BM + 10:
                        new_page()
                        chord_y = y
                    chord_y = y
                    lyric_y = y - 12
                    x = LM
                ch = chord_for(t)
                if ch and ch != last_chord_drawn[0]:
                    c.setFont(CHORD_FONT, CHORD_SIZE)
                    c.setFillColorRGB(*GOLD)
                    c.drawString(x, chord_y, ch)
                    last_chord_drawn[0] = ch
                c.setFont(LYRIC_FONT, LYRIC_SIZE)
                c.setFillColorRGB(*INK)
                c.drawString(x, lyric_y, token.rstrip() if token.strip() in ",.!?;:" else token)
                x += w_width
            y = lyric_y - (LINE_H + ROW_GAP)

    footer(page_no)
    c.save()

    return {
        "pdfPath": str(out_pdf),
        "title": title,
        "key": key,
        "bpm": bpm,
        "timeSig": time_sig,
        "duration": _fmt_dur(duration),
        "notes": len(notes),
        "words": len(lyrics),
        "chords": len(chords),
        "pages": page_no,
    }


def _wrap(text, font, size, max_w):
    words, line, out = text.split(), "", []
    for w in words:
        trial = (line + " " + w).strip()
        if stringWidth(trial, font, size) > max_w and line:
            out.append(line)
            line = w
        else:
            line = trial
    if line:
        out.append(line)
    return out


# ── CLI ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Transcribit Stage 8 — PDF lead sheet export")
    ap.add_argument("midi", help="input MIDI file (with embedded lyrics)")
    ap.add_argument("out", help="output PDF path")
    ap.add_argument("--title")
    ap.add_argument("--artist")
    ap.add_argument("--composer")
    ap.add_argument("--copyright-holder", default="SonicRockstar Records")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--key", help="override detected key, e.g. 'Em'")
    ap.add_argument("--no-chords", action="store_true")
    args = ap.parse_args()

    if not Path(args.midi).exists():
        print(f"ERROR: MIDI not found: {args.midi}", file=sys.stderr)
        sys.exit(1)

    info = build_lead_sheet(
        args.midi, args.out,
        title=args.title, artist=args.artist, composer=args.composer,
        copyright_holder=args.copyright_holder, year=args.year,
        key=args.key, include_chords=not args.no_chords,
    )
    print(f"PDF written: {info['pdfPath']}")
    print(f"Title: {info['title']}  Key: {info['key']}  Tempo: {info['bpm']}  "
          f"Meter: {info['timeSig']}  Pages: {info['pages']}")
    print(f"Notes: {info['notes']}  Words: {info['words']}  Chords: {info['chords']}")


if __name__ == "__main__":
    main()
