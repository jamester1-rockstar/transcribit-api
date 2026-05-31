#!/usr/bin/env python3
"""
transcriber_v2.py — Universal Audio → MIDI Transcriber
=======================================================
Works for any instrument, any BPM, any time signature.

Usage Examples
--------------
# Guitar, auto-detect BPM, 4/4
python3 transcriber_v2.py song.wav out.mid --instrument guitar

# Piano, known BPM, waltz time
python3 transcriber_v2.py song.wav out.mid --instrument piano --bpm 120 --time-sig 3/4

# Bass, slow ballad, auto-detect everything
python3 transcriber_v2.py song.wav out.mid --instrument bass --bpm auto --time-sig auto

# Vocals/melody
python3 transcriber_v2.py song.wav out.mid --instrument voice --bpm 84 --time-sig 6/8

Supported instruments
---------------------
  guitar   — acoustic or electric (E2–E6)
  bass     — bass guitar (E1–G4)
  piano    — full range (A0–C8)
  voice    — singing melody (C3–C6)
  violin   — (G3–A7)
  ukulele  — (G4–A6)
  trumpet  — (F#3–D6)
  flute    — (C4–D7)
  generic  — wide range fallback (C2–C7)

Requirements
------------
  pip install librosa mido pretty_midi soundfile resampy
  Optional: pip install spleeter   (better vocal removal)
"""

import argparse
import sys
import warnings
import json
import time
from pathlib import Path

import numpy as np
import librosa
import pretty_midi
import soundfile as sf

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# INSTRUMENT PROFILES
# ══════════════════════════════════════════════════════
INSTRUMENTS = {
    "guitar":   {"fmin": "E2",  "fmax": "E6",  "program": 24,  "name": "Acoustic Guitar", "suppress_vocals": True,  "transpose": 12},
    "bass":     {"fmin": "E1",  "fmax": "G4",  "program": 33,  "name": "Electric Bass",   "suppress_vocals": False},
    "piano":    {"fmin": "A0",  "fmax": "C8",  "program": 0,   "name": "Acoustic Grand",  "suppress_vocals": False},
    "voice":    {"fmin": "C3",  "fmax": "C6",  "program": 54,  "name": "Synth Voice",     "suppress_vocals": False},
    "violin":   {"fmin": "G3",  "fmax": "A7",  "program": 40,  "name": "Violin",          "suppress_vocals": True},
    "ukulele":  {"fmin": "G4",  "fmax": "A6",  "program": 24,  "name": "Acoustic Guitar", "suppress_vocals": True,  "transpose": 12},
    "trumpet":  {"fmin": "F#3", "fmax": "D6",  "program": 56,  "name": "Trumpet",         "suppress_vocals": True},
    "flute":    {"fmin": "C4",  "fmax": "D7",  "program": 73,  "name": "Flute",           "suppress_vocals": True},
    "generic":  {"fmin": "C2",  "fmax": "C7",  "program": 0,   "name": "Piano",           "suppress_vocals": False},
}

# ══════════════════════════════════════════════════════
# TIME SIGNATURE PRESETS
# ══════════════════════════════════════════════════════
TIME_SIG_PRESETS = {
    "4/4": (4, 4),   # Common time
    "3/4": (3, 4),   # Waltz
    "6/8": (6, 8),   # Compound duple (jig feel)
    "2/4": (2, 4),   # March
    "5/4": (5, 4),   # Odd meter (Dave Brubeck, etc.)
    "7/8": (7, 8),   # Odd compound
    "12/8": (12, 8), # Slow blues / shuffle
    "2/2": (2, 2),   # Cut time
    "6/4": (6, 4),   # Slow 6
    "3/8": (3, 8),   # Fast waltz
}

# ══════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════
SR         = 22050
HOP_LENGTH = 256
N_FFT      = 2048
CONFIDENCE = 0.35
ONSET_DELTA = 0.04
MIN_NOTE_SEC = 0.05

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def parse_time_sig(ts_str):
    """Parse '4/4', '6/8', '3/4' etc. into (numerator, denominator)."""
    try:
        num, den = ts_str.strip().split("/")
        return int(num), int(den)
    except Exception:
        print(f"  ⚠ Could not parse time signature '{ts_str}' — defaulting to 4/4")
        return 4, 4

def beat_duration(bpm, ts_den):
    """Seconds per beat, adjusted for denominator (e.g. 6/8 has 8th-note beats)."""
    base = 60.0 / bpm
    if ts_den == 8:
        return base / 2
    if ts_den == 2:
        return base * 2
    return base

def grid_unit(bpm, ts_num, ts_den, subdivisions=16):
    """Smallest rhythmic grid unit in seconds."""
    spb = beat_duration(bpm, ts_den)
    # For compound meters (6/8, 12/8), group in 3s
    if ts_num % 3 == 0 and ts_den == 8:
        spb = spb * 3  # dotted quarter = one beat in 6/8
    return spb / (subdivisions / 4)

def quantise(value_sec, grid_sec):
    steps = max(1, round(value_sec / grid_sec))
    return steps * grid_sec

def velocity_from_rms(rms, rms_min, rms_max):
    if rms_max <= rms_min:
        return 80
    ratio = (rms - rms_min) / (rms_max - rms_min)
    return int(np.clip(40 + ratio * 70, 40, 110))

def progress(stage, total, msg):
    bar = "█" * stage + "░" * (total - stage)
    print(f"\n  [{bar}] Stage {stage}/{total}")
    print(f"  {msg}")

# ══════════════════════════════════════════════════════
# STAGE 1 — LOAD
# ══════════════════════════════════════════════════════

def load_audio(path):
    progress(1, 7, f"Loading: {path.name}")
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    dur = librosa.get_duration(y=y, sr=SR)
    mins, secs = divmod(int(dur), 60)
    print(f"  Duration: {mins}m {secs}s  |  Sample rate: {SR} Hz")
    return y

# ══════════════════════════════════════════════════════
# STAGE 2 — AUTO-DETECT BPM
# ══════════════════════════════════════════════════════

def detect_bpm(y, user_bpm):
    """
    If user passes 'auto', estimate BPM from audio using two methods and average.
    Returns (float bpm, bool was_auto_detected)
    """
    if str(user_bpm).lower() != "auto":
        bpm = float(user_bpm)
        print(f"\n  [User-supplied BPM]: {bpm}")
        return bpm, False

    progress(2, 7, "Auto-detecting BPM …")

    # Method A: dynamic programming beat tracker
    tempo_a, _ = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP_LENGTH)
    tempo_a = float(np.atleast_1d(tempo_a)[0])

    # Method B: tempogram (robust to percussion-sparse tracks)
    oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_LENGTH)
    tgram = librosa.feature.tempogram(onset_envelope=oenv, sr=SR, hop_length=HOP_LENGTH)
    tempo_b = float(librosa.tempo_frequencies(tgram.shape[0], sr=SR, hop_length=HOP_LENGTH)[np.argmax(tgram.mean(axis=1))])

    # Prefer Method A; use B as sanity check
    if abs(tempo_a - tempo_b) < 10:
        bpm = round((tempo_a + tempo_b) / 2, 1)
    else:
        bpm = tempo_a  # trust beat tracker over tempogram when they diverge

    # Common BPM sanity clamp
    if bpm < 40:
        bpm *= 2
    elif bpm > 220:
        bpm /= 2

    print(f"  Beat tracker: {tempo_a:.1f} BPM")
    print(f"  Tempogram:    {tempo_b:.1f} BPM")
    print(f"  → Using:      {bpm:.1f} BPM  (override with --bpm if wrong)")
    return bpm, True

# ══════════════════════════════════════════════════════
# STAGE 3 — AUTO-DETECT TIME SIGNATURE
# ══════════════════════════════════════════════════════

def detect_time_sig(y, bpm, user_ts):
    """
    If user passes 'auto', attempt to detect meter from beat strength patterns.
    Returns (numerator, denominator).
    """
    if str(user_ts).lower() != "auto":
        num, den = parse_time_sig(user_ts)
        print(f"\n  [User-supplied time sig]: {num}/{den}")
        return num, den

    progress(3, 7, "Auto-detecting time signature …")

    # Compute beat-synchronous spectral flux
    oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_LENGTH)
    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP_LENGTH, bpm=bpm)

    if len(beats) < 8:
        print("  ⚠ Too few beats to detect meter — defaulting to 4/4")
        return 4, 4

    # Sample onset strength at each beat
    beat_strengths = oenv[beats]

    # Test groupings: does beat 1 (downbeat) stand out every N beats?
    scores = {}
    for grouping in [2, 3, 4, 6]:
        if len(beat_strengths) < grouping * 2:
            continue
        # Reshape into groups and measure downbeat prominence
        trimmed = beat_strengths[:len(beat_strengths) - len(beat_strengths) % grouping]
        groups = trimmed.reshape(-1, grouping)
        downbeat_strength = groups[:, 0].mean()
        other_strength = groups[:, 1:].mean()
        scores[grouping] = downbeat_strength / (other_strength + 1e-6)

    if not scores:
        return 4, 4

    best_grouping = max(scores, key=scores.get)

    # Map grouping → time signature
    mapping = {2: (2, 4), 3: (3, 4), 4: (4, 4), 6: (6, 8)}
    num, den = mapping.get(best_grouping, (4, 4))

    print(f"  Meter scores: { {k: f'{v:.2f}' for k, v in scores.items()} }")
    print(f"  → Detected: {num}/{den}  (override with --time-sig if wrong)")
    return num, den

# ══════════════════════════════════════════════════════
# STAGE 4 — SOURCE SEPARATION / VOCAL SUPPRESSION
# ══════════════════════════════════════════════════════

def isolate_instrument(y, instrument_profile, mix_type="solo"):
    progress(4, 7, "Isolating instrument (vocal suppression + harmonic extraction) …")

    suppress = instrument_profile["suppress_vocals"]

    # mix_type tunes how hard we suppress the OTHER sources:
    #   none         -> input is already an isolated stem; skip suppression
    #   solo         -> light suppression (one instrument + vocals)
    #   band-guitar  -> aggressive suppression (extract guitar from a full band)
    #   band-bass    -> aggressive, low-frequency focus
    if mix_type == "none":
        suppress = False
    suppression_gain = {"solo": 0.55, "band-guitar": 0.30, "band-bass": 0.30,
                        "band-vocals": 1.0}.get(mix_type, 0.55)

    # Try spleeter for instruments where vocals interfere
    if suppress:
        try:
            from spleeter.separator import Separator
            import os, tempfile
            sep = Separator("spleeter:2stems")
            tmp_dir = tempfile.mkdtemp(prefix="trans_")
            tmp_in  = os.path.join(tmp_dir, "input.wav")
            sf.write(tmp_in, y, SR)
            sep.separate_to_file(tmp_in, tmp_dir)
            accomp_path = os.path.join(tmp_dir, "input", "accompaniment.wav")
            y_iso, _ = librosa.load(accomp_path, sr=SR, mono=True)
            y_iso = librosa.util.fix_length(y_iso, size=len(y))
            print("  ✓ Spleeter neural vocal separation applied")
            return y_iso
        except Exception:
            pass  # fall through to HPSS

    # HPSS — always applied
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = librosa.magphase(D)
    H, P = librosa.decompose.hpss(mag, margin=(2.0, 1.0))

    if suppress:
        # Spectral dip in vocal zone (200–800 Hz); depth set by mix_type
        freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
        mask = np.ones(len(freqs))
        vocal_zone = (freqs >= 200) & (freqs <= 800)
        mask[vocal_zone] = suppression_gain
        H = H * mask[:, np.newaxis]
        print(f"  ✓ HPSS + vocal suppression (mix={mix_type}, gain={suppression_gain})")
    else:
        why = "already-isolated stem" if mix_type == "none" else "no vocal suppression for this instrument"
        print(f"  ✓ Harmonic isolation applied ({why})")

    y_iso = librosa.istft(H * phase, hop_length=HOP_LENGTH)
    y_iso = librosa.util.fix_length(y_iso, size=len(y))
    return y_iso

# ══════════════════════════════════════════════════════
# STAGE 5 — ONSET DETECTION
# ══════════════════════════════════════════════════════

def detect_onsets(y_iso, bpm, onset_delta=ONSET_DELTA):
    progress(5, 7, "Detecting note onsets …")

    onset_env = librosa.onset.onset_strength(
        y=y_iso, sr=SR, hop_length=HOP_LENGTH,
        feature=librosa.feature.melspectrogram,
        aggregate=np.median
    )
    onset_env_2 = librosa.onset.onset_strength(y=y_iso, sr=SR, hop_length=HOP_LENGTH)
    onset_env = librosa.util.normalize(0.6 * onset_env + 0.4 * onset_env_2)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=SR, hop_length=HOP_LENGTH,
        delta=onset_delta,
        wait=4,
        pre_max=3, post_max=3,
        pre_avg=5, post_avg=5,
        backtrack=True
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP_LENGTH)
    print(f"  Found {len(onset_times)} onsets")
    return onset_times

# ══════════════════════════════════════════════════════
# STAGE 6 — PITCH DETECTION
# ══════════════════════════════════════════════════════

def detect_pitches(y_iso, onset_times, bpm, fmin_hz, fmax_hz, confidence=CONFIDENCE):
    progress(6, 7, f"Pitch detection per onset — fmin:{fmin_hz:.0f}Hz fmax:{fmax_hz:.0f}Hz …")

    spb = 60.0 / bpm
    notes = []
    n_onsets = len(onset_times)

    for i, onset in enumerate(onset_times):
        # Progress every 10%
        if i % max(1, n_onsets // 10) == 0:
            pct = int(100 * i / n_onsets)
            print(f"  {pct}% …", end="\r", flush=True)

        next_onset = onset_times[i + 1] if i + 1 < n_onsets else onset + spb * 2
        window = min(next_onset - onset, spb * 2.0)
        window = max(window, 0.08)

        s = int(onset * SR)
        e = min(int((onset + window) * SR), len(y_iso))
        y_win = y_iso[s:e]

        if len(y_win) < 512:
            continue

        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y_win, sr=SR,
                fmin=fmin_hz, fmax=fmax_hz,
                hop_length=HOP_LENGTH // 2,
                frame_length=2048,
                fill_na=None
            )
        except Exception:
            continue

        mask = voiced_flag & (voiced_prob >= confidence)
        f0_v = f0[mask]
        if len(f0_v) == 0:
            continue

        f0_med = np.nanmedian(f0_v)
        if np.isnan(f0_med) or f0_med <= 0:
            continue

        midi_note = int(round(librosa.hz_to_midi(f0_med)))
        midi_lo = int(round(librosa.hz_to_midi(fmin_hz)))
        midi_hi = int(round(librosa.hz_to_midi(fmax_hz)))
        midi_note = np.clip(midi_note, midi_lo, midi_hi)

        voiced_frames = np.where(mask)[0]
        if len(voiced_frames) > 0:
            dur = (voiced_frames[-1] - voiced_frames[0]) * (HOP_LENGTH // 2) / SR
            dur = max(dur, MIN_NOTE_SEC)
        else:
            dur = window * 0.5

        rms = float(np.sqrt(np.mean(y_win ** 2)))
        notes.append({"onset": onset, "duration": dur, "pitch": int(midi_note), "rms": rms})

    print(f"  Detected {len(notes)} pitched notes      ")
    return notes

# ══════════════════════════════════════════════════════
# STAGE 7 — QUANTISE + EXPORT
# ══════════════════════════════════════════════════════

def quantise_and_export(notes, bpm, ts_num, ts_den, instrument_profile, output_path,
                        fmin_hz=None, fmax_hz=None):
    progress(7, 7, "Quantising to grid and exporting MIDI …")

    grid = grid_unit(bpm, ts_num, ts_den)

    cleaned = []
    for n in notes:
        q_onset = round(n["onset"] / grid) * grid
        q_dur   = quantise(n["duration"], grid)
        q_dur   = max(q_dur, grid)
        q_dur   = min(q_dur, (60.0 / bpm) * ts_num)  # max = one full bar
        cleaned.append({**n, "onset": q_onset, "duration": q_dur})

    cleaned.sort(key=lambda x: x["onset"])

    # Cap overlaps
    for i in range(len(cleaned) - 1):
        gap = cleaned[i + 1]["onset"] - cleaned[i]["onset"]
        cleaned[i]["duration"] = min(cleaned[i]["duration"], max(grid, gap - 0.01))

    # ── Repeat-note collapse ────────────────────────────
    # Merge consecutive same-pitch notes separated by a tiny gap (string
    # resonance / re-pluck artefacts) into one sustained note. Prevents the
    # "burst of identical 16ths" that floods MuseScore.
    merged = []
    for n in cleaned:
        if merged and n["pitch"] == merged[-1]["pitch"]:
            prev = merged[-1]
            gap = n["onset"] - (prev["onset"] + prev["duration"])
            if gap <= grid * 1.25:
                prev["duration"] = (n["onset"] + n["duration"]) - prev["onset"]
                prev["rms"] = max(prev["rms"], n["rms"])
                continue
        merged.append(dict(n))
    cleaned = merged

    # ── Notation transpose + range cap ──────────────────
    # Classical guitar/ukulele are written an octave above sounding pitch, so
    # transpose +12 to keep everything on the treble staff (no bass-clef bleed).
    # Then hard-clip to the instrument range so stray octave errors don't
    # produce out-of-range red notes in MuseScore.
    transpose = instrument_profile.get("transpose", 0)
    if fmin_hz and fmax_hz:
        midi_lo = int(round(librosa.hz_to_midi(fmin_hz)))
        midi_hi = int(round(librosa.hz_to_midi(fmax_hz)))
    else:
        midi_lo, midi_hi = 0, 127
    for n in cleaned:
        p = n["pitch"] + transpose
        n["pitch"] = int(np.clip(p, midi_lo, midi_hi))

    # Deduplicate
    seen = set()
    final = []
    for n in cleaned:
        key = (n["onset"], n["pitch"])
        if key not in seen:
            seen.add(key)
            final.append(n)

    print(f"  {len(final)} notes after quantisation")

    # ── Build MIDI ──────────────────────────────────
    rms_vals = [n["rms"] for n in final] or [0]
    rms_min, rms_max = min(rms_vals), max(rms_vals)

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm), resolution=480)
    pm.time_signature_changes = [
        pretty_midi.TimeSignature(ts_num, ts_den, 0.0)
    ]

    inst = pretty_midi.Instrument(
        program=instrument_profile["program"],
        is_drum=False,
        name=instrument_profile["name"]
    )

    for n in final:
        vel = velocity_from_rms(n["rms"], rms_min, rms_max)
        inst.notes.append(pretty_midi.Note(
            velocity=vel,
            pitch=n["pitch"],
            start=float(n["onset"]),
            end=float(n["onset"] + n["duration"])
        ))

    pm.instruments.append(inst)
    pm.write(str(output_path))

    return final, pm

# ══════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════

def print_summary(notes, bpm, ts_num, ts_den, instrument, output_path, auto_bpm, auto_ts, elapsed):
    rms_vals = [n["rms"] for n in notes] or [0]
    rms_min, rms_max = min(rms_vals), max(rms_vals)

    print(f"""
╔══════════════════════════════════════════════════╗
║            TRANSCRIPTION COMPLETE                ║
╠══════════════════════════════════════════════════╣
║  Output:      {str(output_path):<34} ║
║  Instrument:  {instrument:<34} ║
║  BPM:         {str(bpm) + (' (auto-detected)' if auto_bpm else ' (user-supplied)'):<34} ║
║  Time Sig:    {str(ts_num)+'/'+str(ts_den) + (' (auto-detected)' if auto_ts else ' (user-supplied)'):<34} ║
║  Notes:       {str(len(notes)):<34} ║
║  Duration:    {f'{int(notes[-1]["onset"] // 60)}m {int(notes[-1]["onset"] % 60)}s':<34} ║
║  Processed:   {f'{elapsed:.1f}s':<34} ║
╚══════════════════════════════════════════════════╝
""")

    if auto_bpm or auto_ts:
        print("  ⚠  Auto-detected values — verify in MuseScore before finalising:")
        if auto_bpm:
            print(f"     BPM {bpm} — if beats feel off, rerun with --bpm <correct_value>")
        if auto_ts:
            print(f"     {ts_num}/{ts_den} — if barlines are wrong, rerun with --time-sig <correct_value>")
        print()

    print("  MuseScore Import Tips:")
    print("  • File → Open → select output.mid")
    print("  • Verify BPM in bottom status bar (double-click to change)")
    print("  • Edit → Selection → All Similar Elements to bulk-fix rests")
    print("  • For chords: select notes → Voice 2 button to split voices")
    print()

    # Note preview table
    print(f"  {'#':>4}  {'Time':>7}  {'Dur':>6}  {'MIDI':>5}  {'Note':>5}  {'Vel':>4}")
    print("  " + "─" * 40)
    for i, n in enumerate(notes[:30]):
        name = pretty_midi.note_number_to_name(n["pitch"])
        vel  = velocity_from_rms(n["rms"], rms_min, rms_max)
        print(f"  {i+1:>4}  {n['onset']:>7.3f}  {n['duration']:>6.3f}  {n['pitch']:>5}  {name:>5}  {vel:>4}")
    if len(notes) > 30:
        print(f"  … and {len(notes)-30} more notes")
    print()

# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Universal Audio → MIDI Transcriber v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 transcriber_v2.py song.wav out.mid --instrument guitar
  python3 transcriber_v2.py song.wav out.mid --instrument piano --bpm 120 --time-sig 3/4
  python3 transcriber_v2.py song.wav out.mid --bpm auto --time-sig auto
        """
    )
    parser.add_argument("input",  help="Input audio file (.wav, .mp3, .flac)")
    parser.add_argument("output", help="Output MIDI file (.mid)")
    parser.add_argument("--instrument", default="guitar",
                        choices=list(INSTRUMENTS.keys()),
                        help="Target instrument (default: guitar)")
    parser.add_argument("--mix-type", default="solo",
                        choices=["none","solo","band-guitar","band-vocals","band-bass"],
                        help="Source mix: 'none' for an isolated stem, 'solo' for one instrument+vocals, 'band-*' to extract from a full band")
    parser.add_argument("--bpm", default="auto",
                        help="BPM — number or 'auto' to detect (default: auto)")
    parser.add_argument("--time-sig", default="4/4",
                        help="Time signature — e.g. 4/4, 3/4, 6/8, or 'auto' (default: 4/4)")
    parser.add_argument("--onset-delta", type=float, default=0.04,
                        help="Onset sensitivity (lower = more notes; default: 0.04)")
    parser.add_argument("--confidence", type=float, default=0.35,
                        help="Pitch confidence 0-1 (lower = more notes; default: 0.35)")
    parser.add_argument("--preview", action="store_true",
                        help="Show note table in terminal after processing")

    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"\n  ERROR: File not found: {input_path}\n")
        sys.exit(1)

    profile = INSTRUMENTS[args.instrument]
    fmin_hz = librosa.note_to_hz(profile["fmin"])
    fmax_hz = librosa.note_to_hz(profile["fmax"])

    print(f"""
╔══════════════════════════════════════════════════╗
║      Universal Audio → MIDI Transcriber v2       ║
╠══════════════════════════════════════════════════╣
║  Instrument: {args.instrument:<35} ║
║  BPM:        {str(args.bpm):<35} ║
║  Time Sig:   {args.time_sig:<35} ║
╚══════════════════════════════════════════════════╝
""")

    t_start = time.time()

    y                    = load_audio(input_path)
    bpm, auto_bpm        = detect_bpm(y, args.bpm)
    ts_num, ts_den       = detect_time_sig(y, bpm, args.time_sig)
    y_iso                = isolate_instrument(y, profile, args.mix_type)
    onset_times          = detect_onsets(y_iso, bpm, onset_delta=args.onset_delta)
    notes_raw            = detect_pitches(y_iso, onset_times, bpm, fmin_hz, fmax_hz,
                                          confidence=args.confidence)
    notes, pm            = quantise_and_export(
                               notes_raw, bpm, ts_num, ts_den,
                               profile, output_path, fmin_hz, fmax_hz
                           )

    elapsed = time.time() - t_start

    if args.preview or True:  # always show summary
        print_summary(notes, bpm, ts_num, ts_den, args.instrument,
                      output_path, auto_bpm, (args.time_sig == "auto"), elapsed)


if __name__ == "__main__":
    main()
