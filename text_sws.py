## MIT License
##
## Estimate formants directly from TEXT and synthesise sinewave speech from
## them -- no microphone, no recording. This is a small, deliberately crude
## rule-based formant synthesiser in the spirit of the classic Klatt / DECtalk
## approach:
##
##     text -> phonemes -> per-phoneme formant targets -> interpolate over time
##           -> stack gliding sine waves (one per formant) -> WAV
##
## Vowels and sonorants (l, r, m, n, w, y) have well-defined formants and come
## out recognisable; stops and fricatives are bursts/noise that a pure-sine
## model represents poorly, so they are approximated with weak, short segments.
## Pass --phonemes (ARPAbet) for precise control; --text uses a rough built-in
## letter-to-sound guesser.
##
## The synthesis is the continuous-phase ("truly gliding") variant of the sine
## stacking in sws.sinethesise: because we define smooth formant *targets*, we
## integrate phase per sample so each tone slides without clicks. Output can be
## split per-tone with --separate, exactly like sws.py / stream_sws.py.

import os
import re
import sys
import argparse

import numpy as np
import scipy.io.wavfile


# ----------------------------------------------------------------------------
# Phoneme -> formant model (adult-male averages, Hz; approximate)
# ----------------------------------------------------------------------------

# Monophthong vowels: (F1, F2, F3)
MONO = {
    "AA": (730, 1090, 2440), "AE": (660, 1720, 2410), "AH": (640, 1190, 2390),
    "AO": (570, 840, 2410),  "EH": (530, 1840, 2480), "ER": (490, 1350, 1690),
    "IH": (390, 1990, 2550), "IY": (270, 2290, 3010), "UH": (440, 1020, 2240),
    "UW": (300, 870, 2240),  "AX": (500, 1500, 2500),
}

# Diphthongs: glide from a start target to an end target
DIPH = {
    "AW": ((730, 1090, 2440), (330, 870, 2240)),
    "AY": ((730, 1090, 2440), (300, 2200, 3000)),
    "EY": ((480, 1720, 2520), (300, 2200, 3000)),
    "OW": ((540, 900, 2400),  (330, 870, 2240)),
    "OY": ((570, 840, 2410),  (300, 2200, 3000)),
}

# Consonant formant loci (very approximate; obstruents are not really tonal)
CONS = {
    "L": (360, 1300, 2700), "R": (420, 1300, 1600), "W": (300, 610, 2200),
    "Y": (270, 2200, 3000), "M": (250, 1100, 2300), "N": (250, 1700, 2600),
    "NG": (250, 2000, 2900),
    "V": (300, 1300, 2400), "DH": (300, 1400, 2500), "Z": (300, 1500, 2500),
    "ZH": (300, 1800, 2600), "F": (400, 1300, 2400), "TH": (400, 1600, 2600),
    "S": (320, 1700, 2600), "SH": (400, 1800, 2400), "HH": (500, 1500, 2500),
    "B": (300, 900, 2200), "D": (300, 1700, 2600), "G": (300, 1900, 2500),
    "P": (300, 900, 2200), "T": (300, 1700, 2600), "K": (300, 1900, 2500),
    "CH": (400, 1800, 2400), "JH": (300, 1800, 2600),
}

NASAL = {"M", "N", "NG"}
APPROX = {"L", "R", "W", "Y"}
FRIC_VOICED = {"V", "DH", "Z", "ZH"}
FRIC_UNVOICED = {"F", "TH", "S", "SH", "HH"}
STOP_VOICED = {"B", "D", "G"}
STOP_UNVOICED = {"P", "T", "K"}
AFFR = {"CH", "JH"}

# relative loudness of each phoneme class (overall energy envelope)
def amp_of(ph):
    if ph in ("sp", "sil"):       return 0.0
    if ph in MONO or ph in DIPH:  return 1.0
    if ph in APPROX:              return 0.7
    if ph in NASAL:               return 0.5
    if ph in FRIC_VOICED:         return 0.3
    if ph in AFFR:                return 0.25
    if ph in FRIC_UNVOICED:       return 0.15
    if ph in STOP_VOICED:         return 0.12
    if ph in STOP_UNVOICED:       return 0.06
    return 0.5

# default duration of each phoneme class, in milliseconds
def dur_of(ph):
    if ph == "sil":               return 260.0
    if ph == "sp":                return 60.0
    if ph in DIPH:                return 200.0
    if ph in MONO:                return 140.0
    if ph in NASAL or ph in APPROX: return 75.0
    if ph in FRIC_VOICED or ph in FRIC_UNVOICED: return 100.0
    if ph in AFFR:                return 110.0
    return 70.0  # stops

def targets_of(ph):
    """Return a list of formant vectors (1 for steady sounds, 2 for diphthongs)."""
    if ph in DIPH:
        a, b = DIPH[ph]
        return [a, b]
    if ph in MONO:
        return [MONO[ph]]
    if ph in CONS:
        return [CONS[ph]]
    return [MONO["AX"]]  # unknown -> neutral schwa


# ----------------------------------------------------------------------------
# Text -> phonemes (rough; use --phonemes for precise control)
# ----------------------------------------------------------------------------

EXCEPTIONS = {
    "the": ["DH", "AH"], "a": ["AH"], "to": ["T", "UW"], "of": ["AH", "V"],
    "i": ["AY"], "you": ["Y", "UW"], "are": ["AA", "R"], "is": ["IH", "Z"],
    "he": ["HH", "IY"], "she": ["SH", "IY"], "we": ["W", "IY"], "me": ["M", "IY"],
    "hello": ["HH", "AH", "L", "OW"], "world": ["W", "ER", "L", "D"],
    "one": ["W", "AH", "N"], "two": ["T", "UW"], "three": ["TH", "R", "IY"],
}


def word_to_phones(w):
    """Crude English letter-to-sound. Recognisable, not accurate."""
    w = w.lower()
    out, i, n = [], 0, len(w)
    while i < n:
        c = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        pair = w[i:i + 2]
        tri = w[i:i + 3]
        if tri == "tch":              out.append("CH"); i += 3; continue
        if pair == "ch":              out.append("CH"); i += 2; continue
        if pair == "sh":              out.append("SH"); i += 2; continue
        if pair == "th":              out.append("TH"); i += 2; continue
        if pair == "ph":              out.append("F");  i += 2; continue
        if pair == "wh":              out.append("W");  i += 2; continue
        if pair == "ck":              out.append("K");  i += 2; continue
        if pair == "ng":              out.append("NG"); i += 2; continue
        if pair == "qu":              out += ["K", "W"]; i += 2; continue
        if pair == "gh":              i += 2; continue           # usually silent
        if pair in ("ee", "ea", "ie"): out.append("IY"); i += 2; continue
        if pair == "oo":              out.append("UW"); i += 2; continue
        if pair in ("ou", "ow"):      out.append("AW"); i += 2; continue
        if pair == "oa":              out.append("OW"); i += 2; continue
        if pair in ("ai", "ay"):      out.append("EY"); i += 2; continue
        if pair in ("oy", "oi"):      out.append("OY"); i += 2; continue
        if pair in ("au", "aw"):      out.append("AO"); i += 2; continue
        if c == "e" and i == n - 1 and out:                      # silent final 'e'
            i += 1; continue
        if c in "aeiou":
            magic_e = (i + 2 < n and w[i + 1] not in "aeiou"
                       and w[i + 2] == "e" and i + 2 == n - 1)
            short = {"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH"}[c]
            long_ = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW"}[c]
            out.append(long_ if magic_e else short); i += 1; continue
        if c == "y":
            out.append("Y" if i == 0 else ("IY" if i == n - 1 else "IH"))
            i += 1; continue
        if c == "c":  out.append("S" if nxt in "eiy" else "K"); i += 1; continue
        if c == "g":  out.append("JH" if nxt in "eiy" else "G"); i += 1; continue
        if c == "x":  out += ["K", "S"]; i += 1; continue
        single = {"b": "B", "d": "D", "f": "F", "h": "HH", "j": "JH", "k": "K",
                  "l": "L", "m": "M", "n": "N", "p": "P", "r": "R", "s": "S",
                  "t": "T", "v": "V", "w": "W", "z": "Z"}
        if c in single: out.append(single[c]); i += 1; continue
        i += 1  # skip anything unrecognised
    # collapse doubled phones ("ll" -> L)
    res = []
    for p in out:
        if not res or res[-1] != p:
            res.append(p)
    return res


def text_to_phones(text):
    phones, first = [], True
    for tok in re.findall(r"[a-zA-Z']+|[.,!?;:]", text):
        if tok in ".!?":      phones.append("sil"); first = True; continue
        if tok in ",;:":      phones.append("sp");  continue
        if not first:         phones.append("sp")
        first = False
        w = tok.lower().strip("'")
        phones += EXCEPTIONS.get(w) or word_to_phones(w)
    return phones


# ----------------------------------------------------------------------------
# Synthesis: formant targets -> gliding sine tones
# ----------------------------------------------------------------------------

FIXED_HIGH = [3500, 4500, 5500]               # filler formants above F3
TONE_AMP = [1.0, 0.6, 0.35, 0.18, 0.10, 0.06]  # relative loudness per tone


def synthesise(phones, sr=44100, rate=1.0, n_formants=5):
    """Build per-sample formant tracks from phoneme targets and render them as
    a stack of continuous-phase sine waves. Returns (combined, stems, sr)."""
    # control points: one per steady phone (at its centre), two per diphthong
    ct, cf, ca = [], [], []   # times (s), F1/F2/F3 vectors, amplitude
    t = 0.0
    for ph in phones:
        dur = (dur_of(ph) / 1000.0) / max(rate, 1e-6)
        tgts, amp = targets_of(ph), amp_of(ph)
        if len(tgts) == 1:
            ct.append(t + dur / 2); cf.append(tgts[0]); ca.append(amp)
        else:
            ct.append(t + dur * 0.33); cf.append(tgts[0]); ca.append(amp)
            ct.append(t + dur * 0.67); cf.append(tgts[1]); ca.append(amp)
        t += dur
    total = max(t, 0.05)

    # pad with silent control points at the ends for a click-free fade in/out
    ct = [0.0] + ct + [total]
    cf = [cf[0]] + cf + [cf[-1]]
    ca = [0.0] + ca + [0.0]
    ct = np.array(ct); cf = np.array(cf, dtype=float); ca = np.array(ca)

    N = int(total * sr)
    ts = np.arange(N) / sr
    A = np.interp(ts, ct, ca)                         # energy envelope

    # full set of formant frequency tracks (F1..F3 interpolated, higher fixed)
    tracks = [np.interp(ts, ct, cf[:, i]) for i in range(3)]
    hi = list(FIXED_HIGH)
    while len(tracks) < n_formants and hi:
        tracks.append(np.full(N, hi.pop(0)))
    tracks = tracks[:n_formants]

    # each tone is a sine whose phase integrates its (gliding) frequency
    stems = np.zeros((len(tracks), N))
    for i, f in enumerate(tracks):
        phase = 2 * np.pi * np.cumsum(f) / sr
        stems[i] = TONE_AMP[i] * np.sin(phase) * A
    combined = stems.sum(axis=0)

    scale = 0.5 / max(np.max(np.abs(combined)), 1e-9)   # leave headroom
    return combined * scale, stems * scale, sr


def _write_wav(path, sr, x):
    scipy.io.wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767.0).astype(np.int16))
    print(f"Wrote {path} ({len(x) / sr:.2f} s)")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Estimate formants from text and synthesise sinewave speech.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Text to speak (rough built-in letter-to-sound).")
    g.add_argument("--phonemes", help="ARPAbet phonemes, space separated, e.g. 'HH AH L OW'.")
    parser.add_argument("-o", "--output", default="text_sws.wav", help="Output WAV. Default text_sws.wav.")
    parser.add_argument("--samplerate", type=int, default=44100, help="Sample rate. Default 44100.")
    parser.add_argument("--rate", type=float, default=1.0, help="Speaking rate multiplier (>1 faster). Default 1.0.")
    parser.add_argument("--formants", type=int, default=5, help="Number of sine tones (formants) to stack. Default 5.")
    parser.add_argument("--separate", action="store_true", help="Also write each formant to its own <output>_toneN.wav.")
    parser.add_argument("--play", action="store_true", help="Play the result with afplay after writing (macOS).")
    args = parser.parse_args(argv[1:])

    phones = (args.phonemes.split() if args.phonemes
              else text_to_phones(args.text))
    if not phones:
        print("No phonemes produced."); sys.exit(1)
    print("Phonemes:", " ".join(phones))

    combined, stems, sr = synthesise(
        phones, sr=args.samplerate, rate=args.rate, n_formants=args.formants)

    _write_wav(args.output, sr, combined)
    if args.separate:
        base, ext = os.path.splitext(args.output)
        for tr in range(stems.shape[0]):
            _write_wav(f"{base}_tone{tr}{ext}", sr, stems[tr])

    if args.play:
        os.system(f"afplay {args.output}")


if __name__ == "__main__":
    main(sys.argv)
