## MIT License
##
## Overlay (sum) any number of WAV files into one.
##
## Handy for recombining the per-tone files produced by `--separate`: pick any
## subset of <name>_toneN.wav files and sum them back together. Overlaying all
## of them reproduces the combined output; overlaying a subset gives you just
## those sliding tones.

import sys
import argparse

import numpy as np
import scipy.io.wavfile

from sws import load_wave


def overlay(paths):
    """Sum any number of WAV files sample-for-sample.

    Files are expected to share a sample rate (and normally the same length).
    Shorter files are zero-padded to the longest. Returns (mixed, sr, lengths)
    where `mixed` is a float array in roughly [-1, 1].
    """
    waves, srs = [], []
    for p in paths:
        wave, sr = load_wave(p)   # normalised to [-1, 1], mono
        waves.append(wave)
        srs.append(sr)

    if len(set(srs)) > 1:
        raise ValueError(f"sample rates differ between inputs: {srs}")

    n = max(len(w) for w in waves)
    mixed = np.zeros(n)
    for w in waves:
        mixed[: len(w)] += w
    return mixed, srs[0], [len(w) for w in waves]


def main(argv):
    parser = argparse.ArgumentParser(
        description="Overlay (sum) any number of WAV files into one.")
    parser.add_argument("inputs", nargs="+", help="Input WAV files to overlay.")
    parser.add_argument("-o", "--output", default="overlay.wav",
                        help="Output WAV path. Default overlay.wav.")
    parser.add_argument("--normalize", action="store_true",
                        help="Peak-normalise the mix to 0.5 (default: keep the summed levels).")
    args = parser.parse_args(argv[1:])

    mixed, sr, lengths = overlay(args.inputs)
    if len(set(lengths)) > 1:
        print(f"Warning: input lengths differ {lengths}; zero-padded to {max(lengths)}.")

    if args.normalize:
        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = 0.5 * mixed / peak

    scipy.io.wavfile.write(args.output, sr,
                           (np.clip(mixed, -1.0, 1.0) * 32767.0).astype(np.int16))
    print(f"Overlaid {len(args.inputs)} file(s) -> {args.output}")


if __name__ == "__main__":
    main(sys.argv)
