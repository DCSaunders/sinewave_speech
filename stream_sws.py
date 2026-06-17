## MIT License
##
## Real-time streaming sinewave speech.
##
## Captures audio from the microphone, converts it to sinewave speech (or buzz /
## noise vocoded speech) block-by-block, and plays the result back through the
## default output device (e.g. your headphones). By default it also records the
## raw microphone input and the synthesised output to WAV files.
##
## This is the streaming companion to the offline sws.py. The heavy DSP math is
## imported and reused from sws.py; this file only adds the causal/stateful
## building blocks that streaming needs (a causal bandpass+decimator and a
## causal normaliser) plus a block-driven vocoder and a sounddevice driver.

import sys
import argparse

import numpy as np
import scipy.signal
import scipy.io.wavfile
from scipy.signal.windows import hann

import sounddevice as sd

# Reused as-is from the offline implementation.
from sws import lpc, lpc_to_lsp, formants_from_lsp


# ----------------------------------------------------------------------------
# Streaming building blocks
# ----------------------------------------------------------------------------


class StreamingBandpass:
    """Causal replacement for sws.bp_filter_and_decimate.

    The offline code uses zero-phase ``filtfilt`` (non-causal, needs the whole
    signal). Here we apply the same pre-emphasis and 4th-order Butterworth
    bandpass with single-direction ``lfilter`` and persist the filter state
    (``zi``) across blocks, then decimate by slicing.

    Requires blocksize % decimate == 0 so that the ``[::decimate]`` grid stays
    phase-continuous across block boundaries.
    """

    def __init__(self, low, high, fs, decimate):
        self.decimate = decimate
        # pre-emphasis filter (matches sws.bp_filter_and_decimate)
        self.b_pre = np.array([1.0])
        self.a_pre = np.array([1.0, 0.75])
        self.zi_pre = np.zeros(max(len(self.a_pre), len(self.b_pre)) - 1)
        # bandpass
        self.b, self.a = scipy.signal.butter(4, Wn=[low, high], btype="band", fs=fs)
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)

    def process(self, x):
        x, self.zi_pre = scipy.signal.lfilter(self.b_pre, self.a_pre, x, zi=self.zi_pre)
        x, self.zi = scipy.signal.lfilter(self.b, self.a, x, zi=self.zi)
        return x[:: self.decimate]


class StreamingUpsampler:
    """Stateful polyphase upsampler (used for buzz/noise output only).

    Inserts ``factor-1`` zeros between samples and low-pass filters with a FIR
    whose state is carried across blocks, so there are no per-chunk edge
    transients (unlike calling scipy.signal.resample_poly on each chunk).
    """

    def __init__(self, factor, numtaps=64):
        self.factor = factor
        taps = numtaps * factor + 1
        self.b = scipy.signal.firwin(taps, 1.0 / factor)
        self.zi = np.zeros(len(self.b) - 1)

    def process(self, x):
        if len(x) == 0:
            return np.zeros(0)
        up = np.zeros(len(x) * self.factor)
        up[:: self.factor] = x * self.factor
        y, self.zi = scipy.signal.lfilter(self.b, 1.0, up, zi=self.zi)
        return y


class StreamingNormalizer:
    """Causal AGC replacement for sws.normalize (output only).

    Peak follower with fast attack / slow release; gain is smoothed to avoid
    zippering and the output is hard-limited to keep headphones safe. A maximum
    gain prevents silence from being amplified into a roar.
    """

    def __init__(self, fs, target=0.3, attack_ms=8.0, release_ms=300.0,
                 max_gain=10.0, eps=1e-6):
        self.target = target
        self.max_gain = max_gain
        self.eps = eps
        self.a_att = np.exp(-1.0 / (fs * attack_ms / 1000.0))
        self.a_rel = np.exp(-1.0 / (fs * release_ms / 1000.0))
        self.a_gain = np.exp(-1.0 / (fs * 0.02))  # 20 ms gain smoothing
        self.peak = eps
        self.gain = target

    def process(self, x):
        out = np.empty_like(x)
        peak, gain = self.peak, self.gain
        tgt, eps, mg = self.target, self.eps, self.max_gain
        a_att, a_rel, a_g = self.a_att, self.a_rel, self.a_gain
        for i in range(len(x)):
            ax = abs(x[i])
            a = a_att if ax > peak else a_rel
            peak = a * peak + (1.0 - a) * ax
            target_gain = min(tgt / max(peak, eps), mg)
            gain = a_g * gain + (1.0 - a_g) * target_gain
            y = x[i] * gain
            out[i] = 0.99 if y > 0.99 else (-0.99 if y < -0.99 else y)
        self.peak, self.gain = peak, gain
        return out


class StreamingVocoder:
    """Block-driven LPC analysis + resynthesis, reusing sws.py frame math.

    Analysis is always done on the decimated signal. Sine mode is synthesised
    directly at the full sample rate (phase-continuous, no upsampling needed).
    Buzz/noise modes are synthesised at the decimated rate (reusing the
    lpc_vocode per-frame logic) and upsampled with a stateful resampler.
    """

    def __init__(self, mode, frame_len, order, fs, decimate, overlap, bw_amp,
                 buzz=None, residual=0.0):
        self.mode = mode  # 'sine' | 'buzz' | 'noise'
        self.frame_len = frame_len  # decimated samples (== --window)
        self.order = order
        self.fs = fs
        self.decimate = decimate
        self.sr_dec = fs / decimate
        self.hop = max(1, int(frame_len * overlap))
        self.bw_amp = bw_amp
        self.buzz = buzz
        self.residual = residual

        # sine renders at full rate, buzz/noise at decimated rate
        self.acc_up = decimate if mode == "sine" else 1
        self.win = hann(frame_len * self.acc_up)
        self.upsampler = (StreamingUpsampler(decimate)
                          if (self.acc_up == 1 and decimate > 1) else None)

        self.inbuf = np.zeros(0)   # decimated input
        self.consumed = 0          # absolute decimated index of next frame start
        self.acc = np.zeros(0)     # overlap-add accumulator (acc-rate samples)
        self.acc_base = 0          # absolute acc-rate index of acc[0]
        self.emitted_upto = 0      # absolute acc-rate index already emitted

    # -- per-frame synthesis -------------------------------------------------

    def _synth_sine(self, freqs, bws, rms, s):
        n = self.frame_len * self.decimate
        t = np.arange(n)
        g_full = s * self.decimate
        fr = self.fs / (2 * np.pi)
        syn = np.zeros(n)
        for band in range(len(freqs)):
            f, bw = freqs[band], bws[band]
            if f > 90.0:
                syn += np.sin(f * (t + g_full) / fr) * np.exp(bw / self.bw_amp)
        return syn * rms

    def _synth_vocode(self, a, e, rms, s):
        n = self.frame_len
        if self.mode == "noise":
            carrier = np.random.normal(0, 1, n)
            error = 0.0
        else:  # buzz
            t = np.arange(n)
            f = float(self.buzz)
            # ModFM k from base frequency (see sws.main)
            N = 12 * np.log2(f / 440.0) + 69
            k = np.exp(-0.1513 * N) + 15.927
            kk = k * k
            # global phase so the buzz doesn't reset (click) every frame
            phase = f * 2 * np.pi * ((t + s) / self.sr_dec)
            carrier = np.cos(phase) * np.exp(np.cos(phase) * kk - kk)
            error = e * float(self.residual)

        # LPC as an IIR filter applied to the carrier, stateless per frame
        # (matches sws.lpc_vocode; carrying zi across changing `a` can blow up)
        vocoded = scipy.signal.lfilter([1.0], a, carrier) * (1.0 - error)
        if error > 0:
            residual = scipy.signal.lfilter([1.0], a, np.random.normal(0, 1, n)) * error
        else:
            residual = 0.0

        # match RMS of the analysis frame
        voc_amp = 1e-5 + np.sqrt(np.mean(vocoded ** 2))
        vocoded = vocoded * (rms / voc_amp)
        out = vocoded + residual

        # stability guard: an unstable `a` can ring loudly; drop that frame
        m = np.max(np.abs(out)) if len(out) else 0.0
        if not np.isfinite(m) or m > 50.0:
            out = np.zeros(n)
        return out

    def _frame(self, s):
        frame = self.inbuf[: self.frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        # silent frame: autocorrelation is all-zero and LPC is undefined, so
        # emit a silent (windowed) frame rather than letting Levinson reject it
        if rms < 1e-9:
            return np.zeros(self.frame_len * self.acc_up)
        a, e, _ = lpc(frame, self.order)
        if self.mode == "sine":
            lsp = lpc_to_lsp(a)
            freqs, bws = formants_from_lsp(lsp[None, :, :], self.sr_dec)
            syn = self._synth_sine(freqs[0], bws[0], rms, s)
        else:
            syn = self._synth_vocode(a, e, rms, s)
        # Hann window for click-free overlap-add (matches sws.py)
        return syn * self.win

    # -- overlap-add accumulator ---------------------------------------------

    def _add(self, start, syn):
        end = start + len(syn)
        need = end - self.acc_base
        if need > len(self.acc):
            self.acc = np.concatenate([self.acc, np.zeros(need - len(self.acc))])
        off = start - self.acc_base
        self.acc[off:off + len(syn)] += syn

    def _emit(self, flush_to):
        a = self.emitted_upto - self.acc_base
        b = min(flush_to - self.acc_base, len(self.acc))
        if b <= a:
            return np.zeros(0)
        out = self.acc[a:b].copy()
        self.emitted_upto = self.acc_base + b
        # trim the front of the accumulator to bound memory
        self.acc = self.acc[b:]
        self.acc_base = self.emitted_upto
        return out

    # -- public API ----------------------------------------------------------

    def process(self, dec_block):
        self.inbuf = np.concatenate([self.inbuf, dec_block])
        chunks = []
        while len(self.inbuf) >= self.frame_len:
            s = self.consumed
            syn = self._frame(s)
            self._add(s * self.acc_up, syn)
            self.inbuf = self.inbuf[self.hop:]
            self.consumed += self.hop
            # samples before (s + hop) receive no further contributions
            emitted = self._emit(self.consumed * self.acc_up)
            if len(emitted):
                chunks.append(emitted)
        out = np.concatenate(chunks) if chunks else np.zeros(0)
        return self.upsampler.process(out) if self.upsampler else out

    def flush(self):
        """Emit any remaining buffered output (call at shutdown)."""
        out = self._emit(self.acc_base + len(self.acc))
        return self.upsampler.process(out) if self.upsampler else out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def run(args):
    fs = args.samplerate
    B = args.blocksize
    if B % args.decimate != 0:
        B = ((B // args.decimate) + 1) * args.decimate
        print(f"Adjusted blocksize up to {B} (must be a multiple of --decimate)")

    mode = "noise" if args.noise else ("buzz" if args.buzz else "sine")
    order = 2 * args.order + 2

    bandpass = StreamingBandpass(args.low, args.high, fs, args.decimate)
    vocoder = StreamingVocoder(
        mode, frame_len=args.window, order=order, fs=fs, decimate=args.decimate,
        overlap=args.overlap, bw_amp=args.bw_amp, buzz=args.buzz, residual=args.residual,
    )
    normalizer = StreamingNormalizer(fs, target=args.target)

    mic_rec, sws_rec = [], []
    outq = np.zeros(0)

    print(f"Streaming: mode={mode}, fs={fs}, blocksize={B} "
          f"(~{1000.0 * B / fs:.0f} ms/block). Ctrl-C to stop.")
    if args.duration:
        print(f"Running for {args.duration:.1f} s.")
    if args.write:
        print(f"● REC ON  -> {args.mic_out} (mic) + {args.sws_out} (sinewave)")
    else:
        print("○ recording OFF (--no-write): monitoring only, no files written")

    stream = sd.Stream(samplerate=fs, blocksize=B, channels=1, dtype="float32",
                       device=args.device, latency="high")
    target_frames = int(args.duration * fs) if args.duration else None
    frames_done = 0
    try:
        with stream:
            while target_frames is None or frames_done < target_frames:
                indata, _ = stream.read(B)
                mono = indata[:, 0].astype(np.float64)

                dec = bandpass.process(mono) * args.in_gain
                out = vocoder.process(dec)
                outq = np.concatenate([outq, out])

                if len(outq) >= B:
                    play, outq = outq[:B], outq[B:]
                else:
                    play = np.concatenate([outq, np.zeros(B - len(outq))])
                    outq = np.zeros(0)
                play = normalizer.process(play)

                stream.write(play.reshape(-1, 1).astype("float32"))

                if args.write:
                    mic_rec.append(mono.copy())
                    sws_rec.append(play.copy())
                frames_done += B

                # live status line, overwritten in place each block
                elapsed = frames_done / fs
                tag = "● REC" if args.write else "○ live"
                print(f"\r{tag}  {elapsed:6.1f}s   (Ctrl-C to stop)",
                      end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()  # finish the in-place status line
        if args.write:
            print("Stopped. Writing recordings...")
            tail = normalizer.process(vocoder.flush())
            if len(tail):
                sws_rec.append(tail)
            _write_wav(args.mic_out, fs, np.concatenate(mic_rec) if mic_rec else np.zeros(0))
            _write_wav(args.sws_out, fs, np.concatenate(sws_rec) if sws_rec else np.zeros(0))
        else:
            print("Stopped.")


def _write_wav(path, fs, x):
    scipy.io.wavfile.write(path, fs, (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16))
    print(f"Wrote {path} ({len(x) / fs:.1f} s)")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Real-time microphone -> sinewave speech, with optional recording.")
    # mirrors sws.py
    parser.add_argument("--low", type=float, default=250, help="Lowpass filter cutoff. Default 250.")
    parser.add_argument("--high", type=float, default=3400, help="Highpass filter cutoff. Default 3400.")
    parser.add_argument("--order", "-o", type=int, default=4, help="Number of components in synthesis. Default 4.")
    parser.add_argument("--bw_amp", type=float, default=60, help="Amplitude scaling by bandwidth. Default 60.")
    parser.add_argument("--decimate", "-d", type=int, default=8, help="Sample rate decimation before analysis. Default 8.")
    parser.add_argument("--window", "-w", type=int, default=200, help="LPC window size (decimated samples). Default 200.")
    parser.add_argument("--overlap", "-l", type=float, default=0.25, help="Window overlap as a fraction. Default 0.25.")
    parser.add_argument("--sine", "-s", action="store_true", default=True, help="Resynthesise using sinewave speech (default).")
    parser.add_argument("--buzz", "-b", default=None, help="Resynthesise using a buzz at the given frequency (Hz).")
    parser.add_argument("--residual", type=float, default=0.0, help="Residual noise added in buzz mode. Default 0.0.")
    parser.add_argument("--noise", "-n", action="store_true", help="Resynthesise using filtered white noise.")
    # streaming-specific
    parser.add_argument("--samplerate", type=int, default=44100, help="Audio device sample rate. Default 44100.")
    parser.add_argument("--blocksize", type=int, default=2048, help="Audio block size (rounded up to a multiple of --decimate). Default 2048.")
    parser.add_argument("--device", default=None, help="Input,output device spec for sounddevice (e.g. '1,3').")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to run. Default: until Ctrl-C.")
    parser.add_argument("--in-gain", dest="in_gain", type=float, default=1.0, help="Fixed input gain applied before analysis. Default 1.0.")
    parser.add_argument("--target", type=float, default=0.3, help="Output normaliser target peak level. Default 0.3.")
    parser.add_argument("--no-write", dest="write", action="store_false", default=True, help="Do not record WAV files (recording is on by default).")
    parser.add_argument("--mic-out", default="stream_mic.wav", help="Mic recording output path. Default stream_mic.wav.")
    parser.add_argument("--sws-out", default="stream_sws.wav", help="Sinewave-speech recording output path. Default stream_sws.wav.")

    args = parser.parse_args(argv[1:])

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.device is not None and "," in args.device:
        a, b = args.device.split(",")
        args.device = (int(a) if a.strip() else None, int(b) if b.strip() else None)

    run(args)


if __name__ == "__main__":
    main(sys.argv)
