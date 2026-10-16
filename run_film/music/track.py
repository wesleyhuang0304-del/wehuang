"""120 BPM hybrid sports-anthem score: synthesized drums/bass/FX (numpy) + orchestral stems (FluidSynth GM)."""
import os, sys, subprocess
import numpy as np
import mido
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from timeline import BEAT, DUR, CUTS, IMPACTS, GAPS, heartbeats  # noqa: E402

SR = 48000
N = int(SR * (DUR + 1.0))
SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
BAR = 4 * BEAT
rng = np.random.default_rng(2026)


def bar(i, beat=0.0):
    return i * BAR + beat * BEAT


# ---------------------------------------------------------------- numpy synthesis
def tt(dur):
    return np.arange(int(dur * SR)) / SR


def bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, 'high', fs=SR, output='sos'), x)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, 'low', fs=SR, output='sos'), x)


def noise(dur):
    return rng.standard_normal(int(dur * SR))


def kick(big=False):
    t = tt(0.9 if big else 0.5)
    f = 44 + 120 * np.exp(-t / 0.035) + 40 * np.exp(-t / 0.006)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / (0.42 if big else 0.26))
    click = hp(noise(0.004), 2500) * 0.5
    body[:len(click)] += click
    return np.tanh(body * 2.2) * 0.9


def clap():
    t = tt(0.35)
    env = np.zeros_like(t)
    for d in (0, 0.011, 0.022):
        env += (t >= d) * np.exp(-np.clip(t - d, 0, None) / 0.008)
    env += (t >= 0.03) * np.exp(-np.clip(t - 0.03, 0, None) / 0.11)
    return bp(noise(0.35), 900, 4200) * env * 0.9


def snare():
    t = tt(0.3)
    tone = np.sin(2 * np.pi * (185 + 60 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.07)
    nz = bp(noise(0.3), 1800, 9000) * np.exp(-t / 0.13)
    return (tone * 0.6 + nz * 0.9) * 0.8


def hat(open_=False):
    d = 0.3 if open_ else 0.06
    t = tt(d)
    metal = sum(signal.square(2 * np.pi * f * t) for f in (3140, 4470, 5210, 6830, 8120)) / 5
    x = hp(metal * 0.5 + noise(d) * 0.6, 7000)
    return x * np.exp(-t / (0.09 if open_ else 0.014)) * 0.45


def shaker():
    t = tt(0.08)
    return bp(noise(0.08), 4000, 12000) * np.sin(np.pi * t / 0.08) ** 2 * 0.25


def taiko(pitch=1.0):
    t = tt(1.1)
    f = (62 + 70 * np.exp(-t / 0.05)) * pitch
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.38)
    skin = lp(noise(1.1), 900) * np.exp(-t / 0.05) * 0.8
    return np.tanh((body + skin) * 1.6) * 0.85


def tom(pitch):
    t = tt(0.5)
    f = pitch * (1 + 0.6 * np.exp(-t / 0.03))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18) * 0.7


def crash(dur=2.4):
    t = tt(dur)
    metal = sum(np.sin(2 * np.pi * f * t + rng.random() * 6) for f in rng.uniform(3000, 11000, 40)) / 40
    x = hp(noise(dur) * 0.7 + metal, 3500)
    return x * np.exp(-t / (dur * 0.35)) * 0.5


def impact():
    t = tt(3.0)
    f = 30 + 70 * np.exp(-t / 0.08)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)
    thud = lp(noise(3.0), 400) * np.exp(-t / 0.12)
    return np.tanh((boom * 1.4 + thud * 0.8) * 1.5) * 0.95


def riser(dur, f0=300, f1=9000):
    """Noise + pitch sweep that rises over `dur` seconds (rendered in filtered blocks)."""
    n = int(dur * SR)
    out = np.zeros(n)
    nz = noise(dur)
    blk = 2048
    for s in range(0, n, blk):
        u = s / n
        fc = f0 * (f1 / f0) ** u
        seg = nz[s:s + blk]
        out[s:s + len(seg)] = bp(seg, max(40, fc * 0.6), min(SR / 2 - 100, fc * 1.6), 1)
    t = tt(dur)
    tone_f = 110 * 2 ** (3 * t / dur)
    tone = signal.sawtooth(2 * np.pi * np.cumsum(tone_f) / SR) * 0.15
    env = (t / dur) ** 2.2
    return (out * 0.8 + lp(tone, 3000)) * env


def reverse_swell(dur):
    x = crash(dur)[::-1]
    return x * 1.3


def whoosh(dur=0.45):
    t = tt(dur)
    u = t / dur
    env = np.sin(np.pi * u) ** 2
    x = np.zeros_like(t)
    nz = noise(dur)
    blk = 1024
    for s in range(0, len(t), blk):
        fc = 600 + 5000 * np.sin(np.pi * s / len(t))
        seg = nz[s:s + blk]
        x[s:s + len(seg)] = bp(seg, fc * 0.5, fc * 1.5, 1)
    return x * env * 0.55


def heartbeat():
    t = tt(0.6)
    lub = np.sin(2 * np.pi * 52 * t) * np.exp(-t / 0.06)
    t2 = np.clip(t - 0.22, 0, None)
    dub = np.sin(2 * np.pi * 46 * t2) * np.exp(-t2 / 0.07) * (t >= 0.22) * 0.7
    return np.tanh((lub + dub) * 2.5) * 0.8


def subdrop(dur=1.2):
    t = tt(dur)
    f = 30 + 55 * np.exp(-t / 0.35)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.6) * 0.9


def tick():
    t = tt(0.03)
    return hp(noise(0.03), 5000) * np.exp(-t / 0.004) * 0.4


def crowd(dur):
    """Stadium roar: band-limited noise with slow swells and vowel-like formants."""
    nz = noise(dur)
    t = tt(dur)
    x = bp(nz, 250, 2600) * 0.6 + bp(noise(dur), 700, 1100) * 0.4 + bp(noise(dur), 1800, 2600) * 0.25
    am = 0.75 + 0.25 * np.sin(2 * np.pi * 0.37 * t) * np.sin(2 * np.pi * 0.11 * t + 1)
    return x * am * 0.35


NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
        'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def midi(n):
    name, octv = n[:-1], int(n[-1])
    return 12 * (octv + 1) + NOTE[name]


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def saw_voice(f, dur, cutoff=1200, detune=(-0.12, 0, 0.12), env_a=0.005, env_r=0.08):
    t = tt(dur + env_r)
    x = sum(signal.sawtooth(2 * np.pi * f * 2 ** (d / 12) * t + rng.random() * 6) for d in detune) / len(detune)
    env = np.minimum(1, t / env_a) * np.where(t < dur, 1.0, np.exp(-(t - dur) / max(env_r / 4, 1e-3)))
    return lp(x, cutoff) * env


# chord roots per bar (bar = 2 s); sections follow timeline.SECTIONS
PROG = {}
for b0, seq in [(0, ['D', 'D', 'D', 'D']),                          # intro 0-8
                (4, ['Bb', 'C', 'D', 'D', 'Bb', 'C']),              # heritage 8-20
                (10, ['D', 'Bb', 'F', 'C', 'A']),                   # city 20-30
                (15, ['D', 'D', 'Bb', 'F', 'C', 'Bb', 'A']),        # drop 30-44
                (22, ['D', 'Bb', 'F', 'C', 'A']),                   # chase 44-54
                (27, ['DM', 'DM', 'DM'])]:                          # end card 54-60
    for k, r in enumerate(seq):
        PROG[b0 + k] = r
CHORDS = {'D': ['D', 'F', 'A'], 'Bb': ['Bb', 'D', 'F'], 'F': ['F', 'A', 'C'], 'C': ['C', 'E', 'G'],
          'A': ['A', 'C#', 'E'], 'DM': ['D', 'F#', 'A']}


class Bus:
    def __init__(self, pan=0.0):
        self.x = np.zeros(N)
        self.pan = pan

    def add(self, t, s, g=1.0):
        i = int(round(t * SR))
        if i >= N:
            return
        s = s[:N - i]
        self.x[i:i + len(s)] += s * g


def build_drums():
    kickb, clapb, snb, hatL, hatR, perc, fx, low = Bus(), Bus(), Bus(), Bus(-0.45), Bus(0.45), Bus(), Bus(), Bus()
    K, KB, C, S, HC, HO, SH = kick(), kick(True), clap(), snare(), hat(), hat(True), shaker()
    kicks = []

    def K4(b, big=False):
        for q in range(4):
            kickb.add(bar(b, q), KB if (big and q == 0) else K, 1.0)
            kicks.append(bar(b, q))

    def roll(t0, t1, g0=0.35, g1=0.9):
        t = t0
        while t < t1 - 0.03:
            u = (t - t0) / (t1 - t0)
            snb.add(t, S, g0 + (g1 - g0) * u)
            t += BEAT / (2 if u < 0.3 else 4 if u < 0.65 else 8)

    def fill(b, gain=0.7):
        for k in range(8):
            perc.add(bar(b, 2 + k * 0.25), tom([200, 170, 140, 115, 200, 170, 140, 115][k]), gain)

    # intro 0-8: accelerating heartbeat, clock ticks, reverse swell + riser into the 8.0 impact
    for t in heartbeats():
        low.add(t, heartbeat(), 0.85)
        kicks.append(t)
    for e in range(8, 31):
        hatR.add(e * BEAT / 2, tick(), 0.45 + 0.4 * (e % 2 == 0))
    fx.add(5.0, reverse_swell(3.0), 0.55)
    fx.add(5.0, riser(3.0, 200, 6000), 0.35)

    # heritage 8-20: taiko ensemble, shaker, light kick; ECG heartbeat accents bar 6; build bars 8-9
    for b in range(4, 10):
        for q in (0, 2.5):
            kickb.add(bar(b, q), K, 0.8)
            kicks.append(bar(b, q))
        if b >= 5:
            clapb.add(bar(b, 1), C, 0.5)
            clapb.add(bar(b, 3), C, 0.5)
        for e in range(8):
            perc.add(bar(b, e / 2 + 0.25), SH, 0.8)
        for q in (0, 0.75, 1.5):
            perc.add(bar(b, q), taiko(0.85), 0.7)
    for k in range(40):  # 1906 -> 2026 counter
        hatR.add(8.05 + k * 0.039, tick(), 0.55)
    for q in (0, 1, 2, 3):
        low.add(bar(6, q), heartbeat(), 0.75)
    for k, tm in enumerate([16.0, 16.5, 17.0, 17.5]):
        perc.add(tm, taiko(1.1 - 0.05 * k), 0.85)
    for e in range(8):
        (hatL if e % 2 else hatR).add(bar(8, e / 2), HC, 0.4)
        (hatL if e % 2 else hatR).add(bar(9, e / 2), HC, 0.5)
    fill(9, 0.6)
    roll(18.5, 19.95, 0.3, 0.8)
    fx.add(17.0, riser(3.0, 250, 9000), 0.6)

    # city 20-30: half-time groove, claps from bar 11, toms bar 13, roll + riser + gap into the drop
    for b in range(10, 15):
        for q in (0, 2):
            kickb.add(bar(b, q), K, 0.95)
            kicks.append(bar(b, q))
        for e in range(8):
            (hatL if e % 2 else hatR).add(bar(b, e / 2), HC, 0.45 + 0.25 * (e % 2))
        if b >= 11:
            clapb.add(bar(b, 1), C, 0.6)
            clapb.add(bar(b, 3), C, 0.6)
    fill(13, 0.8)
    roll(28.0, 29.75, 0.35, 0.9)
    for k in range(40):  # 0 -> 42.195 counter
        hatR.add(28.6 + k * 0.025, tick(), 0.5)
    fx.add(26.0, riser(3.75, 250, 11000), 0.75)

    # drop 30-44: four-on-the-floor, taiko, end fill + roll + gap
    def groove(b0, b1, heavy):
        for b in range(b0, b1):
            K4(b, big=(b == b0) or heavy and (b - b0) % 2 == 0)
            clapb.add(bar(b, 1), C, 0.9)
            clapb.add(bar(b, 3), C, 0.9)
            snb.add(bar(b, 1), S, 0.45)
            snb.add(bar(b, 3), S, 0.45)
            for s16 in range(16):
                q = s16 / 4
                if s16 % 4 == 2:
                    (hatL if b % 2 else hatR).add(bar(b, q), HO, 0.5)
                else:
                    (hatL if s16 % 2 else hatR).add(bar(b, q), HC, 0.32 + 0.22 * (s16 % 4 == 0))
            for q in (0, 1.5, 2.5) if heavy else (0, 2.5):
                perc.add(bar(b, q), taiko(1.0 if q == 0 else 1.25), 0.75)

    groove(15, 22, True)
    fill(21, 0.7)
    roll(43.0, 43.75, 0.5, 0.95)
    fx.add(41.0, riser(2.75, 250, 12000), 0.75)

    # chase 44-54: anthem groove, tape-break crash, chant hits bar 26
    groove(22, 26, True)
    crash_t = 51.1
    fx.add(crash_t, crash(2.4), 0.55)
    for q in (0, 1, 2, 3):
        kickb.add(bar(26, q), KB, 1.0)
        kicks.append(bar(26, q))
        clapb.add(bar(26, q), C, 1.0)
        perc.add(bar(26, q), taiko(1.0), 0.9)
    roll(53.0, 53.95, 0.5, 0.9)
    fx.add(52.0, riser(2.0, 300, 9000), 0.55)

    # impacts, crashes, sub drops on every big hit; whoosh into every other cut
    for tm in IMPACTS:
        low.add(tm, impact(), 1.0)
        low.add(tm, subdrop(), 0.7)
        fx.add(tm, crash(3.0), 0.7)
        kicks.append(tm)
    for tm in CUTS[1:]:
        if tm not in IMPACTS:
            fx.add(tm - 0.2, whoosh(0.4), 0.45 if tm < 30 else 0.55)
    fx.add(58.0, crash(2.0), 0.35)
    return dict(kick=kickb, clap=clapb, snare=snb, hatL=hatL, hatR=hatR, perc=perc, fx=fx, low=low), kicks


def build_bass():
    b = Bus()
    for bi, root in PROG.items():
        if bi < 4 or bi >= 27:
            continue
        f = hz(midi(root + '1'))
        if bi < 15 or bi == 26:
            for e in range(8):
                b.add(bar(bi, e / 2), saw_voice(f, BEAT / 2 * 0.85, 700), 0.5 if bi < 10 else 0.6)
        else:  # drop + chase: rolling 16ths, offbeat octave
            for s in range(16):
                ff = f * (2 if s % 4 == 2 else 1)
                b.add(bar(bi, s / 4), saw_voice(ff, BEAT / 4 * 0.8, 1100), 0.62)
    sub = Bus()
    for bi, root in PROG.items():
        if 4 <= bi < 27:
            t = tt(BAR)
            f = hz(midi(root + '1'))
            sub.add(bar(bi), np.sin(2 * np.pi * f * t) * np.minimum(1, t / 0.01) * np.minimum(1, (BAR - t) / 0.02), 0.5)
    t = tt(4.0)
    sub.add(54.0, np.sin(2 * np.pi * hz(midi('D1')) * t) * np.exp(-t / 1.8), 0.7)
    return b, sub


def build_lead():
    """Supersaw stabs for drop and chase (sidechained hard)."""
    L = Bus()
    for bi in range(15, 26):
        ch = CHORDS[PROG[bi]]
        for e in (0, 1.5, 3, 4.5, 6) if bi >= 22 else (0, 1.5, 3, 6):
            for n in ch:
                L.add(bar(bi, e / 2), saw_voice(hz(midi(n + '4')), BEAT * 0.35, 3200,
                                                detune=(-0.18, -0.07, 0, 0.07, 0.18)), 0.18)
    return L


# ---------------------------------------------------------------- orchestral stems (FluidSynth)
def render_midi(name, parts):
    """parts: list of (program, channel, [(t, dur, note, vel)])"""
    mf = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mf.tracks.append(tr)
    tr.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
    ev = []
    for prog, ch, notes in parts:
        ev.append((0, 0, mido.Message('program_change', program=prog, channel=ch)))
        ev.append((0, 0, mido.Message('control_change', control=91, value=70, channel=ch)))
        for t, d, n, v in notes:
            m = midi(n) if isinstance(n, str) else n
            ev.append((t, 2, mido.Message('note_on', note=m, velocity=int(v), channel=ch)))
            ev.append((t + d, 1, mido.Message('note_off', note=m, velocity=0, channel=ch)))
    ev.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for t, _, m in ev:
        tick_ = int(round(t / BEAT * 480))
        tr.append(m.copy(time=max(0, tick_ - last)))
        last = max(last, tick_)
    mid = f'{HERE}/{name}.mid'
    wav = f'{HERE}/{name}.wav'
    mf.save(mid)
    subprocess.run(['fluidsynth', '-ni', '-g', '0.32', '-r', str(SR), '-F', wav, SF2, mid], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32) / 32768.0
    out = np.zeros((N, 2), np.float32)
    out[:min(N, len(x))] = x[:N]
    return out


def chord_notes(root, octave, t, d, v, top=False):
    ns = [c + str(octave) for c in CHORDS[root]]
    if top:
        ns.append(CHORDS[root][0] + str(octave + 1))
    return [(t, d, n, v) for n in ns]


def orchestra():
    strings, ost, brass, choir, ethnic, piano, timp, hits = [], [], [], [], [], [], [], []
    # intro 0-8: swelling Dm pad + sparse piano
    strings += [(0.2, 7.6, n, 55) for n in ('D3', 'A3', 'F4')]
    for bi in range(0, 4):
        for e in range(8):
            piano.append((bar(bi, e / 2), BEAT * 1.5, ['D', 'A', 'F', 'A'][e % 4] + str(3 + (e % 4 == 2)),
                          50 + 6 * bi + 8 * (e == 0)))
    for b_, q, d, n in [(1, 0, 2, 'A5'), (1, 2, 2, 'F5'), (2, 0, 2, 'D5'), (2, 2, 2, 'F5'),
                        (3, 0, 1, 'E5'), (3, 1, 1, 'F5'), (3, 2, 1.8, 'A5')]:
        piano.append((bar(b_, q), d * BEAT, n, 84))

    # heritage 8-20: flute pentatonic melody + koto arpeggios over long strings
    mel = [(4, 0, 1, 'D5'), (4, 1, .5, 'F5'), (4, 1.5, .5, 'G5'), (4, 2, 2, 'A5'),
           (5, 0, 1, 'G5'), (5, 1, .5, 'A5'), (5, 1.5, .5, 'C6'), (5, 2, 2, 'D6'),
           (6, 0, .5, 'C6'), (6, .5, .5, 'A5'), (6, 1, 1, 'G5'), (6, 2, .5, 'F5'), (6, 2.5, .5, 'G5'),
           (6, 3, 1, 'A5'), (7, 0, 3.5, 'D5'),
           (8, 0, 1, 'F5'), (8, 1, 1, 'G5'), (8, 2, 2, 'A5'),
           (9, 0, 1, 'C6'), (9, 1, 1, 'D6'), (9, 2, 1.5, 'G5')]
    for b_, q, d, n in mel:
        ethnic.append((bar(b_, q), d * BEAT, n, 105))
    koto = []
    for bi in range(4, 10):
        ch = CHORDS[PROG[bi]]
        for e in range(8):
            koto.append((bar(bi, e / 2), BEAT, ch[[0, 1, 2, 1][e % 4]] + str(4 + (e % 4 == 2)), 90))

    # string ostinato 16ths: city (low), drop and chase (high)
    pat = [0, 2, 1, 2]
    for bi in range(10, 26):
        ch = CHORDS[PROG[bi]]
        oct_ = 3 if bi < 15 else 4
        for s in range(16):
            n = ch[pat[s % 4]] + str(oct_ + (s % 8 == 7))
            ost.append((bar(bi, s / 4), BEAT / 4 * 0.8, n, 70 + 30 * (s % 4 == 0) + (15 if bi >= 22 else 0)))
    for bi in range(4, 27):
        strings += chord_notes(PROG[bi], 3, bar(bi), BAR, 62 if bi < 10 else 75)

    # drop 30-44: stabs on the first bar, then the brass theme + choir over bars 16-21
    for e in (0, 1.5, 3, 6):
        brass += chord_notes(PROG[15], 4, bar(15, e / 2), BEAT * 0.4, 112)
    hits += [(bar(15), 0.5, n, 120) for n in ('D4', 'A4', 'D5')]
    theme = [  # (bar offset, beat, dur, note) over D Bb F C Bb A
        (0, 0, 1, 'D5'), (0, 1, .5, 'F5'), (0, 1.5, .5, 'G5'), (0, 2, 1.5, 'A5'), (0, 3.5, .5, 'C6'),
        (1, 0, 1, 'A5'), (1, 1, .5, 'G5'), (1, 1.5, .5, 'F5'), (1, 2, 2, 'G5'),
        (2, 0, 1, 'F5'), (2, 1, .5, 'G5'), (2, 1.5, .5, 'A5'), (2, 2, 1.5, 'C6'), (2, 3.5, .5, 'D6'),
        (3, 0, 1, 'C6'), (3, 1, 1, 'A5'), (3, 2, 2, 'G5'),
        (4, 0, 1, 'F5'), (4, 1, .5, 'G5'), (4, 1.5, .5, 'A5'), (4, 2, 2, 'D6'),
        (5, 0, 1, 'C#6'), (5, 1, .5, 'A5'), (5, 1.5, .5, 'G5'), (5, 2, 2, 'A5'),
    ]
    for b_, q, d, n in theme:
        brass.append((bar(16 + b_, q), d * BEAT * 0.95, n, 118))
        brass.append((bar(16 + b_, q), d * BEAT * 0.95, midi(n) - 12, 110))
    for bi in range(16, 22):
        choir += chord_notes(PROG[bi], 4, bar(bi), BAR, 80)

    # chase 44-54: theme reprise (first four bars, over D Bb F C) a step brighter, then the big A chord
    for b_, q, d, n in theme:
        if b_ < 4:
            brass.append((bar(22 + b_, q), d * BEAT * 0.95, n, 122))
            brass.append((bar(22 + b_, q), d * BEAT * 0.95, midi(n) - 12, 116))
            brass.append((bar(22 + b_, q), d * BEAT * 0.95, midi(n) - 24, 100))
    for bi in range(22, 26):
        choir += chord_notes(PROG[bi], 4, bar(bi), BAR, 92, top=True)
    brass += chord_notes('A', 4, bar(26), BAR * 0.98, 115, top=True) + [(bar(26), BAR * 0.98, 'A2', 110)]
    choir += chord_notes('A', 4, bar(26), BAR, 100, top=True)
    for q in (0, 1, 2, 3):
        hits += [(bar(26, q), 0.4, n, 125) for n in ('A3', 'E4', 'A4')]

    # end card 54-60: D major resolution
    brass += chord_notes('DM', 4, 54.0, 3.6, 120, top=True) + [(54.0, 3.6, 'D2', 115), (54.0, 3.6, 'D3', 115)]
    choir += chord_notes('DM', 4, 54.0, 3.8, 110, top=True)
    strings += chord_notes('DM', 3, 54.0, 3.9, 100, top=True) + chord_notes('DM', 5, 54.0, 3.9, 85)
    hits += [(tm, 0.6, n, 127) for tm in IMPACTS[1:] for n in ('D3', 'D4', 'A4')]
    # timpani rolls into each section change, hits on impacts
    for t0, t1 in ((18.2, 19.9), (28.0, 29.7), (42.0, 43.7), (52.0, 53.9)):
        t = t0
        while t < t1:
            timp.append((t, 0.1, 'D2' if t0 != 52.0 else 'A2', int(60 + 60 * (t - t0) / (t1 - t0))))
            t += 0.0625
    for tm in IMPACTS:
        timp.append((tm, 1.0, 'D2', 127))
        timp.append((tm, 1.0, 'A2', 120))
    timp.append((51.1, 0.8, 'D2', 120))

    stems = {
        'strings': render_midi('st_strings', [(48, 0, strings), (49, 1, [(t, d, n, v * 0.8) for t, d, n, v in strings])]),
        'ost': render_midi('st_ost', [(48, 0, ost), (45, 1, [(t, d, n, v * 0.7) for t, d, n, v in ost])]),
        'brass': render_midi('st_brass', [(61, 0, brass), (60, 1, [(t, d, n, v * 0.8) for t, d, n, v in brass]),
                                          (56, 2, [(t, d, n, v * 0.7) for t, d, n, v in brass if isinstance(n, str) and n[-1] >= '5'])]),
        'choir': render_midi('st_choir', [(52, 0, choir)]),
        'ethnic': render_midi('st_ethnic', [(77, 0, ethnic), (73, 1, [(t, d, n, v * 0.6) for t, d, n, v in ethnic]),
                                            (107, 2, koto)]),
        'piano': render_midi('st_piano', [(0, 0, piano)]),
        'timp': render_midi('st_timp', [(47, 0, timp), (55, 1, hits)]),
    }
    return stems


# ---------------------------------------------------------------- mix
def sidechain(kicks, depth=0.6, rel=0.16):
    g = np.ones(N)
    t = tt(0.5)
    shape = 1 - depth * np.exp(-t / rel) * np.minimum(1, t / 0.004 + 0.4)
    for k in kicks:
        i = int(k * SR)
        if i < N:
            seg = shape[:N - i]
            g[i:i + len(seg)] = np.minimum(g[i:i + len(seg)], seg)
    return g


def gap_gate():
    g = np.ones(N)
    for a, b in GAPS:
        i0, i1 = int(a * SR), int(b * SR)
        fade = int(0.01 * SR)
        g[i0 - fade:i0] = np.linspace(1, 0, fade)
        g[i0:i1] = 0
    return g


def st(x, pan=0.0):
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    return np.stack([x * l, x * r], 1) * 1.414


def main():
    drums, kicks = build_drums()
    bass, sub = build_bass()
    lead = build_lead()
    orch = orchestra()
    sc = sidechain(kicks)
    sc_soft = sidechain(kicks, 0.35)
    gate = gap_gate()

    mix = np.zeros((N, 2))
    mix += st(drums['kick'].x) * 0.95
    mix += st(drums['clap'].x, 0.05) * 0.55
    mix += st(drums['snare'].x, -0.05) * 0.55
    mix += st(drums['hatL'].x, -0.45) * 0.5 + st(drums['hatR'].x, 0.45) * 0.5
    mix += st(drums['perc'].x) * 0.6
    mix += st(drums['low'].x) * 0.8
    fxs = st(drums['fx'].x)
    mix += np.stack([fxs[:, 0], np.roll(fxs[:, 1], int(0.012 * SR))], 1) * 0.55
    mix += st(bass.x * sc) * 0.42 + st(sub.x * sc) * 0.5
    ld = lead.x * sc
    mix += np.stack([ld, np.roll(ld, int(0.015 * SR))], 1) * 0.55
    w = {'strings': 0.85, 'ost': 0.75, 'brass': 1.0, 'choir': 0.7, 'ethnic': 0.95, 'piano': 0.9, 'timp': 0.9}
    for k, x in orch.items():
        g = sc_soft[:, None] if k in ('strings', 'ost', 'choir') else 1.0
        mix += x * w[k] * g * 1.6
    # crowd roar under drop 2 and the finale
    cr = np.zeros(N)
    c = crowd(bar(27) - bar(15) + 2)
    i0 = int(bar(15) * SR)
    env = np.ones(len(c))
    env[:int(SR)] = np.linspace(0, 1, int(SR))
    env[-int(2 * SR):] = np.linspace(1, 0, int(2 * SR))
    cr[i0:i0 + len(c)] = c[:N - i0] * env[:N - i0]
    mix += np.stack([cr, np.roll(cr, int(0.02 * SR))], 1) * 0.5
    auto = np.interp(np.arange(N) / SR,
                     [0, 7.9, 8.0, 19.9, 20.0, 29.7, 30.0, 54.0, 60.0],
                     [0.75, 0.8, 0.82, 0.86, 0.86, 0.9, 1.0, 1.0, 1.0])
    mix *= (gate * auto)[:, None]
    # end fade
    fe = int(59.0 * SR)
    mix[fe:] *= np.linspace(1, 0, N - fe)[:, None] ** 2
    mix = mix[:int(DUR * SR)]
    mix /= np.max(np.abs(mix)) + 1e-9
    wavfile.write(f'{HERE}/mix_raw.wav', SR, (mix * 0.9 * 32767).astype(np.int16))
    subprocess.run(['sox', f'{HERE}/mix_raw.wav', f'{HERE}/bgm.wav',
                    'equalizer', '55', '1.0q', '+2', 'equalizer', '3500', '1.2q', '+1.5',
                    'compand', '0.005,0.12', '6:-60,-60,-20,-16,-6,-7,0,-3', '-3', '-90', '0.005',
                    'gain', '-n', '-0.8'], check=True)
    print('bgm written')


if __name__ == '__main__':
    main()
