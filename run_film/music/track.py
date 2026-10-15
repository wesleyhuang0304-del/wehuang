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


# chord roots for the bass, per bar
PROG = {}
for b0, seq in [(3, ['D', 'Bb', 'F', 'C']), (7, ['D', 'Bb', 'F', 'C']), (11, ['Bb', 'C', 'D', 'D']),
                (15, ['D', 'Bb', 'F', 'C']), (19, ['D', 'Bb', 'F', 'C', 'Bb', 'A']),
                (25, ['Bb', 'C', 'D']), (28, ['D', 'D'])]:
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

    # intro: heartbeat accelerating 60 -> ~110 bpm, ticks from bar 1
    for t in heartbeats():
        low.add(t, heartbeat(), 0.85)
        kicks.append(t)
    for e in range(4, 24):
        hatR.add(e * BEAT / 2, tick(), 0.5 + 0.5 * (e % 2 == 0))
    fx.add(6.0 - 3.0, reverse_swell(3.0), 0.55)
    fx.add(6.0 - 3.0, riser(3.0, 200, 6000), 0.35)

    # build bars 3-6: half-time kick, toms, accelerating snare roll, riser, gap
    for b in range(3, 7):
        for q in (0, 2):
            kickb.add(bar(b, q), K, 0.95)
            kicks.append(bar(b, q))
        for e in range(8):
            (hatL if e % 2 else hatR).add(bar(b, e / 2), HC, 0.45 + 0.25 * (e % 2))
        if b >= 4:
            clapb.add(bar(b, 1), C, 0.6)
            clapb.add(bar(b, 3), C, 0.6)
    for k, p in enumerate([180, 150, 120, 95, 180, 150, 120, 95]):
        perc.add(bar(5, 2 + k * 0.25), tom(p), 0.8)
    # snare roll 12.0 -> 13.75 accelerating 8ths -> 32nds
    t = 12.0
    while t < 13.72:
        u = (t - 12.0) / 1.75
        snb.add(t, S, 0.35 + 0.55 * u)
        t += BEAT / (2 if u < 0.25 else 4 if u < 0.6 else 8)
    fx.add(10.0, riser(3.75, 250, 11000), 0.75)

    # drop 1 bars 7-10, drop 2 bars 19-24 (bigger)
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
        # end-of-section fill
        for k in range(8):
            perc.add(bar(b1 - 1, 2 + k * 0.25), tom([200, 170, 140, 115, 200, 170, 140, 115][k]), 0.7)

    groove(7, 11, False)
    groove(19, 25, True)

    # heritage bars 11-14: lighter groove + taiko ensemble; motto hits 26.0-27.5
    for b in range(11, 15):
        for q in (0, 2.5):
            kickb.add(bar(b, q), K, 0.85)
            kicks.append(bar(b, q))
        clapb.add(bar(b, 1), C, 0.55)
        clapb.add(bar(b, 3), C, 0.55)
        for e in range(8):
            perc.add(bar(b, e / 2 + 0.25), SH, 0.8)
        for q in (0, 0.75, 1.5):
            perc.add(bar(b, q), taiko(0.85), 0.7)
    for k, tm in enumerate([26.0, 26.5, 27.0, 27.5, 28.0, 29.0]):
        perc.add(tm, taiko(1.1 - 0.05 * k), 0.9)
    for k in range(40):  # 1906 -> 2026 counter ticks
        hatR.add(22.1 + k * 0.045, tick(), 0.55)

    # breakdown bars 15-18: heartbeat, then kick 8ths -> 16ths, snare roll, riser, gap
    for b in (15, 16):
        for q in (0, 2):
            low.add(bar(b, q), heartbeat(), 0.8)
    for e in range(16):
        kickb.add(bar(17, e / 2), K, 0.45 + 0.03 * e)
        kicks.append(bar(17, e / 2))
    t = 36.0
    while t < 37.72:
        u = (t - 36.0) / 1.75
        snb.add(t, S, 0.4 + 0.55 * u)
        t += BEAT / (2 if u < 0.3 else 4 if u < 0.65 else 8)
    fx.add(34.0, riser(3.75, 200, 12000), 0.8)

    # finale bars 25-27: heavy half-time + chant pattern bar 27, then end hit
    for b in range(25, 28):
        for q in (0, 2):
            kickb.add(bar(b, q), KB, 0.95)
            kicks.append(bar(b, q))
        snb.add(bar(b, 1), S, 0.6)
        clapb.add(bar(b, 3), C, 0.9)
        snb.add(bar(b, 3), S, 0.6)
        for e in range(8):
            (hatL if e % 2 else hatR).add(bar(b, e / 2), HO if e % 2 else HC, 0.35)
        for q in (0, 1, 2, 3):
            perc.add(bar(b, q), taiko(0.9), 0.65)
    for q in (0, 1, 2, 3):  # chant hits 54.0..55.5
        kickb.add(bar(27, q), KB, 1.0)
        clapb.add(bar(27, q), C, 1.0)
        perc.add(bar(27, q), taiko(1.0), 0.9)
    fx.add(56.0 - 2.0, riser(2.0, 300, 9000), 0.55)

    # impacts, crashes, sub drops on every big hit; whoosh into every cut
    for tm in IMPACTS:
        low.add(tm, impact(), 1.0)
        low.add(tm, subdrop(), 0.7)
        fx.add(tm, crash(3.0), 0.7)
        kicks.append(tm)
    for tm in CUTS[1:]:
        if tm not in IMPACTS:
            fx.add(tm - 0.2, whoosh(0.4), 0.45 if tm < 38 else 0.55)
    fx.add(58.0, crash(2.0), 0.35)
    return dict(kick=kickb, clap=clapb, snare=snb, hatL=hatL, hatR=hatR, perc=perc, fx=fx, low=low), kicks


def build_bass():
    b = Bus()
    for bi, root in PROG.items():
        if bi < 3 or bi >= 28:
            continue
        f = hz(midi(root + '1'))
        if 15 <= bi <= 16:
            b.add(bar(bi), saw_voice(f, BAR - 0.05, 300, env_a=0.2, env_r=0.3), 0.55)
            continue
        if bi in (3, 4, 5, 6, 11, 12, 13, 14, 17, 25, 26, 27):
            for e in range(8):
                b.add(bar(bi, e / 2), saw_voice(f, BEAT / 2 * 0.85, 700), 0.6)
        else:  # drops: rolling 16ths, offbeat octave
            for s in range(16):
                ff = f * (2 if s % 4 == 2 else 1)
                b.add(bar(bi, s / 4), saw_voice(ff, BEAT / 4 * 0.8, 1100), 0.62)
    sub = Bus()
    for bi, root in PROG.items():
        if 3 <= bi < 28 and not 15 <= bi <= 16:
            t = tt(BAR)
            f = hz(midi(root + '1'))
            sub.add(bar(bi), np.sin(2 * np.pi * f * t) * np.minimum(1, t / 0.01) * np.minimum(1, (BAR - t) / 0.02), 0.5)
    t = tt(4.0)
    sub.add(56.0, np.sin(2 * np.pi * hz(midi('D1')) * t) * np.exp(-t / 1.8), 0.7)
    return b, sub


def build_lead():
    """Supersaw stabs for the drops (sidechained hard)."""
    L = Bus()
    for bi in list(range(7, 11)) + list(range(19, 25)):
        ch = CHORDS[PROG[bi]]
        for e in (0, 1.5, 3, 4.5, 6) if bi >= 19 else (0, 1.5, 3, 6):
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
    # intro pad (Dm), swelling
    strings += [(0.2, 5.8, n, 55) for n in ('D3', 'A3', 'F4')]
    # string ostinato 16ths: build, drop1, drop2
    pat = [0, 2, 1, 2]
    for bi in list(range(3, 11)) + list(range(19, 25)):
        ch = CHORDS[PROG[bi]]
        oct_ = 4 if bi >= 7 else 3
        for s in range(16):
            n = ch[pat[s % 4]] + str(oct_ + (s % 8 == 7))
            ost.append((bar(bi, s / 4), BEAT / 4 * 0.8, n, 70 + 30 * (s % 4 == 0) + (15 if bi >= 19 else 0)))
    # long strings under build / drops / heritage / finale
    for bi in list(range(3, 15)) + list(range(19, 28)):
        strings += chord_notes(PROG[bi], 3, bar(bi), BAR, 62 if bi < 7 else 75)
    # brass stabs drop1, theme drop2
    for bi in range(7, 11):
        for e in (0, 1.5, 3, 6):
            brass += chord_notes(PROG[bi], 4, bar(bi, e / 2), BEAT * 0.4, 112)
        hits += [(bar(bi), 0.5, n, 120) for n in ('D4', 'A4', 'D5')]
    theme = [  # (bar, beat, dur, note)
        (19, 0, 1, 'D5'), (19, 1, .5, 'F5'), (19, 1.5, .5, 'G5'), (19, 2, 1.5, 'A5'), (19, 3.5, .5, 'C6'),
        (20, 0, 1, 'A5'), (20, 1, .5, 'G5'), (20, 1.5, .5, 'F5'), (20, 2, 2, 'G5'),
        (21, 0, 1, 'F5'), (21, 1, .5, 'G5'), (21, 1.5, .5, 'A5'), (21, 2, 1.5, 'C6'), (21, 3.5, .5, 'D6'),
        (22, 0, 1, 'C6'), (22, 1, 1, 'A5'), (22, 2, 2, 'G5'),
        (23, 0, 1, 'F5'), (23, 1, .5, 'G5'), (23, 1.5, .5, 'A5'), (23, 2, 2, 'D6'),
        (24, 0, 1, 'C#6'), (24, 1, .5, 'A5'), (24, 1.5, .5, 'G5'), (24, 2, 2, 'A5'),
    ]
    for b_, q, d, n in theme:
        brass.append((bar(b_, q), d * BEAT * 0.95, n, 118))
        brass.append((bar(b_, q), d * BEAT * 0.95, midi(n) - 12, 110))
    for bi in range(19, 25):
        choir += chord_notes(PROG[bi], 4, bar(bi), BAR, 80)
    # heritage: shakuhachi/flute pentatonic melody + koto arpeggios
    mel = [(11, 0, 1, 'D5'), (11, 1, .5, 'F5'), (11, 1.5, .5, 'G5'), (11, 2, 2, 'A5'),
           (12, 0, 1, 'G5'), (12, 1, .5, 'A5'), (12, 1.5, .5, 'C6'), (12, 2, 2, 'D6'),
           (13, 0, .5, 'C6'), (13, .5, .5, 'A5'), (13, 1, 1, 'G5'), (13, 2, .5, 'F5'), (13, 2.5, .5, 'G5'),
           (13, 3, 1, 'A5'), (14, 0, 3.5, 'D5')]
    for b_, q, d, n in mel:
        ethnic.append((bar(b_, q), d * BEAT, n, 105))
    koto = []
    for bi in range(11, 15):
        ch = CHORDS[PROG[bi]]
        for e in range(8):
            koto.append((bar(bi, e / 2), BEAT, ch[[0, 1, 2, 1][e % 4]] + str(4 + (e % 4 == 2)), 90))
    # breakdown: piano arpeggios + simple motif, pad strings
    for bi in range(15, 19):
        ch = CHORDS[PROG[bi]]
        for e in range(8):
            piano.append((bar(bi, e / 2), BEAT * 1.5, ch[[0, 2, 1, 2][e % 4]] + str(3 + (e % 4 == 3)), 70 + 10 * (e == 0)))
        strings += chord_notes(PROG[bi], 3, bar(bi), BAR, 60)
    for b_, q, d, n in [(15, 0, 2, 'A5'), (15, 2, 2, 'F5'), (16, 0, 2, 'D5'), (16, 2, 2, 'F5'),
                        (17, 0, 1, 'C6'), (17, 1, 1, 'A5'), (17, 2, 2, 'G5'), (18, 0, 3.5, 'A5')]:
        piano.append((bar(b_, q), d * BEAT, n, 92))
    # finale: big brass + choir chords, end D major
    for bi in (25, 26, 27):
        brass += chord_notes(PROG[bi], 4, bar(bi), BAR * 0.98, 115, top=True)
        brass += [(bar(bi), BAR * 0.98, CHORDS[PROG[bi]][0] + '2', 110)]
        choir += chord_notes(PROG[bi], 4, bar(bi), BAR, 100, top=True)
    for q in (0, 1, 2, 3):
        hits += [(bar(27, q), 0.4, n, 125) for n in ('D4', 'A4', 'D5')]
    brass += chord_notes('DM', 4, 56.0, 3.6, 120, top=True) + [(56.0, 3.6, 'D2', 115), (56.0, 3.6, 'D3', 115)]
    choir += chord_notes('DM', 4, 56.0, 3.8, 110, top=True)
    strings += chord_notes('DM', 3, 56.0, 3.9, 100, top=True) + chord_notes('DM', 5, 56.0, 3.9, 85)
    hits += [(tm, 0.6, n, 127) for tm in (14.0, 38.0, 50.0, 56.0) for n in ('D3', 'D4', 'A4')]
    # timpani rolls into drops, hits on impacts
    for t0, t1 in ((12.0, 13.7), (36.0, 37.7), (48.0, 49.9)):
        t = t0
        while t < t1:
            timp.append((t, 0.1, 'D2', int(60 + 60 * (t - t0) / (t1 - t0))))
            t += 0.0625
    for tm in IMPACTS:
        timp.append((tm, 1.0, 'D2', 127))
        timp.append((tm, 1.0, 'A2', 120))
    for t, _, _, _ in [(x, 0, 0, 0) for x in (50.0, 52.0, 54.0)]:
        timp.append((t, 0.8, 'D2', 120))

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
    c = crowd(bar(28) - bar(19) + 2)
    i0 = int(bar(19) * SR)
    env = np.ones(len(c))
    env[:int(SR)] = np.linspace(0, 1, int(SR))
    env[-int(2 * SR):] = np.linspace(1, 0, int(2 * SR))
    cr[i0:i0 + len(c)] = c[:N - i0] * env[:N - i0]
    mix += np.stack([cr, np.roll(cr, int(0.02 * SR))], 1) * 0.5
    auto = np.interp(np.arange(N) / SR,
                     [0, 5.9, 6.0, 13.7, 14.0, 21.9, 22.0, 29.9, 30.0, 37.7, 38.0, 49.9, 50.0, 56.0, 60.0],
                     [0.75, 0.8, 0.62, 0.86, 1.0, 1.0, 0.82, 0.86, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0])
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
