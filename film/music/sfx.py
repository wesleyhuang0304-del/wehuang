"""Synthesised period foley: page turns, fire, wind, candle, quill."""
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

SR = 48000
DUR = 40.0
n = int(SR * DUR)
rng = np.random.default_rng(3)
out = np.zeros((n, 2), np.float32)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype='band', fs=SR, output='sos'), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, btype='low', fs=SR, output='sos'), x)


def env(length, attack, release, shape=2.0):
    t = np.linspace(0, 1, length)
    a = np.clip(t / max(attack, 1e-6), 0, 1)
    r = np.clip((1 - t) / max(release, 1e-6), 0, 1)
    return (np.minimum(a, r) ** shape).astype(np.float32)


def add(t0, sig, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    j = min(n, i + len(sig))
    if j <= i:
        return
    s = sig[:j - i] * gain
    out[i:j, 0] += s * np.sqrt(0.5 * (1 - pan))
    out[i:j, 1] += s * np.sqrt(0.5 * (1 + pan))


def page_turn(t0, dur=0.8):
    L = int(dur * SR)
    noise = rng.standard_normal(L)
    swish = bp(noise, 1800, 7000) * env(L, 0.35, 0.5, 1.5)
    body = bp(noise, 300, 1500) * env(L, 0.5, 0.4, 2.0) * 0.6
    crk = np.zeros(L)
    for k in rng.integers(0, L, 70):
        crk[k] = rng.uniform(-1, 1)
    crk = bp(crk, 2500, 9000) * 6 * env(L, 0.2, 0.6, 1.0)
    add(t0, (swish + body + crk).astype(np.float32), 0.20, pan=0.35)
    # page lands
    th = bp(rng.standard_normal(int(0.12 * SR)), 120, 900) * env(int(0.12 * SR), 0.05, 0.95, 2)
    add(t0 + dur * 0.82, th.astype(np.float32), 0.25, pan=-0.2)


def fire(t0, dur, gain=0.25):
    L = int(dur * SR)
    noise = rng.standard_normal(L)
    roar = lp(noise, 700) * env(L, 0.35, 0.45, 1.5) * 3
    hiss = bp(noise, 3000, 9000) * env(L, 0.3, 0.5, 1.5) * 0.25
    crk = np.zeros(L)
    for k in rng.integers(0, L, int(60 * dur)):
        crk[k:k + 40] += rng.uniform(-1, 1) * np.exp(-np.arange(min(40, L - k)) / 6)
    crk = bp(crk, 1200, 8000) * 3 * env(L, 0.15, 0.4, 1.0)
    add(t0, (roar + hiss + crk).astype(np.float32), gain, pan=rng.uniform(-0.3, 0.3))


def wind(t0, dur, gain, lo=150, hi=900):
    L = int(dur * SR)
    noise = rng.standard_normal(L)
    x = bp(noise, lo, hi)
    m = 0.6 + 0.4 * np.sin(np.linspace(0, dur * 1.3, L) * 2 * np.pi * 0.35 + 1) * \
        np.sin(np.linspace(0, dur, L) * 2 * np.pi * 0.21)
    x = x * m * env(L, 0.15, 0.2, 1.0)
    whistle = bp(noise, hi * 1.8, hi * 2.2, order=4) * 0.6 * (0.5 + 0.5 * np.sin(np.linspace(0, dur, L) * 1.7))
    add(t0, (x + whistle * env(L, 0.3, 0.3)).astype(np.float32) * 2.0, gain, pan=-0.4)
    add(t0 + 0.13, (x * 0.8).astype(np.float32) * 2.0, gain, pan=0.5)


def candle(t0, dur, gain):
    L = int(dur * SR)
    crk = np.zeros(L)
    for k in rng.integers(0, L, int(9 * dur)):
        crk[k:k + 60] += rng.uniform(-1, 1) * np.exp(-np.arange(min(60, L - k)) / 9)
    x = bp(crk, 800, 6000) * 2.5 + lp(rng.standard_normal(L), 300) * 0.4
    add(t0, (x * env(L, 0.15, 0.2, 1)).astype(np.float32), gain)


def quill(t0, dur, gain):
    L = int(dur * SR)
    noise = rng.standard_normal(L)
    tt = np.arange(L) / SR
    strokes = np.clip(np.sin(2 * np.pi * 3.2 * tt + rng.uniform(0, 6)) * np.sin(2 * np.pi * 1.1 * tt), 0, 1) ** 1.5
    x = bp(noise, 3500, 9000) * strokes * env(L, 0.1, 0.3, 1)
    add(t0, x.astype(np.float32), gain, pan=0.15)


candle(0.0, 4.2, 0.10)
fire(3.45, 1.15, 0.30)
page_turn(9.6)
wind(14.7, 1.1, 0.10, 120, 600)          # ink transition breath
fire(19.65, 1.15, 0.26)
wind(20.0, 6.2, 0.07, 180, 1100)        # mountain wind and snow
page_turn(25.6)
page_turn(31.4)
wind(32.0, 5.0, 0.045, 400, 2400)       # desert wind, sand
fire(36.65, 1.15, 0.28)
candle(37.4, 2.0, 0.07)
fire(39.05, 0.95, 0.22)
# quill writing under each title reveal
for t in [0.75, 4.35, 10.35, 15.55, 20.55, 26.35, 32.15, 37.45]:
    quill(t, 1.6, 0.035)

out = np.tanh(out * 1.2) / 1.2
wavfile.write('/workspace/film/music/sfx.wav', SR, out.astype(np.float32))
print('sfx peak', float(np.abs(out).max()))
