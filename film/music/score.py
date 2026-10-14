"""Score for Chronica Mediae Terrae (40 s). Tempo 60 bpm, so one beat = one second.

Theme in D dorian; each chapter re-orchestrates and re-harmonises it:
  0-4   Prologus      solo harp, bass pedal, distant voices, a tolling bell
  4-10  Terra Hominum recorder & lute over harp, tabor - pastoral
  10-15 Silva Aeterna flute & choir, harp arpeggios, F lydian - ethereal
  15-20 Regnum Altum  horns carry the theme, strings, timpani
  20-26 Sub Montibus  low brass, male voices, cello ostinato, heavy drums - D aeolian
  26-32 Insula Cael.  celesta, tremolo strings, voices - suspended harmony
  32-37 Finis Itin.   tutti, theme in brass & strings, rising to climax
  37-40 Picardy D major, timpani roll, harp sweep, bell
"""
import random
import mido

random.seed(7)
TPB = 480
EV = []   # (time_s, kind, ch, a, b)
PROG = {
    0: 46,   # harp
    1: 48,   # string ensemble
    2: 43,   # contrabass
    3: 52,   # choir aahs
    4: 74,   # recorder
    5: 73,   # flute
    6: 60,   # french horn
    7: 57,   # trombone
    8: 8,    # celesta
    10: 24,  # nylon guitar ~ lute
    11: 14,  # tubular bells
    12: 61,  # brass section
    13: 44,  # tremolo strings
    14: 47,  # timpani
    15: 42,  # cello
}

N = {}
for i, name in enumerate(['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'Bb', 'B']):
    for octv in range(0, 8):
        N[f'{name}{octv}'] = 12 * (octv + 1) + i


def note(t, dur, ch, n, vel, human=0.012):
    if isinstance(n, str):
        n = N[n]
    t = max(0.0, t + random.uniform(-human, human))
    vel = max(1, min(127, int(vel + random.uniform(-6, 6))))
    EV.append((t, 'on', ch, n, vel))
    EV.append((t + dur * 0.97, 'off', ch, n, 0))


def chord(t, dur, ch, notes, vel, roll=0.0):
    for i, n in enumerate(notes):
        note(t + i * roll, dur - i * roll, ch, n, vel)


def cc(t, ch, ctrl, val):
    EV.append((t, 'cc', ch, ctrl, int(max(0, min(127, val)))))


def ramp(t0, t1, ch, v0, v1, ctrl=11, steps=24):
    for i in range(steps + 1):
        u = i / steps
        cc(t0 + (t1 - t0) * u, ch, ctrl, v0 + (v1 - v0) * u)


def drum(t, n, vel):
    note(t, 0.3, 9, n, vel, human=0.006)


def arp(t0, t1, ch, notes, step, vel, up_down=True):
    seq = notes + (notes[-2:0:-1] if up_down else [])
    t, i = t0, 0
    while t < t1 - 1e-6:
        note(t, step * 2.5, ch, seq[i % len(seq)], vel - (i % len(seq)) * 2)
        t += step
        i += 1


THEME = [  # (offset beats, dur, note) in D dorian, 6 s
    (0.0, 1.0, 'A4'), (1.0, 0.5, 'D5'), (1.5, 0.5, 'C5'), (2.0, 1.0, 'A4'),
    (3.0, 0.5, 'G4'), (3.5, 0.5, 'F4'), (4.0, 0.75, 'G4'), (4.75, 0.25, 'E4'), (5.0, 1.0, 'D4'),
]


def theme(t0, ch, vel, transpose=0, scale=1.0, cut=None, legato=1.0):
    for off, d, n in THEME:
        if cut is not None and off >= cut:
            break
        note(t0 + off * scale, d * scale * legato, ch, N[n] + transpose, vel)


# ---------------------------------------------------------------- setup
for ch, p in PROG.items():
    EV.append((0.0, 'prog', ch, p, 0))
    cc(0.0, ch, 7, 100)
    cc(0.0, ch, 91, 70)   # reverb send
    cc(0.0, ch, 93, 25)   # chorus
    cc(0.0, ch, 11, 0)
pans = {0: 40, 1: 64, 2: 70, 3: 64, 4: 78, 5: 84, 6: 50, 7: 58, 8: 90, 10: 36, 11: 64, 12: 64, 13: 72, 14: 64, 15: 56}
for ch, pv in pans.items():
    cc(0.0, ch, 10, pv)

# ---------------------------------------------------------------- 0-4 Prologus
ramp(0.0, 1.0, 0, 70, 105)
ramp(0.0, 3.5, 2, 30, 90)
note(0.0, 4.2, 2, 'D1', 70)
note(0.0, 4.2, 2, 'D2', 55)
ramp(0.6, 3.8, 3, 0, 70)
chord(0.6, 3.6, 3, ['D3', 'A3', 'F4'], 55)
cc(0.0, 11, 11, 95)
note(0.15, 3.0, 11, 'D4', 70)
arp(0.3, 1.9, 0, ['D3', 'A3', 'D4', 'F4'], 0.25, 70, up_down=False)
for off, d, n in [(2.0, 0.5, 'A4'), (2.5, 0.5, 'D5'), (3.0, 0.5, 'C5'), (3.5, 0.6, 'A4')]:
    note(off, d * 1.3, 0, n, 82)
note(2.0, 2.0, 0, 'D3', 60)

# swell into chapter I (burn transition at 3.6-4.4)
ramp(3.2, 4.0, 1, 0, 80)
chord(3.4, 1.0, 1, ['D3', 'A3', 'D4'], 55)
note(3.85, 0.4, 14, 'A2', 45)

# ---------------------------------------------------------------- 4-10 Terra Hominum
ramp(4.0, 4.6, 4, 60, 100)
ramp(4.0, 4.6, 10, 80, 100)
cc(4.0, 0, 11, 95)
theme(4.0, 4, 92, transpose=12)            # recorder, high and light
harm = [(4.0, ['D3', 'A3', 'D4', 'F4']), (6.0, ['C3', 'G3', 'C4', 'E4']), (7.0, ['G2', 'D3', 'G3', 'B3']),
        (8.0, ['A2', 'E3', 'A3', 'C4']), (9.0, ['D3', 'A3', 'D4', 'F4'])]
for i, (t, ns) in enumerate(harm):
    t1 = harm[i + 1][0] if i + 1 < len(harm) else 10.0
    # lute strums on each half beat
    tt = t
    while tt < t1 - 1e-6:
        chord(tt, 0.45, 10, ns, 62 if (tt - t) % 1 == 0 else 48, roll=0.018)
        tt += 0.5
    note(t, t1 - t, 2, N[ns[0]] - 12, 70)
arp(4.0, 10.0, 0, ['D4', 'F4', 'A4', 'D5'], 0.25, 55)
ramp(4.0, 10.0, 1, 50, 65)
chord(4.0, 6.0, 1, ['D3', 'A3', 'F4'], 45)
for b in range(4, 10):
    drum(b, 45, 58)          # low tom as tabor
    drum(b + 0.5, 45, 34)
    if b % 2 == 1:
        drum(b + 0.75, 45, 28)
ramp(9.3, 10.0, 4, 100, 30)
ramp(9.4, 10.2, 10, 100, 0)

# ---------------------------------------------------------------- 10-15 Silva Aeterna (F lydian)
ramp(9.6, 10.6, 3, 40, 92)
ramp(9.6, 10.4, 5, 50, 100)
lyd = [(10.0, ['F3', 'C4', 'F4', 'A4'], ['F2']), (11.25, ['G3', 'B3', 'D4', 'G4'], ['F2']),
       (12.5, ['A3', 'C4', 'E4', 'A4'], ['A2']), (13.75, ['C4', 'E4', 'G4', 'C5'], ['C3'])]
for i, (t, ns, bass) in enumerate(lyd):
    t1 = lyd[i + 1][0] if i + 1 < len(lyd) else 15.0
    chord(t, t1 - t + 0.2, 3, ns[:3], 64)
    arp(t, t1, 0, [n + 12 if isinstance(n, int) else N[n] + 12 for n in ns], 0.125, 60, up_down=True)
    note(t, t1 - t, 2, bass[0], 60)
fl = [(10.0, 1.0, 'C6'), (11.0, 0.5, 'F6'), (11.5, 0.5, 'E6'), (12.0, 1.0, 'C6'), (13.0, 0.5, 'B5'),
      (13.5, 0.5, 'A5'), (14.0, 0.6, 'G5'), (14.6, 0.5, 'A5')]
for t, d, n in fl:
    note(t, d * 1.05, 5, n, 80)
ramp(10.0, 15.0, 1, 50, 70)
chord(10.0, 5.0, 1, ['F3', 'C4', 'A4'], 40)
note(10.0, 3.0, 11, 'F5', 40)
ramp(14.6, 15.4, 5, 100, 0)

# ---------------------------------------------------------------- 15-20 Regnum Altum
ramp(14.8, 15.4, 6, 50, 108)
cc(15.0, 3, 11, 70)
theme(15.0, 6, 96, transpose=0, scale=5.0 / 6.0)          # horns, theme slightly hastened
theme(15.0, 6, 78, transpose=-12, scale=5.0 / 6.0)
ramp(15.0, 20.0, 1, 70, 96)
kh = [(15.0, ['D3', 'A3', 'D4', 'F4']), (16.66, ['C3', 'G3', 'C4', 'E4']), (17.5, ['G2', 'D3', 'G3', 'B3']),
      (18.33, ['A2', 'E3', 'A3', 'C4']), (19.17, ['D3', 'A3', 'D4', 'F4'])]
for i, (t, ns) in enumerate(kh):
    t1 = kh[i + 1][0] if i + 1 < len(kh) else 20.0
    chord(t, t1 - t + 0.05, 1, ns, 72)
    note(t, t1 - t, 2, N[ns[0]] - 12, 80)
    chord(t, t1 - t, 3, ns[1:], 50)
for b in [15.0, 16.66, 18.33]:
    note(b, 0.6, 14, 'D2', 82)
ramp(19.0, 20.0, 14, 60, 110)
for i in range(16):  # timpani roll into the dwarves
    note(19.2 + i * 0.05, 0.08, 14, 'A1', 40 + i * 4)
arp(15.0, 20.0, 0, ['D3', 'A3', 'D4', 'A4'], 0.25, 50, up_down=False)
ramp(19.7, 20.3, 6, 108, 40)

# ---------------------------------------------------------------- 20-26 Sub Montibus (D aeolian, low)
ramp(19.8, 20.4, 7, 50, 110)
ramp(19.8, 20.4, 15, 50, 110)
cc(20.0, 3, 11, 95)
ramp(20.0, 26.0, 1, 50, 80)
ost = ['D2', 'D2', 'F2', 'D2', 'A2', 'G2', 'F2', 'E2']
t = 20.0
i = 0
while t < 26.0 - 1e-6:
    note(t, 0.36, 15, ost[i % 8], 92 if i % 4 == 0 else 74)   # cello ostinato
    t += 0.375
    i += 1
low = [(20.0, ['D2', 'A2', 'D3'], 'D1'), (21.5, ['Bb1', 'F2', 'D3'], 'Bb0'), (23.0, ['C2', 'G2', 'E3'], 'C1'),
       (24.5, ['A1', 'E2', 'Cs3'], 'A0')]
for j, (tt, ns, bs) in enumerate(low):
    t1 = low[j + 1][0] if j + 1 < len(low) else 26.0
    chord(tt, t1 - tt, 7, ns, 88)
    chord(tt, t1 - tt, 3, [N[n] + 12 for n in ns], 70)
    note(tt, t1 - tt, 2, bs, 90)
    chord(tt, t1 - tt, 1, [N[n] + 12 for n in ns], 60)
# augmented theme in trombones/horn, two octaves down
for off, d, n in THEME[:5]:
    note(20.0 + off * 1.2, d * 1.2, 6, N[n] - 12, 92)
for b in [20.0, 21.5, 23.0, 24.5]:
    note(b, 0.7, 14, 'D2', 110)
    drum(b, 35, 100)          # deep drum
    drum(b + 0.75, 35, 70)
for b in [20.75, 22.25, 23.75, 25.25]:
    note(b, 0.2, 14, 'A1', 70)
note(22.25, 2.5, 11, 'D3', 58)   # forge bell under the mountain
note(25.25, 0.8, 11, 'A2', 52)
ramp(25.5, 26.2, 7, 110, 0)
ramp(25.5, 26.2, 15, 110, 20)
ramp(25.5, 26.2, 14, 110, 60)

# ---------------------------------------------------------------- 26-32 Insula Caelestis
ramp(25.8, 26.6, 8, 40, 105)
ramp(25.8, 27.0, 13, 0, 72)
cc(26.0, 3, 11, 80)
cc(26.0, 3, 0, 0)
sus = [(26.0, ['Bb3', 'D4', 'F4', 'A4'], 'Bb1'), (27.5, ['A3', 'C4', 'F4', 'A4'], 'A1'),
       (29.0, ['G3', 'Bb3', 'D4', 'F4'], 'G1'), (30.5, ['A3', 'Cs4', 'E4', 'G4'], 'A1')]
for j, (tt, ns, bs) in enumerate(sus):
    t1 = sus[j + 1][0] if j + 1 < len(sus) else 32.0
    chord(tt, t1 - tt, 13, [N[n] + 12 for n in ns], 52)
    chord(tt, t1 - tt, 3, ns[:3], 56)
    note(tt, t1 - tt, 2, bs, 58)
    arp(tt, t1, 8, [N[n] + 12 for n in ns], 0.1875, 70, up_down=True)
cel = [(26.0, 1.5, 'A5'), (27.5, 0.75, 'D6'), (28.25, 0.75, 'C6'), (29.0, 1.5, 'A5'), (30.5, 0.75, 'G5'),
       (31.25, 0.75, 'E5')]
for tt, d, n in cel:
    note(tt, d, 5, n, 70)
ramp(26.0, 27.0, 5, 30, 85)
note(26.0, 4.0, 11, 'A5', 34)
note(29.0, 3.0, 11, 'F5', 30)
ramp(31.0, 32.0, 14, 50, 115)
for i in range(20):
    note(31.0 + i * 0.05, 0.08, 14, 'D2', 40 + i * 4)

# ---------------------------------------------------------------- 32-37 Finis Itineris - tutti
for ch_ in (1, 3, 6, 12, 2, 15, 7):
    cc(31.9, ch_, 11, 90)
ramp(32.0, 37.0, 1, 90, 120)
ramp(32.0, 37.0, 12, 85, 120)
ramp(32.0, 37.0, 3, 80, 115)
theme(32.0, 12, 104, transpose=0, scale=5.0 / 6.0)
theme(32.0, 1, 98, transpose=12, scale=5.0 / 6.0)
theme(32.0, 6, 96, transpose=-12, scale=5.0 / 6.0)
fin = [(32.0, ['D3', 'A3', 'D4', 'F4'], 'D1'), (33.66, ['Bb2', 'F3', 'Bb3', 'D4'], 'Bb0'),
       (34.5, ['C3', 'G3', 'C4', 'E4'], 'C1'), (35.33, ['A2', 'E3', 'A3', 'Cs4'], 'A0'),
       (36.17, ['A2', 'E3', 'A3', 'Cs4'], 'A0')]
for j, (tt, ns, bs) in enumerate(fin):
    t1 = fin[j + 1][0] if j + 1 < len(fin) else 37.0
    chord(tt, t1 - tt, 3, ns, 90)
    chord(tt, t1 - tt, 7, [N[n] - 12 for n in ns[:3]], 80)
    note(tt, t1 - tt, 2, bs, 100)
    t2 = tt
    while t2 < t1 - 1e-6:
        note(t2, 0.2, 15, N[ns[0]] - 12, 96)
        note(t2 + 0.25, 0.2, 15, N[ns[0]], 70)
        t2 += 0.5
for b in [32.0, 33.66, 34.5, 35.33]:
    note(b, 0.6, 14, 'D2', 112)
    drum(b, 35, 105)
    drum(b, 49, 70)   # crash as cymbal swell
for i in range(28):
    note(35.8 + i * 0.043, 0.06, 14, 'A1', 50 + i * 2.5)
arp(32.0, 37.0, 0, ['D3', 'A3', 'D4', 'F4', 'A4', 'D5'], 0.125, 66, up_down=True)

# ---------------------------------------------------------------- 37-40 Picardy cadence in D major
DMAJ = ['D2', 'A2', 'D3', 'Fs3', 'A3', 'D4', 'Fs4', 'A4']
chord(37.0, 2.9, 1, DMAJ[2:], 112)
chord(37.0, 2.9, 3, ['D3', 'Fs3', 'A3', 'D4', 'Fs4'], 108)
chord(37.0, 2.9, 12, ['D3', 'A3', 'D4', 'Fs4'], 110)
chord(37.0, 2.9, 7, ['D2', 'A2', 'D3'], 100)
note(37.0, 2.9, 2, 'D1', 110)
note(37.0, 2.9, 6, 'Fs4', 100)
note(37.0, 2.9, 6, 'A4', 100)
note(37.0, 2.0, 11, 'D4', 96)
note(37.0, 2.0, 11, 'D5', 70)
drum(37.0, 35, 120)
drum(37.0, 49, 96)
note(37.0, 0.8, 14, 'D2', 125)
for i in range(30):
    note(37.4 + i * 0.07, 0.1, 14, 'D2', 90 - i * 2.6)
for i, n in enumerate(['D3', 'Fs3', 'A3', 'D4', 'Fs4', 'A4', 'D5', 'Fs5', 'A5', 'D6']):
    note(37.0 + i * 0.04, 2.6, 0, n, 92)
for ch_ in (1, 3, 12, 7, 2, 6, 0):
    ramp(37.6, 39.8, ch_, 118, 0)
EV.append((40.0, 'end', 0, 0, 0))


def write(path):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage('set_tempo', tempo=1_000_000, time=0))
    order = {'prog': 0, 'cc': 1, 'off': 2, 'on': 3, 'end': 4}
    evs = sorted(EV, key=lambda e: (e[0], order[e[1]]))
    last = 0
    for t, kind, ch, a, b in evs:
        tick = int(round(t * TPB))
        dt = tick - last
        last = tick
        if kind == 'on':
            tr.append(mido.Message('note_on', channel=ch, note=a, velocity=b, time=dt))
        elif kind == 'off':
            tr.append(mido.Message('note_off', channel=ch, note=a, velocity=0, time=dt))
        elif kind == 'cc':
            tr.append(mido.Message('control_change', channel=ch, control=a, value=b, time=dt))
        elif kind == 'prog':
            tr.append(mido.Message('program_change', channel=ch, program=a, time=dt))
        else:
            tr.append(mido.MetaMessage('end_of_track', time=dt))
    mid.save(path)


if __name__ == '__main__':
    write('/workspace/film/music/score.mid')
    print('events', len(EV))
