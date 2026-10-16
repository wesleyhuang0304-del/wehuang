"""Shared cue sheet for picture and score. 120 BPM: 1 beat = 0.5 s, 1 bar = 2 s, 30 bars = 60 s."""
BPM = 120
BEAT = 60 / BPM
DUR = 60.0
FPS = 24

# section name, start, end, palette key
SECTIONS = [
    ('intro', 0.0, 8.0, 'indigo'),
    ('heritage', 8.0, 20.0, 'sepia'),
    ('city', 20.0, 30.0, 'dawn'),
    ('drop', 30.0, 44.0, 'red'),
    ('chase', 44.0, 54.0, 'gold'),
    ('endcard', 54.0, 60.0, 'white'),
]

# shot start times (cuts) - every cut lands on a beat; one entry per unique shot in film.SHOTS
CUTS = [0.0, 2.0, 4.0, 6.0,
        8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
        20.0, 22.0, 24.0, 26.0, 28.0,
        30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 42.0,
        44.0, 46.0, 48.0, 50.0, 52.0,
        54.0]

# big hits: impact boom + white flash + heavy shake
IMPACTS = [8.0, 20.0, 30.0, 44.0, 54.0]
# silence gap right before a drop
GAPS = [(29.75, 30.0), (43.75, 44.0)]


def heartbeats():
    """Intro heartbeat, accelerating from 60 bpm toward the heritage section."""
    out, t, period = [], 0.35, 1.0
    while t < 7.7:
        out.append(t)
        t += period
        period = max(0.5, period * 0.9)
    return out


def section_at(t):
    for s in SECTIONS:
        if s[1] <= t < s[2]:
            return s
    return SECTIONS[-1]
