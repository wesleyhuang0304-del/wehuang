"""Shared cue sheet for picture and score. 120 BPM: 1 beat = 0.5 s, 1 bar = 2 s, 30 bars = 60 s."""
BPM = 120
BEAT = 60 / BPM
DUR = 60.0
FPS = 24

# section name, start, end, palette key
SECTIONS = [
    ('intro', 0.0, 6.0, 'indigo'),
    ('build', 6.0, 14.0, 'dawn'),
    ('drop1', 14.0, 22.0, 'red'),
    ('heritage', 22.0, 30.0, 'crimson'),
    ('breakdown', 30.0, 38.0, 'teal'),
    ('drop2', 38.0, 50.0, 'orange'),
    ('finale', 50.0, 56.0, 'gold'),
    ('endcard', 56.0, 60.0, 'white'),
]

# shot start times (cuts) - every cut lands on a beat
CUTS = [0.0, 3.0, 6.0, 8.0, 10.0, 12.0,
        14.0, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 19.0, 20.0, 21.0,
        22.0, 26.0, 28.0,
        30.0, 33.0, 36.0,
        38.0, 39.0, 40.0, 41.0, 41.5, 42.0, 42.5, 43.0, 44.0, 46.0, 47.0, 48.0,
        50.0, 54.0, 56.0]

# big hits: impact boom + white flash + heavy shake
IMPACTS = [6.0, 14.0, 22.0, 38.0, 50.0, 56.0]
# silence gap right before a drop
GAPS = [(13.75, 14.0), (37.75, 38.0)]


def heartbeats():
    """Intro heartbeat, accelerating from 60 bpm toward the build."""
    out, t, period = [], 0.35, 1.0
    while t < 5.8:
        out.append(t)
        t += period
        period = max(0.55, period * 0.9)
    return out


def section_at(t):
    for s in SECTIONS:
        if s[1] <= t < s[2]:
            return s
    return SECTIONS[-1]
