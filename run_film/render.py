"""60 s vertical promo renderer: 湘雅追光乐跑团 · 2026 长沙马拉松. 1080x1920 @ 24 fps, beat-locked to 120 BPM."""
import os, sys, math, functools
import numpy as np, cv2
from kit import *
from timeline import *

OUT = f'{ROOT}/out/frames'
SUNSET = (1.0, 0.62, 0.22)
MAGENTA = np.array([1.0, 0.30, 0.55], np.float32)
ORANGE = np.array([1.0, 0.58, 0.16], np.float32)
CYAN = np.array([0.45, 0.95, 1.0], np.float32)
AMBER = np.array([1.0, 0.68, 0.32], np.float32)

PAL = {
    'indigo': dict(g=dict(lift=(0.01, 0.015, 0.05), gain=(0.86, 0.93, 1.08), gamma=(1.04, 1.0, 0.96), sat=0.9, con=1.06),
                   acc=AMBER, acc2=np.array([0.35, 0.55, 1.0], np.float32), flash=(0.7, 0.8, 1.0)),
    'dawn': dict(g=dict(lift=(0.03, 0.015, 0.0), gain=(1.08, 0.97, 0.82), gamma=(0.98, 1.0, 1.06), sat=1.1, con=1.05),
                 acc=GOLD, acc2=ORANGE, flash=(1.0, 0.9, 0.7)),
    'red': dict(g=dict(lift=(0.03, 0.0, 0.01), gain=(1.08, 0.96, 0.92), gamma=(0.98, 1.0, 1.02), sat=1.15, con=1.1),
                acc=XY_RED, acc2=GOLD, flash=(1.0, 1.0, 1.0)),
    'crimson': dict(g=dict(lift=(0.05, 0.01, 0.0), gain=(1.1, 0.9, 0.76), gamma=(0.97, 1.0, 1.08), sat=1.05, con=1.08),
                    acc=GOLD, acc2=XY_RED, flash=(1.0, 0.85, 0.6)),
    'teal': dict(g=dict(lift=(0.0, 0.03, 0.045), gain=(0.82, 1.0, 1.04), gamma=(1.06, 1.0, 0.97), sat=0.8, con=1.08),
                 acc=CYAN, acc2=WHITE, flash=(0.75, 1.0, 1.0)),
    'orange': dict(g=dict(lift=(0.04, 0.0, 0.03), gain=(1.1, 0.9, 0.86), gamma=(0.96, 1.02, 1.02), sat=1.25, con=1.1),
                   acc=MAGENTA, acc2=ORANGE, flash=(1.0, 0.8, 0.7)),
    'gold': dict(g=dict(lift=(0.04, 0.03, 0.0), gain=(1.12, 1.0, 0.82), gamma=(0.97, 1.0, 1.05), sat=1.1, con=1.05),
                 acc=GOLD, acc2=WHITE, flash=(1.0, 0.95, 0.8)),
    'white': dict(g=dict(lift=(0.05, 0.04, 0.02), gain=(1.1, 1.04, 0.92), gamma=(0.98, 1.0, 1.04), sat=1.0, con=1.04),
                  acc=GOLD, acc2=XY_RED, flash=(1.0, 1.0, 1.0)),
}


def pal_at(t):
    return PAL[section_at(t)[3]]


def ease(kind, u):
    u = min(1.0, max(0.0, u))
    if kind == 'out':
        return 1 - (1 - u) ** 3
    if kind == 'expo':
        return 1 - 2 ** (-9 * u)
    if kind == 'in':
        return u ** 2.4
    if kind == 'io':
        return u * u * (3 - 2 * u)
    return u


def bezier(p, u):
    p = np.asarray(p, np.float32)
    n = len(p) - 1
    return sum(math.comb(n, i) * (1 - u) ** (n - i) * u ** i * p[i] for i in range(n + 1))


# ------------------------------------------------------------------ assets (cached per process)
@functools.lru_cache(maxsize=4)
def plate(name):
    return load_plate(name)


@functools.lru_cache(maxsize=4)
def pmask(name):
    p = plate(name)
    m = load_mask(name, p.shape[:2])
    return clamp01((m - 0.12) / 0.7)


@functools.lru_cache(maxsize=256)
def tmask(text, fname, size, tracking=0, weight=None):
    return text_mask(text, fname, size, tracking, weight)


def cam_M(pshape, z, cx, cy, rot, shake=(0, 0)):
    ph, pw = pshape[:2]
    s = max(W / pw, H / ph) * z
    M = cv2.getRotationMatrix2D((pw * cx, ph * cy), math.degrees(rot), s)
    M[0, 2] += W / 2 - pw * cx + shake[0]
    M[1, 2] += H / 2 - ph * cy + shake[1]
    return M


def warp(img, M):
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def to_screen(M, pshape, nx, ny):
    ph, pw = pshape[:2]
    return M @ np.array([nx * pw, ny * ph, 1.0])


# ------------------------------------------------------------------ shots
class Shot:
    def __init__(self, t0, plate, z=(1.1, 1.0), cx=(0.5, 0.5), cy=(0.5, 0.5), rot=(0, 0), ease='out', mask=False,
                 sun=None, flare=0.0, rib=(), fx=(), blur=None, whip=(1, 0), tint=None, shade=False):
        self.t0, self.plate, self.z, self.cx, self.cy, self.rot = t0, plate, z, cx, cy, rot
        self.ease, self.mask, self.sun, self.flare, self.rib, self.fx = ease, mask, sun, flare, rib, fx
        self.blur, self.whip, self.tint, self.shade = blur, whip, tint, shade


R = math.radians
SHOTS = [
    Shot(0.0, 's01_lace', z=(1.28, 1.12), cy=(0.52, 0.6), rot=(R(-2), R(1)), ease='io', sun=(0.78, 0.16), flare=0.32,
         fx=[('dust', AMBER, 60)]),
    Shot(3.0, 's02_city_dawn', z=(1.18, 1.04), cy=(0.62, 0.47), ease='io',
         rib=[dict(pts='road', col=AMBER, w=7, sp=0.55, ph=0.0)], fx=[('dust', AMBER, 30)]),
    Shot(6.0, 's03_sunrise_isle', z=(1.45, 1.08), cy=(0.44, 0.5), ease='expo', sun=(0.54, 0.41), flare=0.55,
         blur=('radial', 0.09, 'decay')),
    Shot(8.0, 'hero_runner', z=(1.0, 1.1), cx=(0.5, 0.53), cy=(0.53, 0.5), ease='out', mask=True, sun=(0.337, 0.387),
         flare=0.4, blur=('radial', 0.04, 'const'),
         rib=[dict(pts=[(-80, 1500), (250, 1250), (120, 860), 'sun'], col=SUNSET, w=11, sp=0.9, ph=0.2),
              dict(pts=[(1200, 1650), (820, 1300), (980, 820), 'sun'], col=(1.0, 0.3, 0.25), w=8, sp=0.9, ph=0.6)],
         fx=[('sparks', SUNSET, 90)]),
    Shot(10.0, 's04_feet', z=(1.02, 1.3), cy=(0.5, 0.42), ease='in', mask=True, blur=('dir', 60, 'ramp'), whip=(0, 1),
         fx=[('sparks', GOLD, 120)]),
    Shot(12.0, 's05_team', z=(1.0, 1.22), cy=(0.5, 0.48), ease='in', mask=True, flare=0.3, sun=(0.5, 0.33),
         rib=[dict(pts=[(-60, 1800), (300, 1500), (200, 900), (540, 640)], col=GOLD, w=8, sp=1.3, ph=0.0),
              dict(pts=[(1140, 1800), (780, 1500), (880, 900), (540, 640)], col=(1.0, 0.35, 0.3), w=8, sp=1.3, ph=0.3)]),
    # ---- drop 1
    Shot(14.0, 'hero_runner', z=(1.12, 1.05), cx=(0.52, 0.52), cy=(0.52, 0.52), ease='expo', mask=True,
         sun=(0.337, 0.387), flare=0.45, blur=('radial', 0.05, 'const'),
         rib=[dict(pts=[(-80, 1500), (250, 1250), (120, 860), 'sun'], col=SUNSET, w=11, sp=1.1, ph=0.5),
              dict(pts=[(1200, 1650), (820, 1300), (980, 820), 'sun'], col=(1.0, 0.3, 0.25), w=8, sp=1.1, ph=0.9)],
         fx=[('sparks', SUNSET, 140)]),
    Shot(15.0, 's07_start_crowd', z=(1.25, 1.1), ease='expo', whip=(1, 0), fx=[('confetti', None, 140)]),
    Shot(15.5, 's04_feet', z=(1.35, 1.2), cy=(0.45, 0.45), ease='expo', whip=(-1, 0), mask=True, blur=('dir', 50, 'const')),
    Shot(16.0, 's06_female', z=(1.25, 1.12), cx=(0.55, 0.5), ease='expo', whip=(1, 0), mask=True, blur=('dir', 45, 'const')),
    Shot(16.5, 'hero_runner', z=(1.75, 1.6), cx=(0.58, 0.58), cy=(0.36, 0.38), ease='expo', whip=(-1, 0), mask=True,
         sun=(0.337, 0.387), flare=0.3),
    Shot(17.0, 's05_team', z=(1.3, 1.18), ease='expo', whip=(0, 1), mask=True),
    Shot(17.5, 's07_start_crowd', z=(1.0, 1.35), cy=(0.4, 0.3), ease='in', whip=(0, -1), blur=('radial', 0.06, 'ramp')),
    Shot(18.0, 's06_female', z=(1.05, 1.15), cx=(0.5, 0.56), ease='out', mask=True, blur=('dir', 70, 'const'),
         rib=[dict(pts=[(1150, 1500), (700, 1350), (500, 1100), (-100, 900)], col=XY_RED, w=10, sp=1.4, ph=0.1)]),
    Shot(19.0, 's04_feet', z=(1.1, 1.2), ease='out', mask=True, whip=(-1, 0), fx=[('sparks', GOLD, 160)]),
    Shot(20.0, 'hero_runner', z=(1.25, 1.15), cx=(0.56, 0.54), cy=(0.6, 0.56), ease='out', mask=True,
         sun=(0.337, 0.387), flare=0.4, blur=('radial', 0.05, 'const'),
         rib=[dict(pts=[(-80, 1800), (250, 1450), (120, 1060), 'sun'], col=SUNSET, w=11, sp=1.2, ph=0.0)]),
    Shot(21.0, 's05_team', z=(1.0, 1.15), ease='in', mask=True, whip=(1, 0), sun=(0.5, 0.33), flare=0.3),
    # ---- heritage
    Shot(22.0, 's08_redbuilding', z=(1.2, 1.03), cy=(0.62, 0.45), ease='out', mask=True, sun=(0.95, 0.2), flare=0.35,
         fx=[('dust', GOLD, 70)],
         rib=[dict(pts=[(-100, 1700), (400, 1500), (700, 900), (1200, 400)], col=GOLD, w=8, sp=0.6, ph=0.0)]),
    Shot(26.0, 's09_doctor', z=(1.0, 1.32), cy=(0.5, 0.47), ease='io', mask=True, sun=(0.5, 0.42), flare=0.45,
         fx=[('dust', GOLD, 50)]),
    Shot(28.0, 's06_female', z=(1.2, 1.06), cx=(0.52, 0.5), ease='expo', mask=True, blur=('dir', 60, 'const'),
         rib=[dict(pts=[(1150, 1600), (700, 1350), (500, 1000), (-100, 700)], col=GOLD, w=10, sp=0.8, ph=0.1),
              dict(pts=[(1150, 1750), (800, 1500), (600, 1250), (-100, 1000)], col=XY_RED, w=7, sp=0.8, ph=0.4)]),
    # ---- breakdown
    Shot(30.0, 's10_or_lamp', z=(1.05, 1.16), rot=(0, R(4)), cy=(0.4, 0.4), ease='lin', sun=(0.53, 0.25), flare=0.35,
         fx=[('dust', CYAN, 90)]),
    Shot(33.0, 's11_nurse_night', z=(1.12, 1.05), cx=(0.45, 0.55), ease='lin', mask=True, fx=[('dust', CYAN, 30)]),
    Shot(36.0, 's15_grit', z=(1.55, 1.95), cy=(0.38, 0.34), ease='in', mask=True),
    # ---- drop 2
    Shot(38.0, 's12_aerial_bridge', z=(1.3, 1.1), rot=(R(-4), R(0)), ease='expo',
         rib=[dict(pts='bridge', col=MAGENTA, w=12, sp=1.0, ph=0.0), dict(pts='bridge', col=ORANGE, w=7, sp=1.0, ph=0.5)]),
    Shot(39.0, 's13_yuelu', z=(1.25, 1.08), cy=(0.5, 0.56), ease='expo', whip=(-1, 0), fx=[('leaves', None, 70)]),
    Shot(40.0, 's14_cheer', z=(1.15, 1.04), ease='expo', whip=(1, 0), fx=[('confetti', None, 160)]),
    Shot(41.0, 's15_grit', z=(1.15, 1.05), ease='expo', whip=(0, 1), mask=True),
    Shot(41.5, 's04_feet', z=(1.3, 1.15), ease='expo', whip=(-1, 0), mask=True, blur=('dir', 55, 'const')),
    Shot(42.0, 's06_female', z=(1.3, 1.15), ease='expo', whip=(1, 0), mask=True, blur=('dir', 55, 'const')),
    Shot(42.5, 's05_team', z=(1.3, 1.15), ease='expo', whip=(0, -1), mask=True),
    Shot(43.0, 's13_yuelu', z=(1.0, 1.18), cx=(0.5, 0.55), ease='in', mask=True, fx=[('leaves', None, 90)],
         blur=('radial', 0.05, 'ramp')),
    Shot(44.0, 's12_aerial_bridge', z=(1.05, 1.2), cy=(0.55, 0.45), rot=(R(3), R(-2)), ease='io',
         rib=[dict(pts='bridge', col=MAGENTA, w=12, sp=0.75, ph=0.0), dict(pts='bridge', col=ORANGE, w=7, sp=0.75, ph=0.45),
              dict(pts='bridge', col=GOLD, w=5, sp=0.75, ph=0.9)]),
    Shot(46.0, 's15_grit', z=(1.25, 1.45), cy=(0.4, 0.38), ease='in', mask=True),
    Shot(47.0, 's14_cheer', z=(1.25, 1.1), ease='expo', whip=(1, 0), fx=[('confetti', None, 200)]),
    Shot(48.0, 'hero_runner', z=(1.02, 1.28), cx=(0.52, 0.53), cy=(0.5, 0.5), ease='in', mask=True, sun=(0.337, 0.387),
         flare=0.5, blur=('radial', 0.11, 'ramp'),
         rib=[dict(pts=[(-80, 1500), (250, 1250), (120, 860), 'sun'], col=MAGENTA, w=12, sp=1.8, ph=0.0),
              dict(pts=[(1200, 1650), (820, 1300), (980, 820), 'sun'], col=ORANGE, w=9, sp=1.8, ph=0.4)],
         fx=[('sparks', ORANGE, 160)]),
    # ---- finale
    Shot(50.0, 's16_finish', z=(1.72, 1.56), cy=(0.36, 0.37), ease='expo', mask=True, sun=(0.6, 0.2), flare=0.28,
         fx=[('confetti', None, 260)]),
    Shot(54.0, 's14_cheer', z=(1.2, 1.1), ease='out', fx=[('confetti', None, 220)]),
    Shot(56.0, 's17_endcard', z=(1.25, 1.05), cy=(0.45, 0.5), ease='expo', mask=True, sun=(0.62, 0.45), flare=0.32, shade=True,
         rib=[dict(pts=[(-100, 1700), (200, 1300), (300, 1000), 'sun'], col=GOLD, w=10, sp=0.55, ph=0.0),
              dict(pts=[(1180, 1750), (900, 1400), (1000, 1000), 'sun'], col=XY_RED, w=8, sp=0.55, ph=0.3)],
         fx=[('dust', GOLD, 80)]),
]
assert [s.t0 for s in SHOTS] == CUTS, 'shot list out of sync with timeline.CUTS'


def shot_at(t):
    i = max(k for k, s in enumerate(SHOTS) if s.t0 <= t)
    t1 = SHOTS[i + 1].t0 if i + 1 < len(SHOTS) else DUR
    return i, SHOTS[i], t1


# ------------------------------------------------------------------ text events
TX = []


def slam(t0, t1, text, x, y, size, font='sport', rot=-3, behind=False, fill=None, ext=None, weight=None, glowc=None):
    TX.append(dict(k='slam', t0=t0, t1=t1, text=text, x=x, y=y, size=size, font=font, rot=R(rot), behind=behind,
                   fill=fill, ext=ext, weight=weight, glowc=glowc))


def line(t0, t1, text, x, y, size, font='sans', weight=900, color=WHITE, bar=None, rot=0, track=4):
    TX.append(dict(k='line', t0=t0, t1=t1, text=text, x=x, y=y, size=size, font=font, weight=weight, color=color,
                   bar=bar, rot=R(rot), track=track, behind=False))


def en(t0, t1, text, x, y, size, color=WHITE, track=(4, 18), rot=0):
    TX.append(dict(k='en', t0=t0, t1=t1, text=text, x=x, y=y, size=size, color=color, track=track, rot=R(rot),
                   behind=False))


def num(t0, t1, x, y, size, text=None, count=None, behind=False, color=WHITE, stroke=GOLD, rot=0):
    TX.append(dict(k='num', t0=t0, t1=t1, text=text, count=count, x=x, y=y, size=size, behind=behind, color=color,
                   stroke=stroke, rot=R(rot)))


def vert(t0, t1, text, x, y, size, color=WHITE, font='serif', weight=900):
    TX.append(dict(k='vert', t0=t0, t1=t1, text=text, x=x, y=y, size=size, color=color, font=font, weight=weight,
                   behind=False))


# intro
en(0.5, 2.9, 'CHANGSHA · 05:30 AM', 540, 360, 44, color=AMBER)
line(1.0, 2.9, '长沙 · 清晨五点半', 540, 450, 64, weight=700)
line(3.4, 5.9, '这座城 还在沉睡', 540, 520, 76, weight=700)
line(4.6, 5.9, '有人 已经出发', 540, 640, 76, weight=900, color=AMBER)
# build
slam(6.0, 10.0, '追', 330, 470, 520, rot=-3, behind=False)
slam(8.0, 10.0, '光', 760, 470, 520, rot=-3, behind=True)
en(9.0, 9.95, 'CHASE THE LIGHT', 545, 215, 70, track=(2, 22))
num(10.0, 11.95, 420, 560, 200, text='42.195', behind=True)
en(10.25, 11.95, 'KILOMETERS · 11.15 · CHANGSHA', 420, 740, 40, color=GOLD)
line(12.25, 13.74, '湘雅追光乐跑团', 540, 470, 112, bar=XY_RED, track=8)
en(12.75, 13.74, 'XIANGYA · CHASE THE LIGHT RUNNING CLUB', 540, 590, 34, color=GOLD, track=(3, 8))
# drop 1
slam(14.0, 14.98, '追光', 560, 520, 560, rot=-3, behind=True)
en(14.0, 14.98, 'CHASE THE LIGHT', 545, 215, 70, track=(16, 16), rot=-3)
for tt_, word, fnt, sz, x, y, rot, beh in [(15.0, '2026', 'num', 380, 540, 900, 3, False),
                                            (15.5, '长沙', 'sport', 360, 400, 820, -4, True),
                                            (16.0, '马拉松', 'sport', 300, 560, 1350, 3, False),
                                            (16.5, '11.15', 'num', 300, 540, 380, -4, True),
                                            (17.0, '42.195', 'num', 230, 540, 520, 3, True),
                                            (17.5, 'GO!', 'num', 460, 540, 760, -4, False)]:
    slam(tt_, tt_ + 0.48, word, x, y, sz, font=fnt, rot=rot, behind=beh)
slam(18.0, 18.98, '每一步', 560, 1420, 260, behind=False)
slam(19.0, 19.98, '都算数', 400, 820, 260, behind=True)
slam(20.0, 20.98, '向光而行', 520, 300, 200, behind=True)
slam(21.0, 21.98, '全力以赴', 540, 560, 250, behind=True)
# heritage
num(22.1, 24.0, 540, 1330, 360, count=(1906, 2026, 22.1, 23.9, '{:d}'), behind=False, stroke=GOLD)
en(22.4, 25.95, 'SINCE 1906 · XIANGYA HOSPITAL', 540, 1560, 40, color=GOLD)
slam(24.0, 25.95, '湘雅', 330, 380, 300, behind=True, rot=-2)
slam(24.5, 25.95, '120年', 620, 1380, 260, font='sport', behind=False, rot=-2)
for k, word in enumerate(['公勇勤慎', '诚爱谦廉', '求真求确', '必邃必专']):
    vert(26.0 + 0.5 * k, 27.95, word, (950, 820, 260, 130)[k], 360, 100, color=GOLD if k % 2 == 0 else WHITE)
slam(28.0, 29.95, '白衣为甲', 540, 1160, 230, behind=False, rot=-3)
slam(29.0, 29.95, '跑鞋为翼', 560, 1420, 230, behind=False, rot=-3, ext=XY_RED)
# breakdown
line(30.5, 32.95, '有一种光', 540, 1180, 92, font='serif', weight=900)
line(31.5, 32.95, '在无影灯下', 540, 1320, 92, font='serif', weight=900, color=CYAN)
line(33.5, 35.95, '守护', 760, 520, 120, font='serif', weight=900)
line(34.5, 35.95, '这座城的', 760, 680, 76, font='serif', weight=700, color=CYAN)
line(34.9, 35.95, '每一个夜晚', 760, 790, 76, font='serif', weight=700, color=CYAN)
line(36.0, 37.74, '我们追光', 540, 1300, 130, font='serif', weight=900)
slam(37.0, 37.74, '也成为光', 540, 1500, 180, rot=-2)
# drop 2
num(38.0, 38.98, 440, 1300, 300, text='10KM', stroke=MAGENTA)
line(38.2, 38.98, '橘子洲大桥', 440, 1500, 72, bar=MAGENTA)
num(39.0, 39.98, 760, 1280, 300, text='21KM', stroke=ORANGE)
line(39.2, 39.98, '岳麓山下', 760, 1480, 72, bar=ORANGE)
TX.append(dict(k='banner', t0=40.0, t1=40.98, text='湘雅加油！', behind=False))
TX.append(dict(k='banner', t0=47.0, t1=47.98, text='湘雅加油！', behind=False))
TX.append(dict(k='banner', t0=54.0, t1=55.98, text='湘雅加油！', behind=False))
for tt_, word in [(41.0, '35KM'), (41.5, '38KM'), (42.0, '40KM'), (42.5, '41KM')]:
    num(tt_, tt_ + 0.48, 540, 1250 if tt_ in (41.0, 42.0) else 700, 330, text=word, stroke=MAGENTA if tt_ % 1 else ORANGE)
slam(43.0, 43.48, '超越', 540, 640, 380, behind=True)
slam(43.5, 43.98, '自我', 540, 640, 380, behind=True, ext=MAGENTA)
vert(44.0, 45.95, '湘江北去', 170, 330, 120, color=WHITE)
vert(45.0, 45.95, '橘子洲头', 910, 330, 120, color=GOLD)
slam(46.0, 46.48, '别停！', 540, 1300, 360, rot=-4)
slam(46.5, 46.98, '再坚持！', 540, 1300, 300, rot=3, ext=MAGENTA)
slam(47.0, 47.48, '加油！', 540, 1150, 340, rot=-4)
slam(47.5, 47.98, '加油！！', 540, 1150, 270, rot=4, ext=MAGENTA)
num(48.0, 49.95, 560, 430, 280, count=(0.0, 42.195, 48.0, 49.5, '{:.3f}'), behind=True, stroke=ORANGE)
en(48.3, 49.95, 'KM  TO  THE  LIGHT', 545, 275, 46, color=GOLD, track=(6, 14))
# finale
en(50.0, 53.95, '42.195KM · FINISH', 540, 1640, 46, color=GOLD)
line(50.5, 53.95, '致敬每一位', 540, 1220, 96, weight=900)
slam(51.5, 53.95, '追光者', 540, 1410, 280, rot=-3)
for k, (tt_, word) in enumerate([(54.0, '湘雅'), (54.5, '追光'), (55.0, '加油'), (55.5, '冲！')]):
    slam(tt_, tt_ + 0.48, word, 540, 900, 400, rot=(-4, 3, -3, 4)[k],
         ext=(XY_RED, GOLD, XY_RED, GOLD)[k], fill=((1, 1, 1), (1, 1, 1)) if k % 2 == 0 else None)
# end card
vert(56.6, 60.0, '我们追光', 150, 280, 100, color=WHITE)
vert(57.0, 60.0, '也成为光', 930, 280, 100, color=GOLD)
slam(56.0, 60.0, '湘雅追光乐跑团', 540, 1370, 128, font='sans', weight=900, rot=-2, ext=XY_RED)
en(56.3, 60.0, 'XIANGYA · CHASE THE LIGHT RUNNING CLUB', 540, 1505, 34, color=GOLD, track=(3, 8), rot=-2)
line(56.6, 60.0, '2026 长沙马拉松 · 11.15 · 42.195KM', 540, 1600, 50, font='sport', weight=None, bar=XY_RED, track=2)
TX.append(dict(k='badge', t0=57.2, t1=60.0, text='湘雅 1906—2026', behind=False))


# ------------------------------------------------------------------ text renderers
def anim_in(dt, dur=0.07):
    return clamp01(dt / dur)


def exit_a(t, t1, dur=0.06):
    return clamp01((t1 - t) / dur)


def draw_slam(img, e, t, pal, sh):
    dt = t - e['t0']
    a = anim_in(dt, 0.05) * exit_a(t, e['t1'])
    if a <= 0:
        return img
    m = tmask(e['text'], e['font'], e['size'], 0, e['weight'])
    sc = 1 + 0.75 * math.exp(-dt / 0.045) + 0.035 * dt
    if t > e['t1'] - 0.06:
        sc *= 1 + 2.0 * (t - (e['t1'] - 0.06))
    T = paste_mask((H, W), m, e['x'], e['y'], sc, e['rot']) * a
    ext_c = np.asarray(e['ext'] if e['ext'] is not None else pal['acc'], np.float32)
    if dt < 0.16:
        for k, (s2, al) in enumerate([(1.12, 0.35), (1.28, 0.18), (1.5, 0.08)]):
            ghost = paste_mask((H, W), m, e['x'], e['y'], sc * s2, e['rot']) * a * al * (1 - dt / 0.16)
            img = img + ghost[..., None] * ext_c * 1.2
    d = e['size'] * 0.03 * sc
    ext = extrude(T, d * 0.7, d, 10)
    img = img * (1 - (cv2.GaussianBlur(ext, (0, 0), 16) * 0.45)[..., None])
    img = over(img, ext_c * 0.82, clamp01(ext - T) * 0.95)
    for dx, al in [(-0.12, 0.16), (-0.065, 0.3)]:
        echo = cv2.warpAffine(T, np.float32([[1, 0, dx * e['size']], [0, 1, 0]]), (W, H))
        img = over(img, np.asarray(pal['acc2'], np.float32), echo * al)
    top, bot = e['fill'] if e['fill'] is not None else ((1.0, 1.0, 0.98), tuple(0.55 + 0.45 * np.asarray(pal['acc2'])))
    img = over(img, gradient_fill(T, top, bot), T)
    img = glow(img, T, e['glowc'] if e['glowc'] is not None else pal['acc'], 22, 0.32)
    return img


def draw_line(img, e, t, pal, sh):
    dt = t - e['t0']
    a = exit_a(t, e['t1'], 0.08)
    if a <= 0:
        return img
    m = tmask(e['text'], e['font'], e['size'], e['track'], e['weight'])
    h, w = m.shape
    rev = ease('expo', dt / 0.28)
    xoff = (1 - ease('expo', dt / 0.35)) * -60
    T = paste_mask((H, W), m, e['x'] + xoff, e['y'], 1.0, e['rot'])
    x0 = e['x'] + xoff - w / 2 - 10
    xs = np.arange(W, dtype=np.float32)
    wipe = clamp01((x0 + (w + 40) * rev - xs) / 24)[None, :]
    if e['bar'] is not None:
        bar = np.zeros((H, W), np.float32)
        bw = (w + 60) * ease('expo', dt / 0.2)
        bx0 = e['x'] - w / 2 - 30
        cv2.rectangle(bar, (int(bx0), int(e['y'] - h / 2 - 16)), (int(bx0 + bw), int(e['y'] + h / 2 + 16)), 1.0, -1)
        img = over(img, np.asarray(e['bar'], np.float32), bar * 0.92 * a)
    else:
        sh_ = cv2.GaussianBlur(T, (0, 0), 10)
        img = img * (1 - (sh_ * 0.55 * a)[..., None])
    img = over(img, np.asarray(e['color'], np.float32), T * wipe * a)
    return img


def draw_en(img, e, t, pal, sh):
    dt = t - e['t0']
    a = anim_in(dt, 0.2) * exit_a(t, e['t1'], 0.1)
    if a <= 0:
        return img
    tr0, tr1 = e['track']
    track = int(tr0 + (tr1 - tr0) * ease('out', dt / 1.2))
    m = tmask(e['text'], 'num', e['size'], track)
    T = paste_mask((H, W), m, e['x'], e['y'], 1.0, e['rot']) * a
    sh_ = cv2.GaussianBlur(T, (0, 0), 6)
    img = img * (1 - (sh_ * 0.5)[..., None])
    return over(img, np.asarray(e['color'], np.float32), T)


def draw_num(img, e, t, pal, sh):
    dt = t - e['t0']
    a = anim_in(dt, 0.05) * exit_a(t, e['t1'])
    if a <= 0:
        return img
    if e['count']:
        v0, v1, c0, c1, fmt = e['count']
        u = ease('out', (t - c0) / (c1 - c0))
        text = fmt.format(type(v0)(v0 + (v1 - v0) * u)) if fmt == '{:d}' else fmt.format(v0 + (v1 - v0) * u)
        if fmt == '{:d}':
            text = '{:d}'.format(int(round(v0 + (v1 - v0) * u)))
    else:
        text = e['text']
    m = tmask(text, 'num', e['size'], 6)
    sc = 1 + 0.4 * math.exp(-dt / 0.05)
    T = paste_mask((H, W), m, e['x'], e['y'], sc, e['rot']) * a
    k = max(3, int(e['size'] * 0.025))
    outer = cv2.dilate(T, np.ones((k, k), np.uint8))
    stroke = clamp01(outer - T)
    img = img * (1 - (cv2.GaussianBlur(outer, (0, 0), 14) * 0.45)[..., None])
    img = over(img, np.asarray(e['stroke'], np.float32), stroke)
    img = over(img, np.asarray(e['color'], np.float32), T * 0.94)
    img = glow(img, outer, e['stroke'], 18, 0.3)
    return img


def draw_vert(img, e, t, pal, sh):
    a_exit = exit_a(t, e['t1'], 0.1)
    for i, ch in enumerate(e['text']):
        dt = t - e['t0'] - i * 0.07
        if dt < 0:
            continue
        a = anim_in(dt, 0.12) * a_exit
        m = tmask(ch, e['font'], e['size'], 0, e['weight'])
        y = e['y'] + i * e['size'] * 1.08 - (1 - ease('expo', dt / 0.3)) * 50
        T = paste_mask((H, W), m, e['x'], y, 1.0, 0) * a
        img = img * (1 - (cv2.GaussianBlur(T, (0, 0), 12) * 0.6)[..., None])
        img = over(img, np.asarray(e['color'], np.float32), T)
    return img


def draw_banner(img, e, t, pal, sh):
    """Paint 湘雅加油！ onto the blank red banner of the cheer plate, following its perspective."""
    dt = t - e['t0']
    a = anim_in(dt, 0.04) * exit_a(t, e['t1'])
    M, pshape = sh['M'], sh['pshape']
    quad = np.float32([to_screen(M, pshape, *p) for p in [(0.01, 0.04), (0.63, 0.09), (0.6, 0.35), (0.02, 0.33)]])
    m = tmask(e['text'], 'sport', 220, 6)
    h, w = m.shape
    pad = 0.12
    src = np.float32([[-w * pad, -h * pad * 2], [w * (1 + pad), -h * pad * 2], [w * (1 + pad), h * (1 + pad * 2)],
                      [-w * pad, h * (1 + pad * 2)]])
    P = cv2.getPerspectiveTransform(src, quad)
    T = cv2.warpPerspective(m, P, (W, H)) * a
    sc = 1 + 0.25 * math.exp(-dt / 0.06)
    if sc > 1.001:
        c = quad.mean(0)
        T = cv2.warpAffine(T, cv2.getRotationMatrix2D((float(c[0]), float(c[1])), 0, sc), (W, H))
    # print the lettering into the cloth: follow the fabric's folds in shading and a slight warp
    L = warp(plate(sh['name']), M) @ np.array([0.299, 0.587, 0.114], np.float32)
    Ls = cv2.GaussianBlur(L, (0, 0), 3)
    region = cv2.fillConvexPoly(np.zeros((H, W), np.uint8), quad.astype(np.int32), 1).astype(bool)
    ref = float(np.median(Ls[region])) if region.any() else 0.5
    gy, gx = np.gradient(cv2.GaussianBlur(L, (0, 0), 9))
    mx, my = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    T = cv2.remap(T, mx - gx * 260, my - gy * 260, cv2.INTER_LINEAR)
    T = cv2.GaussianBlur(T, (0, 0), 0.7)
    shade = np.clip(Ls / max(ref, 1e-3), 0.45, 1.35) ** 0.9
    ink = np.array([1.0, 0.92, 0.55], np.float32) * shade[..., None]
    img = img * (1 - T[..., None] * 0.92) + ink * T[..., None] * 0.92
    return glow(img, T, GOLD, 10, 0.18)


def draw_badge(img, e, t, pal, sh):
    dt = t - e['t0']
    a = anim_in(dt, 0.15)
    tag = tmask(e['text'], 'sans', 38, 4, 700)
    badge = np.zeros((H, W), np.float32)
    bw = tag.shape[1] + 44
    x0 = int(540 - bw / 2)
    cv2.rectangle(badge, (x0, 1690), (x0 + int(bw * ease('expo', dt / 0.3)), 1690 + tag.shape[0] + 26), 1.0, -1)
    img = over(img, XY_RED, badge * 0.92 * a)
    return over(img, WHITE, paste_mask((H, W), tag, 540, 1703 + tag.shape[0] / 2) * a)


DRAW = dict(slam=draw_slam, line=draw_line, en=draw_en, num=draw_num, vert=draw_vert, banner=draw_banner,
            badge=draw_badge)


# ------------------------------------------------------------------ particles
def dust(t, seed, n, color):
    rng = np.random.default_rng(seed)
    lay = np.zeros((H // 2, W // 2), np.float32)
    x0, y0 = rng.random(n) * W, rng.random(n) * H
    vx, vy = rng.normal(0, 14, n), -12 - rng.random(n) * 30
    sz = rng.random(n) ** 3 * 4 + 1
    tw = rng.random(n) * 6
    for i in range(n):
        x = (x0[i] + vx[i] * t + 30 * math.sin(t * 0.7 + tw[i])) % W
        y = (y0[i] + vy[i] * t) % H
        b = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 3 + tw[i]))
        cv2.circle(lay, (int(x / 2), int(y / 2)), max(1, int(sz[i] / 2)), float(b), -1, cv2.LINE_AA)
    a = cv2.GaussianBlur(lay, (0, 0), 1.2) * 0.8 + cv2.GaussianBlur(lay, (0, 0), 5) * 0.8
    return cv2.resize(a, (W, H))[..., None] * np.asarray(color, np.float32) * 0.6


CONF_COLS = [np.array(c, np.float32) for c in [(0.85, 0.08, 0.12), (1.0, 0.82, 0.36), (1, 1, 1), (1.0, 0.6, 0.15),
                                               (0.95, 0.3, 0.5)]]


def confetti(img, t, seed, n, t0):
    rng = np.random.default_rng(seed)
    age = t - t0
    for i in range(n):
        x0 = rng.random() * W
        y0 = -rng.random() * H * 0.9 - 40
        vy = 260 + rng.random() * 420
        burst = rng.normal(0, 260)
        sz = 9 + rng.random() * 14
        sp = rng.random() * 10 + 4
        ph = rng.random() * 6
        col = CONF_COLS[rng.integers(len(CONF_COLS))]
        x = x0 + burst * (1 - math.exp(-age * 3)) + 40 * math.sin(age * sp * 0.5 + ph)
        y = y0 + vy * age + 400 * (1 - math.exp(-age * 2)) * 0.3
        if y < -40 or y > H + 40:
            continue
        ang = age * sp + ph
        flip = abs(math.cos(age * sp * 0.8 + ph))
        c, s = math.cos(ang), math.sin(ang)
        hw, hh = sz, sz * 0.45 * (0.2 + flip)
        pts = np.array([[x + c * dx - s * dy, y + s * dx + c * dy] for dx, dy in
                        [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]], np.int32)
        cv2.fillPoly(img, [pts], tuple(float(v) * (0.65 + 0.35 * flip) for v in col), cv2.LINE_AA)
    return img


def leaves(img, t, seed, n):
    rng = np.random.default_rng(seed)
    for i in range(n):
        x0, y0 = rng.random() * W, rng.random() * H
        vy = 120 + rng.random() * 220
        sw = rng.random() * 3 + 1
        ph = rng.random() * 6
        sz = 12 + rng.random() * 18
        col = [(0.85, 0.12, 0.08), (0.95, 0.38, 0.06), (0.75, 0.05, 0.12), (1.0, 0.6, 0.1)][rng.integers(4)]
        x = (x0 + 80 * math.sin(t * sw + ph) + 60 * t) % W
        y = (y0 + vy * t) % H
        ang = math.degrees(t * sw * 1.5 + ph)
        fl = 0.35 + 0.65 * abs(math.sin(t * sw + ph))
        cv2.ellipse(img, (int(x), int(y)), (int(sz), max(2, int(sz * 0.45 * fl))), ang, 0, 360,
                    tuple(float(v) * (0.7 + 0.3 * fl) for v in col), -1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ rhythm envelopes
def beat_env(t, period, decay, phase=0.0):
    d = (t - phase) % period
    return math.exp(-d / decay)


def rhythm(t):
    """zoom punch amount, shake amplitude (px)"""
    sec = section_at(t)[0]
    punch, shake = 0.0, 0.0
    if sec == 'intro':
        for hb in heartbeats():
            if 0 <= t - hb < 0.4:
                punch += 0.012 * math.exp(-(t - hb) / 0.08)
    elif sec == 'build':
        punch = 0.014 * beat_env(t, 2 * BEAT, 0.1)
        if t >= 12.0:
            punch += 0.02 * beat_env(t, BEAT / 2, 0.05) * (t - 12.0) / 1.75
            shake = 6 * (t - 12.0) / 1.75
    elif sec in ('drop1', 'drop2'):
        punch = 0.035 * beat_env(t, BEAT, 0.09)
        shake = 11 * beat_env(t, BEAT, 0.1)
    elif sec == 'heritage':
        punch = 0.016 * beat_env(t, BEAT, 0.1)
        shake = 3 * beat_env(t, BEAT, 0.1)
    elif sec == 'breakdown':
        if t >= 34.0:
            punch = 0.012 * beat_env(t, BEAT / 2, 0.06) * (t - 34) / 4
            shake = 5 * max(0, (t - 36.0) / 1.75)
    elif sec == 'finale':
        punch = 0.03 * beat_env(t, 2 * BEAT, 0.12)
        if t >= 54.0:
            punch = 0.07 * beat_env(t, BEAT, 0.08)
            shake = 26 * beat_env(t, BEAT, 0.12)
    for tm in IMPACTS:
        if 0 <= t - tm < 0.6:
            shake += 42 * math.exp(-(t - tm) / 0.16)
            punch += 0.08 * math.exp(-(t - tm) / 0.12)
    return punch, shake


def shake_vec(t, amp):
    if amp < 0.3:
        return (0.0, 0.0)
    fi = int(t * FPS)
    rng = np.random.default_rng(fi * 13 + 5)
    return tuple(rng.normal(0, amp, 2))


# ------------------------------------------------------------------ ribbons
def rib_points(spec, M, pshape, sun_xy):
    pts = spec['pts']
    if pts == 'road':
        return [to_screen(M, pshape, *p) for p in [(-0.1, 0.648), (0.3, 0.628), (0.62, 0.612), (1.1, 0.6)]]
    if pts == 'bridge':
        return [to_screen(M, pshape, *p) for p in [(0.9, 1.04), (0.66, 0.68), (0.48, 0.4), (0.3, 0.08)]]
    return [sun_xy if p == 'sun' else p for p in pts]


def render_frame(fi):
    t = fi / FPS
    i, sh, t1 = shot_at(t)
    pal = pal_at(t)
    lt = t - sh.t0
    u = lt / (t1 - sh.t0)
    e = ease(sh.ease, u)
    punch, shk = rhythm(t)
    z = (sh.z[0] + (sh.z[1] - sh.z[0]) * e) * (1 + punch)
    cx = sh.cx[0] + (sh.cx[1] - sh.cx[0]) * e
    cy = sh.cy[0] + (sh.cy[1] - sh.cy[0]) * e
    rot = sh.rot[0] + (sh.rot[1] - sh.rot[0]) * e
    P = plate(sh.plate)
    M = cam_M(P.shape, z, cx, cy, rot, shake_vec(t, shk))
    base = warp(P, M)
    m = warp(pmask(sh.plate), M) if sh.mask else None
    sun_xy = tuple(to_screen(M, P.shape, *sh.sun)) if sh.sun else (540, 700)
    ctx = dict(M=M, pshape=P.shape, name=sh.plate)

    bg = base
    if sh.blur:
        kind, amt, mode = sh.blur
        k = {'const': 1.0, 'ramp': 0.3 + 0.7 * u, 'decay': math.exp(-lt / 0.5)}[mode]
        if kind == 'radial':
            bg = radial_blur(base, sun_xy[0], sun_xy[1], amt * k, 8)
        else:
            bg = dir_blur(base, amt * k, 0, 7)
    if lt < 0.13 and i > 0:
        w = 1 - lt / 0.13
        bg = dir_blur(bg, sh.whip[0] * 160 * w, sh.whip[1] * 160 * w, 9)
        base = bg if m is None else base
    img = grade(bg, **pal['g'])
    if sh.tint is not None:
        img = img * np.asarray(sh.tint, np.float32)
    if sh.shade:
        yy = np.arange(H, dtype=np.float32)
        d = 1 - 0.55 * (1 - smoothstep(0, 760, yy)) - 0.62 * smoothstep(1180, 1560, yy)
        img = img * d[:, None, None]

    behind = [x for x in TX if x['t0'] <= t < x['t1'] and x.get('behind') and m is not None]
    front = [x for x in TX if x['t0'] <= t < x['t1'] and not (x.get('behind') and m is not None)]
    for x in behind:
        img = DRAW[x['k']](img, x, t, pal, ctx)

    for k, r in enumerate(sh.rib):
        pts = rib_points(r, M, P.shape, sun_xy)
        img = img + light_ribbon((H, W), lambda uu, tt_, pts=pts: bezier(pts, uu), lt * r['sp'] + r['ph'], r['w'],
                                 r['col'], seed=k) * 1.2

    if m is not None:
        fgsrc = base if lt >= 0.13 else bg
        fg = grade(fgsrc, **pal['g'])
        edge = clamp01(m - cv2.GaussianBlur(m, (0, 0), 6)) * 2.2
        fg = fg + edge[..., None] * np.asarray(pal['acc'] if sh.sun is None else SUNSET, np.float32) * 0.45
        img = img * (1 - m[..., None]) + fg * m[..., None]

    for x in front:
        img = DRAW[x['k']](img, x, t, pal, ctx)

    for kind, col, n in sh.fx:
        if kind == 'dust':
            img = img + dust(t, i, n, col)
        elif kind == 'sparks':
            img = img + sparks((H, W), lt + 1.0, i, n, lambda r, j: (r * W, H + 20 - (j % 9) * 8), col, speed=560,
                               life=0.85)
        elif kind == 'confetti':
            img = confetti(img, t, i, n, sh.t0 - 0.4)
        elif kind == 'leaves':
            img = leaves(img, t, i, n)

    if sh.sun and sh.flare > 0:
        pulse = 1 + 0.35 * punch / 0.035
        fl_lay = flare((H, W), sun_xy[0], sun_xy[1], sh.flare * min(pulse, 2.0), tuple(0.6 + 0.4 * np.asarray(pal['acc2'])))
        if m is not None:
            fl_lay = fl_lay * (1 - 0.85 * m[..., None])
        img = img + fl_lay

    # broadcast bug
    if 14.0 <= t < 56.0 and not any(a <= t < b for a, b in GAPS):
        img = bug(img, t)

    # flashes: impacts, cut frames in the drops, heartbeat glows
    fl = 0.0
    fcol = np.asarray(pal['flash'], np.float32)
    for tm in IMPACTS:
        if 0 <= t - tm < 0.5:
            fl = max(fl, 0.95 * math.exp(-(t - tm) / 0.09))
    sec = section_at(t)[0]
    if sec in ('drop1', 'drop2', 'finale') and lt < 1.0 / FPS + 1e-6 and sh.t0 not in IMPACTS:
        fl = max(fl, 0.35)
        fcol = np.asarray(pal['acc'] if (i % 2) else pal['flash'], np.float32)
    if fl > 0:
        img = img * (1 - fl * 0.5) + fcol * fl
    chroma = 1.4 + 9 * (fl if fl > 0.2 else 0) + shk * 0.08

    # silent gaps before drops: black with a thin anamorphic streak charging up
    for a, b in GAPS:
        if a <= t < b:
            g = (t - a) / (b - a)
            img = img * 0.04
            yy = np.arange(H, dtype=np.float32)[:, None]
            xx = np.arange(W, dtype=np.float32)[None, :]
            streak = np.exp(-((yy - 960) / (3 + 10 * g)) ** 2) * np.exp(-np.abs(xx - 540) / (80 + 900 * g))
            img = img + streak[..., None] * np.asarray(pal['flash'], np.float32) * (0.6 + 1.2 * g)

    # end fade
    if t > 59.3:
        img = img * (1 - ease('io', (t - 59.3) / 0.7))
    if t < 0.4:
        img = img * ease('io', t / 0.4)
    return finish(img, fi, grain=0.032, vign=0.3, chroma=chroma)


def bug(img, t):
    m = tmask('湘雅追光乐跑团', 'sans', 30, 2, 800)
    m2 = tmask('2026 长马 · 11.15', 'sport', 30, 1)
    x0, y0 = 54, 130
    w1 = m.shape[1] + 30
    w2 = m2.shape[1] + 26
    hgt = max(m.shape[0], m2.shape[0]) + 22
    box = np.zeros((H, W), np.float32)
    cv2.fillPoly(box, [np.array([[x0, y0], [x0 + w1 + 10, y0], [x0 + w1, y0 + hgt], [x0 - 10, y0 + hgt]], np.int32)], 1.0,
                 cv2.LINE_AA)
    box2 = np.zeros((H, W), np.float32)
    cv2.fillPoly(box2, [np.array([[x0 + w1 + 10, y0], [x0 + w1 + w2 + 20, y0], [x0 + w1 + w2 + 10, y0 + hgt],
                                  [x0 + w1, y0 + hgt]], np.int32)], 1.0, cv2.LINE_AA)
    img = over(img, XY_RED, box * 0.95)
    img = over(img, np.array([0.06, 0.06, 0.08], np.float32), box2 * 0.7)
    img = over(img, WHITE, paste_mask((H, W), m, x0 + w1 / 2, y0 + hgt / 2))
    return over(img, GOLD, paste_mask((H, W), m2, x0 + w1 + 10 + w2 / 2, y0 + hgt / 2))


def main(a, b, step=1):
    os.makedirs(OUT, exist_ok=True)
    for fi in range(a, b, step):
        p = f'{OUT}/f{fi:05d}.jpg'
        if os.path.exists(p):
            continue
        out = render_frame(fi)
        cv2.imwrite(p, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if fi % 24 == 0:
            print('frame', fi, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'still':
        for ts in sys.argv[2:]:
            fi = int(round(float(ts) * FPS))
            out = render_frame(fi)
            cv2.imwrite(f'{ROOT}/out/still_{float(ts):05.2f}.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            print('still', ts, flush=True)
    else:
        main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
