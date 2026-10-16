"""Kinetic typography and particle layers shared by the promo renderers (1080x1920)."""
import math, functools
import numpy as np, cv2
from kit import *

R = math.radians


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


@functools.lru_cache(maxsize=256)
def tmask(text, fname, size, tracking=0, weight=None):
    return text_mask(text, fname, size, tracking, weight)


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


DRAW = dict(slam=draw_slam, line=draw_line, en=draw_en, num=draw_num, vert=draw_vert, badge=draw_badge)
