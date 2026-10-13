"""Chronica Mediae Terrae - procedural 40s film renderer.

Renders 2.5D parallax shots from painted plates + depth maps, per-chapter grading,
period film treatment, gilded Latin titles and page-turn / burn / ink transitions.
"""
import os, sys, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = f'{ROOT}/assets'
GEN = '/opt/cursor/artifacts/assets'
FONTS = f'{ROOT}/fonts'
W, H, FPS = 1920, 1080, 24
DURATION = 40.0
NFRAMES = int(DURATION * FPS)
BAR = int(round((H - W / 2.39) / 2))  # 2.39:1 letterbox

cv2.setNumThreads(1)


# ---------------------------------------------------------------- utilities
def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(e0, e1, x):
    t = clamp01((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    s = t * t * t * (t * (t * 6 - 15) + 10)
    return 0.55 * t + 0.45 * s


def lerp(a, b, t):
    return a + (b - a) * t


def fractal_noise(w, h, seed, octaves=5, base=6, persistence=0.55):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * (2 ** o)
        g = rng.random((max(2, int(n * h / w)) + 1, n + 1)).astype(np.float32)
        acc += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistence
    acc /= tot
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
    return acc


def load_plate(name):
    p = f'{ASSETS}/{name}_2x.png'
    if os.path.exists(p):
        img = cv2.imread(p)
    else:
        img = cv2.imread(f'{GEN}/{name}.jpg')
        img = cv2.resize(img, (2560, 1440), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_depth(name, shape):
    p = f'{ASSETS}/{name}_depth.png'
    if not os.path.exists(p):
        return np.zeros(shape, np.float32)
    d = cv2.imread(p, -1).astype(np.float32) / 65535.0
    d = cv2.resize(d, (shape[1], shape[0]))
    # dilate near layers so foreground silhouettes move as solid bodies
    d = cv2.dilate(d, np.ones((9, 9), np.uint8))
    d = cv2.GaussianBlur(d, (0, 0), 4)
    return d


# ---------------------------------------------------------------- grading
GRADES = {
    # sat, lift(rgb), gamma(rgb), gain(rgb), shadow tint, highlight tint, contrast
    'sepia':   dict(sat=0.35, lift=(0.05, 0.03, 0.01), gamma=(1.0, 1.05, 1.18), gain=(1.02, 0.92, 0.74),
                    sh=(0.05, 0.02, -0.02), hi=(0.04, 0.02, -0.04), con=1.12, expo=0.92),
    'amber':   dict(sat=0.92, lift=(0.035, 0.02, 0.0), gamma=(0.96, 1.0, 1.12), gain=(1.06, 0.97, 0.80),
                    sh=(0.03, 0.01, -0.02), hi=(0.05, 0.025, -0.05), con=1.10, expo=0.98),
    'emerald': dict(sat=0.88, lift=(0.0, 0.03, 0.03), gamma=(1.06, 0.98, 1.0), gain=(0.94, 1.01, 0.93),
                    sh=(-0.02, 0.02, 0.025), hi=(0.03, 0.03, -0.01), con=1.08, expo=0.97),
    'mist':    dict(sat=0.78, lift=(0.05, 0.06, 0.05), gamma=(1.02, 0.98, 1.04), gain=(0.97, 1.0, 0.90),
                    sh=(0.0, 0.015, 0.01), hi=(0.03, 0.03, 0.0), con=0.98, expo=1.0),
    'ice':     dict(sat=0.62, lift=(0.03, 0.045, 0.07), gamma=(1.02, 0.96, 0.88), gain=(0.96, 1.02, 1.08),
                    sh=(-0.03, 0.0, 0.04), hi=(0.0, 0.015, 0.03), con=1.10, expo=1.06),
    'violet':  dict(sat=0.72, lift=(0.04, 0.025, 0.06), gamma=(1.0, 1.06, 0.96), gain=(0.98, 0.93, 1.02),
                    sh=(0.03, -0.01, 0.05), hi=(0.02, 0.0, 0.03), con=1.06, expo=0.97),
    'crimson': dict(sat=0.86, lift=(0.05, 0.02, 0.01), gamma=(0.95, 1.02, 1.10), gain=(1.04, 0.91, 0.76),
                    sh=(0.045, 0.0, 0.0), hi=(0.04, 0.01, -0.04), con=1.08, expo=0.95),
}


def grade(img, g):
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    out = lum[..., None] + (img - lum[..., None]) * g['sat']
    out = out * g['expo']
    lift = np.array(g['lift'], np.float32); gain = np.array(g['gain'], np.float32)
    gam = np.array(g['gamma'], np.float32)
    out = clamp01(lift + out * (gain - lift))
    out = np.power(out, gam)
    l = lum[..., None]
    out = out + np.array(g['sh'], np.float32) * (1 - l) ** 2 + np.array(g['hi'], np.float32) * l ** 2
    c = g['con']
    out = clamp01(out)
    out = clamp01(0.5 + (out - 0.5) * c)
    s = out * out * (3 - 2 * out)
    return (out * 0.65 + s * 0.35).astype(np.float32)


# ---------------------------------------------------------------- particles
class Particles:
    def __init__(self, kind, seed, n):
        r = np.random.default_rng(seed)
        self.kind, self.n = kind, n
        self.x0 = r.random(n) * W
        self.y0 = r.random(n) * H
        self.z = r.random(n) ** 1.6  # 0 far .. 1 near
        self.ph = r.random(n) * 6.283
        self.fr = 0.4 + r.random(n) * 1.2
        self.rnd = r.random(n)

    def render(self, t):
        k, z = self.kind, self.z
        if k == 'snow':
            vx, vy = -40 - 90 * z, 60 + 170 * z
            amp = 12 + 25 * z
            x = self.x0 + vx * t + amp * np.sin(self.fr * t + self.ph)
            y = self.y0 + vy * t
            size = 0.9 + 4.6 * z ** 1.5
            br = 0.50 + 0.7 * z
            col = np.array([0.93, 0.96, 1.0])
        elif k == 'dust':
            x = self.x0 + (6 + 10 * z) * t + 18 * np.sin(0.5 * self.fr * t + self.ph)
            y = self.y0 - (4 + 6 * z) * t + 14 * np.cos(0.4 * self.fr * t + self.ph)
            size = 0.7 + 2.6 * z ** 2
            br = (0.18 + 0.45 * z) * (0.6 + 0.4 * np.sin(2 * self.fr * t + self.ph))
            col = np.array([1.0, 0.86, 0.55])
        elif k == 'motes':
            x = self.x0 + 22 * np.sin(0.35 * self.fr * t + self.ph) + 5 * t
            y = self.y0 - 10 * t + 16 * np.sin(0.5 * self.fr * t + 2 * self.ph)
            size = 1.2 + 3.0 * z
            br = clamp01(np.sin(1.3 * self.fr * t + self.ph)) ** 3 * (0.5 + 0.6 * z)
            col = np.array([0.85, 1.0, 0.55])
        elif k == 'wisps':
            x = self.x0 + 14 * t + 30 * np.sin(0.3 * self.fr * t + self.ph)
            y = self.y0 - 6 * t + 20 * np.sin(0.45 * self.fr * t + self.ph * 1.7)
            size = 1.0 + 3.5 * z
            br = (0.25 + 0.6 * z) * (0.5 + 0.5 * np.sin(1.1 * self.fr * t + self.ph))
            col = np.array([0.80, 0.92, 1.0])
        elif k == 'magic':
            x = self.x0 + 18 * np.sin(0.5 * self.fr * t + self.ph)
            y = self.y0 - (18 + 40 * z) * t
            size = 0.8 + 2.8 * z
            br = clamp01(np.sin(2.2 * self.fr * t + self.ph)) ** 4 * (0.6 + 0.6 * z)
            col = np.array([0.86, 0.80, 1.0])
        elif k == 'sand':
            x = self.x0 + (120 + 260 * z) * t
            y = self.y0 + 12 * np.sin(1.5 * self.fr * t + self.ph) + 8 * t
            size = 0.6 + 1.8 * z
            br = 0.20 + 0.35 * z
            col = np.array([1.0, 0.82, 0.55])
        elif k == 'embers':
            x = self.x0 + 30 * np.sin(0.8 * self.fr * t + self.ph) + 10 * t
            y = self.y0 - (40 + 110 * z) * t
            size = 0.9 + 2.6 * z
            br = (0.5 + 0.5 * np.sin(5 * self.fr * t + self.ph)) * (0.55 + 0.6 * z)
            col = np.array([1.0, 0.55, 0.18])
        else:
            raise ValueError(k)
        m = 60
        x = np.mod(x + m, W + 2 * m) - m
        y = np.mod(y + m, H + 2 * m) - m
        return x, y, np.broadcast_to(size, x.shape), np.broadcast_to(br, x.shape), col


def splat(layer, x, y, size, br, col, blur_scale=1.0):
    """Draw soft glowing points at half resolution into layer (H/2,W/2,3)."""
    h, w = layer.shape[:2]
    sharp = np.zeros((h, w), np.float32)
    soft = np.zeros((h, w), np.float32)
    for xi, yi, si, bi in zip(x * 0.5, y * 0.5, size * 0.5, br):
        if bi <= 0.01:
            continue
        tgt = soft if si > 1.1 else sharp
        cv2.circle(tgt, (int(xi * 4), int(yi * 4)), max(1, int(si * 4)), float(bi), -1, cv2.LINE_AA, 2)
    a = cv2.GaussianBlur(sharp, (0, 0), 0.8 * blur_scale) + cv2.GaussianBlur(soft, (0, 0), 2.2 * blur_scale)
    glow = cv2.GaussianBlur(a, (0, 0), 6)
    a = a + 0.5 * glow
    layer += a[..., None] * col[None, None, :].astype(np.float32)
    return layer


# ---------------------------------------------------------------- titles
FONT = {
    'black': f'{FONTS}/UnifrakturMaguntia-Book.ttf',
    'italic': f'{FONTS}/IMFeENit28P.ttf',
    'roman': f'{FONTS}/IMFeENrm28P.ttf',
    'deco': f'{FONTS}/CinzelDecorative-Regular.ttf',
    'decob': f'{FONTS}/CinzelDecorative-Bold.ttf',
    'uncial': f'{FONTS}/UncialAntiqua-Regular.ttf',
}

MATERIALS = {
    'gold': ((0.42, 0.28, 0.08), (1.0, 0.90, 0.58), (0.78, 0.58, 0.22)),
    'pale': ((0.70, 0.62, 0.48), (1.0, 0.97, 0.88), (0.90, 0.84, 0.70)),
    'red':  ((0.36, 0.04, 0.03), (0.86, 0.22, 0.12), (0.62, 0.10, 0.06)),
    'ink':  ((0.10, 0.06, 0.03), (0.30, 0.20, 0.12), (0.18, 0.11, 0.06)),
}


def material_fill(mask, mat, seed=0):
    h, w = mask.shape
    lo, hi, mid = [np.array(c, np.float32) for c in MATERIALS[mat]]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    band = np.where(yy < 0.45, lo + (hi - lo) * smoothstep(0, 0.45, yy), hi + (mid - hi) * smoothstep(0.45, 1, yy))
    rgb = np.broadcast_to(band, (h, w, 3)).copy()
    n = fractal_noise(max(w, 8), max(h, 8), seed, octaves=4, base=8)[:h, :w]
    rgb *= (0.88 + 0.24 * n)[..., None]
    if mat in ('gold', 'pale', 'red'):
        b = cv2.GaussianBlur(mask, (0, 0), 1.6)
        gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        light = clamp01(-(gx * 0.5 + gy * 0.85) * 2.5)
        dark = clamp01((gx * 0.5 + gy * 0.85) * 2.5)
        rgb = rgb + light[..., None] * 0.55 - dark[..., None] * 0.35
    return clamp01(rgb).astype(np.float32)


class Element:
    def __init__(self, mask, x0, y0, mat, wipe='lr', start=0.0, end=1.0, glow=0.5, shadow=0.75, seed=0):
        self.mask, self.x0, self.y0 = mask, x0, y0
        self.fill = material_fill(mask, mat, seed)
        self.glow_m = cv2.GaussianBlur(mask, (0, 0), 14)
        self.sh_m = cv2.GaussianBlur(mask, (0, 0), 5)
        self.wipe, self.start, self.end = wipe, start, end
        self.glow, self.shadow = glow, shadow
        self.glow_col = np.array(MATERIALS[mat][1], np.float32)


def text_element(text, font, size, cx, cy, mat, tracking=0, anchor='mm', **kw):
    f = ImageFont.truetype(FONT[font], size)
    pad = int(size * 0.6)
    chars = list(text)
    widths = [f.getlength(c) for c in chars]
    total = sum(widths) + tracking * (len(chars) - 1)
    asc, desc = f.getmetrics()
    w, h = int(total + 2 * pad), int(asc + desc + 2 * pad)
    img = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(img)
    x = pad
    for c, cw in zip(chars, widths):
        d.text((x, pad), c, font=f, fill=255)
        x += cw + tracking
    m = np.asarray(img, np.float32) / 255.0
    ys, xs = np.where(m > 0.01)
    by = (ys.min() + ys.max()) / 2
    x0 = int(cx - w / 2) if anchor[0] == 'm' else (int(cx - pad) if anchor[0] == 'l' else int(cx - w + pad))
    y0 = int(cy - by)
    return Element(m, x0, y0, mat, **kw)


def flourish_element(cx, cy, half, mat='gold', **kw):
    s = 4
    w, h = int(2 * half + 80), 70
    img = np.zeros((h * s, w * s), np.uint8)
    c = (w * s // 2, h * s // 2)
    th = int(1.6 * s)

    def P(x, y):
        return (int(c[0] + x * s), int(c[1] + y * s))

    for sign in (-1, 1):
        cv2.line(img, P(sign * 22, 0), P(sign * (half - 40), 0), 255, th, cv2.LINE_AA)
        cv2.line(img, P(sign * 30, 5), P(sign * (half * 0.55), 5), 255, max(1, th // 2), cv2.LINE_AA)
        pts = []
        for i in range(120):
            u = i / 119
            a = u * 2.4 * math.pi
            r = 15 * (1 - u) + 1.2
            pts.append(P(sign * (half - 40 + 15 - r * math.cos(a)) , -r * math.sin(a) + 0))
        cv2.polylines(img, [np.array(pts, np.int32)], False, 255, th, cv2.LINE_AA)
        # leaf
        leaf = np.array([P(sign * 50, 0), P(sign * 62, -8), P(sign * 78, -2), P(sign * 62, 4)], np.int32)
        cv2.fillPoly(img, [leaf], 255, cv2.LINE_AA)
        cv2.circle(img, P(sign * (half * 0.55 + 8), 5), int(2.5 * s), 255, -1, cv2.LINE_AA)
    lozenge = np.array([P(0, -11), P(11, 0), P(0, 11), P(-11, 0)], np.int32)
    cv2.fillPoly(img, [lozenge], 255, cv2.LINE_AA)
    inner = np.array([P(0, -5), P(5, 0), P(0, 5), P(-5, 0)], np.int32)
    cv2.fillPoly(img, [inner], 0, cv2.LINE_AA)
    m = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return Element(m, int(cx - w / 2), int(cy - h / 2), mat, wipe='center', **kw)


class Card:
    """A set of gilded elements with staggered calligraphic reveal."""

    def __init__(self, elements, t_in, t_full, t_out0, t_out1, backing=0.45, backing_box=None):
        self.els, self.t_in, self.t_full = elements, t_in, t_full
        self.t_out0, self.t_out1 = t_out0, t_out1
        self.backing = backing
        if backing_box is None:
            x0 = min(e.x0 for e in elements); y0 = min(e.y0 for e in elements)
            x1 = max(e.x0 + e.mask.shape[1] for e in elements); y1 = max(e.y0 + e.mask.shape[0] for e in elements)
            backing_box = (x0, y0, x1, y1)
        x0, y0, x1, y1 = backing_box
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2 * 1.15 + 40, (y1 - y0) / 2 * 1.2 + 40
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        self.back = np.exp(-d * 1.6).astype(np.float32)

    def composite(self, frame, t):
        if t < self.t_in or t > self.t_out1:
            return frame
        fade = 1.0 - smoothstep(self.t_out0, self.t_out1, t)
        p = (t - self.t_in) / (self.t_full - self.t_in)
        bk = smoothstep(0, 0.4, p) * fade * self.backing
        if bk > 0:
            frame *= (1 - bk * self.back)[..., None]
        for e in self.els:
            lp = clamp01((p - e.start) / (e.end - e.start))
            if lp <= 0:
                continue
            h, w = e.mask.shape
            xs = np.arange(w, dtype=np.float32)
            if e.wipe == 'lr':
                pos = lp * (w + 120) - 60
                wv = clamp01((pos - xs) / 60.0 + 0.5)
            elif e.wipe == 'center':
                dist = np.abs(xs - w / 2)
                pos = lp * (w / 2 + 80)
                wv = clamp01((pos - dist) / 50.0 + 0.5)
            else:
                wv = np.full(w, smoothstep(0, 1, lp), np.float32)
            wv = wv[None, :] * fade * smoothstep(0, 0.35, lp)
            x0, y0 = e.x0, e.y0
            fx0, fy0 = max(0, x0), max(0, y0)
            fx1, fy1 = min(W, x0 + w), min(H, y0 + h)
            if fx1 <= fx0 or fy1 <= fy0:
                continue
            sl = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
            reg = frame[fy0:fy1, fx0:fx1]
            wvs = wv[:, sl[1]]
            shadow = e.sh_m[sl] * wvs * e.shadow
            reg *= (1 - shadow * 0.85)[..., None]
            glow = e.glow_m[sl] * wvs * e.glow
            reg += glow[..., None] * e.glow_col * 0.55
            a = (e.mask[sl] * wvs)[..., None]
            reg[:] = reg * (1 - a) + e.fill[sl] * a
        return frame


# ---------------------------------------------------------------- shots
LAT = {
    'prol1': 'Antiquis temporibus,',
    'prol2': 'cum mundus adhuc iuvenis erat\u2026',
}

CHAPTERS = [
    dict(key='ch1', plate='plate_ch1_humans', t0=3.6, t1=10.4, grade='amber', parts='dust', npart=260,
         subject=(0.39, 0.40),
         cam0=dict(cx=0.42, cy=0.53, z=1.12, dolly=0.0, px=0, py=0),
         cam1=dict(cx=0.40, cy=0.47, z=1.30, dolly=0.22, px=-14, py=-8),
         num='I', title='Terra Hominum', sub='Ubi homines agros colunt', pos=(0.70, 0.33),
         ambient=dict(wheat=True, clouds=0.020)),
    dict(key='ch2', plate='plate_ch2_elves', t0=9.6, t1=15.6, grade='emerald', parts='motes', npart=140,
         subject=(0.58, 0.55),
         cam0=dict(cx=0.52, cy=0.60, z=1.20, dolly=0.0, px=0, py=10),
         cam1=dict(cx=0.55, cy=0.45, z=1.26, dolly=0.08, px=0, py=-34),
         num='II', title='Silva Aeterna', sub='Ubi primogeniti sub arbore vivunt', pos=(0.30, 0.70),
         ambient=dict(water=(0.62, 1.0), clouds=0.0)),
    dict(key='ch3', plate='plate_ch3_kingdom', t0=14.8, t1=20.6, grade='mist', parts='wisps', npart=70,
         subject=(0.44, 0.52),
         cam0=dict(cx=0.41, cy=0.56, z=1.22, dolly=0.0, px=-30, py=0),
         cam1=dict(cx=0.55, cy=0.54, z=1.24, dolly=0.04, px=34, py=0),
         num='III', title='Regnum Altum', sub='Tres viatores, una via', pos=(0.68, 0.30),
         ambient=dict(clouds=0.025, haze=0.10)),
    dict(key='ch4', plate='plate_ch4_dwarves', t0=19.8, t1=26.4, grade='ice', parts='snow', npart=900,
         subject=(0.70, 0.36),
         cam0=dict(cx=0.60, cy=0.52, z=1.06, dolly=0.0, px=0, py=0),
         cam1=dict(cx=0.655, cy=0.50, z=1.22, dolly=0.12, px=-10, py=-6),
         num='IV', title='Sub Montibus', sub='Ubi nani in saxo regnant', pos=(0.29, 0.30),
         ambient=dict(water=(0.80, 1.0), clouds=0.012)),
    dict(key='ch5', plate='plate_ch5_isle', t0=25.6, t1=32.2, grade='violet', parts='magic', npart=220,
         subject=(0.54, 0.45),
         cam0=dict(cx=0.46, cy=0.50, z=1.16, dolly=0.0, px=44, py=0),
         cam1=dict(cx=0.54, cy=0.46, z=1.27, dolly=0.06, px=-44, py=-10),
         num='V', title='Insula Caelestis', sub='Magia antiqua aerem tenet', pos=(0.50, 0.70),
         ambient=dict(water=(0.64, 0.92), clouds=0.02, mist=True)),
    dict(key='ch6', plate='plate_ch6_desert', t0=31.4, t1=37.6, grade='crimson', parts='sand', npart=320,
         subject=(0.52, 0.42),
         cam0=dict(cx=0.46, cy=0.56, z=1.42, dolly=0.16, px=0, py=0),
         cam1=dict(cx=0.50, cy=0.51, z=1.14, dolly=0.0, px=16, py=0),
         num='VI', title='Finis Itineris', sub='Post arenas, aqua vitae', pos=(0.72, 0.30),
         ambient=dict(heat=(0.30, 0.52), clouds=0.015)),
]

PROLOGUE = dict(t0=0.0, t1=4.4)
FINALE = dict(t0=36.8, t1=40.0)

TRANSITIONS = [  # (t_start, t_end, kind, from_index, to_index) indices into SEQ
    (3.6, 4.4, 'burn', 0, 1),
    (9.6, 10.4, 'page', 1, 2),
    (14.8, 15.6, 'ink', 2, 3),
    (19.8, 20.6, 'burn', 3, 4),
    (25.6, 26.4, 'page', 4, 5),
    (31.4, 32.2, 'page', 5, 6),
    (36.8, 37.6, 'burn', 6, 7),
]


def chapter_card(ch):
    px, py = ch['pos']
    cx, cy = px * W, py * H
    els = [
        text_element(f'CAPITULUM \u00b7 {ch["num"]}', 'decob', 34, cx, cy - 80, 'pale', tracking=7,
                     start=0.0, end=0.35, glow=0.55, seed=1, shadow=0.9),
        text_element(ch['title'], 'black', 112, cx, cy, 'gold', start=0.12, end=0.72, glow=0.6, seed=2),
        flourish_element(cx, cy + 66, 230, start=0.45, end=0.85, glow=0.4, seed=3),
        text_element(ch['sub'], 'italic', 42, cx, cy + 112, 'pale', start=0.55, end=1.0, glow=0.3, seed=4),
    ]
    t0, t1 = ch['t0'], ch['t1']
    return Card(els, t0 + 0.75, t0 + 2.6, t1 - 1.55, t1 - 0.95, backing=0.42)


def prologue_card():
    cx, cy = W * 0.5, H * 0.47
    els = [
        text_element('A', 'black', 150, cx - 300, cy - 34, 'red', start=0.0, end=0.22, glow=0.5, seed=10, wipe='fade'),
        text_element('ntiquis temporibus,', 'italic', 76, cx - 248, cy - 40, 'gold', anchor='lm', start=0.12,
                     end=0.55, glow=0.5, seed=11),
        text_element(LAT['prol2'], 'italic', 56, cx + 20, cy + 52, 'pale', start=0.5, end=0.95, glow=0.35, seed=12),
    ]
    return Card(els, 0.7, 3.0, 3.25, 3.75, backing=0.5)


def finale_card():
    cx, cy = W * 0.5, H * 0.45
    tx = cx - 116
    els = [
        text_element('C', 'black', 190, tx - 418, cy - 4, 'red', start=0.0, end=0.18, glow=0.4, seed=20, wipe='fade'),
        text_element('hronica Mediae Terrae', 'black', 118, tx - 352, cy, 'ink', anchor='lm', start=0.08, end=0.6,
                     glow=0.0, seed=21, shadow=0.25),
        flourish_element(cx, cy + 86, 300, mat='red', start=0.45, end=0.8, glow=0.2, seed=22, shadow=0.4),
        text_element('Fabula de hominibus, primogenitis et nanis', 'italic', 44, cx, cy + 140, 'ink',
                     start=0.55, end=0.95, glow=0.0, seed=23, shadow=0.15),
        text_element('\u2014  FINIS  \u2014', 'deco', 30, cx, cy + 210, 'red', tracking=8, start=0.8, end=1.0,
                     glow=0.1, seed=24, shadow=0.2),
    ]
    return Card(els, 37.4, 38.75, 41.0, 42.0, backing=0.0)


# ---------------------------------------------------------------- renderer
class Film:
    def __init__(self):
        self.uu, self.vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.plates, self.depths, self.Df = {}, {}, {}
        for ch in CHAPTERS:
            p = load_plate(ch['plate'])
            self.plates[ch['key']] = p
            d = load_depth(ch['plate'], p.shape[:2])
            self.depths[ch['key']] = d
            sx, sy = ch['subject']
            self.Df[ch['key']] = float(d[int(sy * d.shape[0]), int(sx * d.shape[1])])
            ch['card'] = chapter_card(ch)
            ch['particles'] = Particles(ch['parts'], hash(ch['key']) % 1000, ch['npart'])
        self.map = load_plate('map_parchment')
        self.parch = load_plate('parchment_blank')
        self.parch_small = cv2.resize(self.parch, (W, H), interpolation=cv2.INTER_AREA)
        self.parch_tex = cv2.cvtColor(self.parch_small, cv2.COLOR_RGB2GRAY)
        self.parch_tex = (self.parch_tex / self.parch_tex.mean()).astype(np.float32)
        self.prol_card = prologue_card()
        self.fin_card = finale_card()
        self.embers = Particles('embers', 77, 160)
        self.trans_embers = Particles('embers', 78, 320)
        # noise fields for transitions & ambient
        self.n_burn = cv2.resize(fractal_noise(480, 270, 5, octaves=6, base=3), (W, H), interpolation=cv2.INTER_CUBIC)
        self.n_fine = cv2.resize(fractal_noise(960, 540, 6, octaves=4, base=24), (W, H), interpolation=cv2.INTER_CUBIC)
        self.n_ink = cv2.resize(fractal_noise(480, 270, 7, octaves=6, base=4, persistence=0.62), (W, H),
                                interpolation=cv2.INTER_CUBIC)
        self.n_mist = fractal_noise(1280, 360, 8, octaves=5, base=5)
        self.n_amb = fractal_noise(640, 360, 9, octaves=4, base=6)
        yy = self.vv / H
        xx = self.uu / W
        self.vign = (1 - 0.62 * clamp01(((xx - 0.5) ** 2 * 1.25 + (yy - 0.5) ** 2 * 1.6) * 2.1) ** 1.35).astype(np.float32)

    # -------- one painted shot with 2.5D camera
    def shot(self, ch, t):
        key = ch['key']
        plate, depth = self.plates[key], self.depths[key]
        PH, PW = plate.shape[:2]
        u = ease((t - ch['t0']) / (ch['t1'] - ch['t0']))
        c = {k: lerp(ch['cam0'][k], ch['cam1'][k], u) for k in ch['cam0']}
        k = PW / W / c['z']
        # gate weave
        wx = 0.35 * math.sin(t * 7.3) + 0.25 * math.sin(t * 13.1)
        wy = 0.30 * math.sin(t * 5.9 + 1.0)
        ccx, ccy = c['cx'] * PW, c['cy'] * PH
        bx = ccx + (self.uu - W / 2 + wx) * k
        by = ccy + (self.vv - H / 2 + wy) * k
        Df = self.Df[key]
        sx, sy = bx, by
        for _ in range(2):
            D = cv2.remap(depth, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            dd = D - Df
            sc = 1.0 / (1.0 + c['dolly'] * dd * 2.2)
            sx = ccx + (bx - ccx) * sc - c['px'] * k * dd * 2.0
            sy = ccy + (by - ccy) * sc - c['py'] * k * dd * 2.0
        amb = ch['ambient']
        far = clamp01(1 - D / 0.10)
        if amb.get('clouds'):
            sky = far * clamp01(1 - (sy / PH) / 0.42)
            sx = sx - amb['clouds'] * PW * (t - ch['t0']) * sky / 6.0
        if amb.get('wheat'):
            ny = sy / PH
            m = smoothstep(0.55, 0.75, ny) * clamp01(D * 1.4)
            sway = np.sin(sx * 0.018 + sy * 0.006 - t * 2.4) + 0.5 * np.sin(sx * 0.043 - t * 3.7 + sy * 0.01)
            sx = sx + m * sway * 2.6
            sy = sy + m * np.sin(sx * 0.02 - t * 2.0) * 0.8
        if amb.get('water'):
            y0, y1 = amb['water']
            ny = sy / PH
            wm = smoothstep(y0, y0 + 0.03, ny) * (1 - smoothstep(y1 - 0.03, y1, ny))
            sh = np.sin(sy * 0.35 + t * 3.1 + np.sin(sx * 0.02 + t) * 2.0) * 1.6
            sx = sx + wm * sh
            sy = sy + wm * np.sin(sx * 0.05 + t * 2.3) * 0.6
        if amb.get('heat'):
            y0, y1 = amb['heat']
            ny = sy / PH
            hm = smoothstep(y0, y0 + 0.06, ny) * (1 - smoothstep(y1 - 0.08, y1, ny))
            sx = sx + hm * np.sin(sy * 0.09 + t * 7.0 + np.sin(sx * 0.013) * 3) * 1.8
            sy = sy + hm * np.sin(sx * 0.05 + t * 5.0) * 0.9
        img = cv2.remap(plate, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        img = clamp01(img)
        if amb.get('mist'):
            off = (t * 38.0) % self.n_mist.shape[1]
            m = np.roll(self.n_mist, -int(off), axis=1)
            m = cv2.resize(m[:, :640], (W, H // 2), interpolation=cv2.INTER_CUBIC)
            mm = np.zeros((H, W), np.float32)
            mm[int(H * 0.38):int(H * 0.38) + H // 2] = m
            band = smoothstep(0.30, 0.48, self.vv / H) * (1 - smoothstep(0.62, 0.86, self.vv / H))
            fog = clamp01(mm * 1.4 - 0.35) * band * 0.38
            img = img * (1 - fog[..., None]) + np.array([0.92, 0.93, 1.0], np.float32) * fog[..., None]
        if amb.get('haze'):
            hz = amb['haze'] * (1 - smoothstep(0.25, 0.8, D))
            img = img * (1 - hz[..., None]) + np.array([0.86, 0.9, 0.86], np.float32) * hz[..., None]
        img = grade(img, GRADES[ch['grade']])
        # particles
        lay = np.zeros((H // 2, W // 2, 3), np.float32)
        x, y, s, b, col = ch['particles'].render(t - ch['t0'] + 3.0)
        lay = splat(lay, x, y, s, b, col)
        lay = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)
        if ch['parts'] == 'snow':
            img = img + lay * (1 - img) * 1.0
        else:
            img = img + lay * (1 - img * 0.6)
        if key == 'ch1':
            img = self.sunrays(img, t, (0.15, -0.1), (1.0, 0.85, 0.55), 0.16)
        if key == 'ch6':
            img = self.sunrays(img, t, (0.85, -0.05), (1.0, 0.6, 0.35), 0.14)
        img = ch['card'].composite(img, t)
        return clamp01(img)

    def sunrays(self, img, t, src, col, strength):
        ang = np.arctan2(self.vv / H - src[1], self.uu / W - src[0])
        r = np.sqrt((self.vv / H - src[1]) ** 2 + (self.uu / W - src[0]) ** 2)
        rays = 0.5 + 0.5 * np.sin(ang * 23 + t * 0.15) * np.sin(ang * 37 - t * 0.1 + 1.3)
        fall = np.exp(-r * 1.6)
        a = rays * fall * strength
        return img + a[..., None] * np.array(col, np.float32) * (1 - img)

    def prologue(self, t):
        u = ease((t - 0.0) / 4.4)
        z = lerp(1.12, 1.42, u)
        cx, cy = lerp(0.53, 0.47, u), lerp(0.50, 0.44, u)
        PH, PW = self.map.shape[:2]
        k = PW / W / z
        rot = math.radians(lerp(-1.2, 0.8, u))
        du, dv = self.uu - W / 2, self.vv - H / 2
        bx = cx * PW + (du * math.cos(rot) - dv * math.sin(rot)) * k
        by = cy * PH + (du * math.sin(rot) + dv * math.cos(rot)) * k
        img = cv2.remap(self.map, bx, by, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        img = grade(clamp01(img), GRADES['sepia'])
        # candlelight pool, flickering
        fl = 0.92 + 0.05 * math.sin(t * 9.1) + 0.03 * math.sin(t * 23.7 + 1) + 0.02 * math.sin(t * 41)
        r = np.sqrt(((self.uu / W - 0.42) * 1.3) ** 2 + (self.vv / H - 0.62) ** 2)
        light = (0.28 + 0.95 * np.exp(-r * r * 3.2)) * fl
        img = img * light[..., None] * np.array([1.08, 0.95, 0.78], np.float32)
        lay = np.zeros((H // 2, W // 2, 3), np.float32)
        x, y, s, b, col = self.embers.render(t + 4)
        lay = splat(lay, x, y, s, b * 0.6, col)
        img = img + cv2.resize(lay, (W, H)) * 0.9
        img = self.prol_card.composite(img, t)
        img *= smoothstep(0.0, 0.9, t)
        return clamp01(img)

    def finale(self, t):
        u = ease((t - 36.8) / 3.2)
        z = lerp(1.04, 1.10, u)
        PH, PW = self.parch.shape[:2]
        k = PW / W / z
        bx = 0.5 * PW + (self.uu - W / 2) * k
        by = 0.5 * PH + (self.vv - H / 2) * k
        img = cv2.remap(self.parch, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = grade(clamp01(img), GRADES['sepia'])
        fl = 0.96 + 0.03 * math.sin(t * 8.3) + 0.02 * math.sin(t * 19.1)
        r = np.sqrt(((self.uu / W - 0.5) * 1.2) ** 2 + (self.vv / H - 0.5) ** 2)
        img = img * ((0.42 + 0.40 * np.exp(-r * r * 2.6)) * fl)[..., None]
        img = self.fin_card.composite(img, t)
        lay = np.zeros((H // 2, W // 2, 3), np.float32)
        x, y, s, b, col = self.embers.render(t)
        lay = splat(lay, x, y, s, b * 0.45, col)
        img = img + cv2.resize(lay, (W, H)) * 0.8
        # final burn to black from edges
        if t > 39.15:
            p = smoothstep(39.15, 39.97, t)
            n = self.n_burn * 0.55 + (1 - self.vign) * 0.9
            e = (1.15 - p * 1.4) - n
            img = self.burn_mix(img, np.zeros_like(img), e, t)
        return clamp01(img)

    # -------- transitions
    def burn_mix(self, A, B, e, t, glow=1.0):
        # e > 0: A intact; e < 0: burned through to B
        fine = (self.n_fine - 0.5) * 0.06 * (1 + 0.4 * math.sin(t * 17))
        e = e + fine
        # charred, curling brown-black band ahead of the flame
        char = smoothstep(0.012, 0.055, e)
        scorch = smoothstep(0.0, 0.16, e)
        burnt_col = np.array([0.035, 0.018, 0.008], np.float32)
        brown = np.array([0.30, 0.17, 0.07], np.float32)
        Ac = A * (0.35 + 0.65 * scorch)[..., None] + brown * (1 - scorch[..., None]) * 0.25
        Ac = Ac * char[..., None] + burnt_col * (1 - char[..., None])
        reveal = smoothstep(-0.004, 0.003, -e)
        out = Ac * (1 - reveal[..., None]) + B * reveal[..., None]
        line = np.exp(-((e - 0.006) / 0.0045) ** 2)
        flick = 0.75 + 0.25 * np.sin(self.n_fine * 40 + t * 23)
        halo = np.exp(-(np.abs(e) / 0.03) ** 2)
        g = (line * 0.95 * flick + halo * 0.10) * glow
        fire = np.array([1.0, 0.40, 0.07], np.float32)
        hot = np.array([1.0, 0.78, 0.40], np.float32)
        out = out + g[..., None] * fire + (line ** 4 * flick)[..., None] * hot * 0.45 * glow
        return clamp01(out)

    def transition(self, kind, A, B, p, t):
        if kind == 'burn':
            T = lerp(-0.08, 1.12, ease(p))
            n = self.n_burn * 0.75 + (self.uu / W) * 0.12 + (1 - self.vv / H) * 0.13
            n = (n - n.min()) / (n.max() - n.min())
            out = self.burn_mix(A, B, n - T, t)
            lay = np.zeros((H // 2, W // 2, 3), np.float32)
            x, y, s, b, col = self.trans_embers.render(t * 1.4)
            # embers only near the burning front
            xi = np.clip(x.astype(int), 0, W - 1); yi = np.clip(y.astype(int), 0, H - 1)
            near = np.exp(-((n[yi, xi] - T) / 0.12) ** 2)
            lay = splat(lay, x, y, s * 1.2, b * near * 1.6, col)
            return clamp01(out + cv2.resize(lay, (W, H)))
        if kind == 'ink':
            T = lerp(-0.15, 1.15, ease(p))
            warp = (self.n_amb - 0.5)
            warp = cv2.resize(warp, (W, H))
            n = self.n_ink * 0.8 + warp * 0.25 + np.abs(self.uu / W - 0.5) * 0.15
            n = (n - n.min()) / (n.max() - n.min())
            e = n - T
            ink = np.array([0.07, 0.045, 0.03], np.float32)
            e = e + (self.n_fine - 0.5) * 0.03
            inkA = smoothstep(0.0, 0.025, e)  # 1 = A visible
            showB = smoothstep(0.03, 0.09, -e)  # 1 = B visible
            mid = ink * np.ones_like(A)
            # wet sepia halo soaking into A ahead of the ink front
            soak = np.exp(-(np.maximum(e, 0) / 0.05) ** 2) * (e > 0)
            Aw = A * (1 - 0.45 * soak[..., None]) + np.array([0.28, 0.17, 0.08], np.float32) * 0.25 * soak[..., None]
            sideA = Aw * inkA[..., None] + mid * (1 - inkA[..., None])
            sepB = B * np.array([1.0, 0.85, 0.65], np.float32) * 0.6
            sideB = mid * (1 - showB[..., None]) + B * showB[..., None] + sepB * (showB * (1 - showB) * 1.2)[..., None]
            out = np.where((e >= 0)[..., None], sideA, sideB)
            return clamp01(out)
        if kind == 'page':
            return self.page_turn(A, B, ease(p))
        raise ValueError(kind)

    def page_turn(self, A, B, p):
        ang = math.radians(14)
        ca, sa = math.cos(ang), math.sin(ang)
        # rotated coordinate along curl direction
        X = (self.uu - W) * ca + (self.vv - H / 2) * sa  # <= ~0 on page
        Xmin = float(X.min())
        r = 230.0
        xf = lerp(40.0, Xmin - math.pi * r - 60, p)
        Y = -(self.uu - W) * sa + (self.vv - H / 2) * ca
        out = B.copy()
        # shadow on B beyond curl
        dist = X - (xf + r)
        shadow = np.where(dist > 0, 1 - 0.55 * np.exp(-dist / 90.0), 1.0)
        out *= shadow[..., None].astype(np.float32)

        def sample(img, Xp):
            # map rotated coord back to pixel coords
            uu = W + Xp * ca - Y * sa
            vv = H / 2 + Xp * sa + Y * ca
            return cv2.remap(img, uu.astype(np.float32), vv.astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)

        page_max = 0.0  # page right edge in X
        # flat unturned A
        flatA = X < xf
        # front of cylinder
        s1 = clamp01((X - xf) / r)
        th1 = np.arcsin(s1)
        p1 = xf + r * th1
        front = (X >= xf) & (X <= xf + r) & (p1 <= page_max)
        # back of cylinder
        th2 = math.pi - th1
        p2 = xf + r * th2
        back_cyl = (X >= xf) & (X <= xf + r) & (p2 <= page_max)
        # flat backside lying over A
        p3 = xf + math.pi * r + (xf - X)
        back_flat = (X < xf) & (p3 <= page_max)
        A_flat = A
        Afront = sample(A, p1)
        m = flatA[..., None]
        out = np.where(m, A_flat, out)
        shade1 = (0.75 + 0.25 * np.cos(th1)).astype(np.float32)
        out = np.where(front[..., None], Afront * shade1[..., None], out)
        # backside: parchment with faint mirrored bleed-through of A
        def backside(Pp, shade):
            bp = sample(self.parch_small, Pp)
            ba = sample(A, Pp)
            lum = ba @ np.array([0.3, 0.59, 0.11], np.float32)
            lum = cv2.GaussianBlur(lum, (0, 0), 2.5)
            b = bp * (0.42 + 0.58 * lum[..., None]) * np.array([0.92, 0.86, 0.74], np.float32)
            return b * shade[..., None]
        shade2 = (0.45 + 0.55 * np.sin(th2) ** 0.7).astype(np.float32)
        bc = backside(p2, shade2)
        out = np.where(back_cyl[..., None], bc, out)
        d3 = np.maximum(xf - X, 0)
        shade3 = (0.58 + 0.42 * (1 - np.exp(-d3 / 260))).astype(np.float32) * (0.92 + 0.08 * np.exp(-d3 / 900))
        bf = backside(p3, shade3)
        out = np.where(back_flat[..., None], bf, out)
        # contact shadow of lifted page on A
        cs = np.where(flatA & ~back_flat, 1 - 0.35 * np.exp(-(xf - X) / 40.0), 1.0)
        out *= cs[..., None].astype(np.float32)
        return clamp01(out.astype(np.float32))

    # -------- per-frame
    def seq_render(self, idx, t):
        if idx == 0:
            return self.prologue(t)
        if idx == 7:
            return self.finale(t)
        return self.shot(CHAPTERS[idx - 1], t)

    def frame(self, fi):
        t = fi / FPS
        img = None
        for (ta, tb, kind, a, b) in TRANSITIONS:
            if ta <= t < tb:
                A = self.seq_render(a, t)
                B = self.seq_render(b, t)
                img = self.transition(kind, A, B, (t - ta) / (tb - ta), t)
                break
        if img is None:
            bounds = [(0, PROLOGUE['t0'], PROLOGUE['t1'])] + \
                     [(i + 1, c['t0'], c['t1']) for i, c in enumerate(CHAPTERS)] + [(7, FINALE['t0'], FINALE['t1'])]
            for idx, a, b in bounds:
                if a <= t < b:
                    img = self.seq_render(idx, t)
                    break
        return self.post(img, fi)

    def post(self, img, fi):
        t = fi / FPS
        # bloom
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        hi = clamp01(small - 0.62) * 1.6
        bl = cv2.GaussianBlur(hi, (0, 0), 9) + cv2.GaussianBlur(hi, (0, 0), 28)
        img = img + cv2.resize(bl, (W, H)) * np.array([1.0, 0.88, 0.70], np.float32) * 0.32
        # halation-like soft diffusion (old lens)
        soft = cv2.GaussianBlur(img, (0, 0), 2.2)
        img = img * 0.82 + soft * 0.18
        # aged varnish: warm the highlights, lift blacks into umber
        lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
        img = img + (lum ** 2)[..., None] * np.array([0.02, 0.005, -0.035], np.float32)
        img = img * 0.97 + np.array([0.022, 0.014, 0.006], np.float32)
        # parchment / canvas texture overlay
        tex = self.parch_tex[..., None]
        img = img * (0.93 + 0.07 * tex)
        # vignette + exposure flicker
        flick = 1.0 + 0.012 * math.sin(fi * 2.1) + 0.008 * math.sin(fi * 0.77 + 2)
        img = img * self.vign[..., None] * flick
        # film grain (luma-weighted)
        rng = np.random.default_rng(1000 + fi)
        g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
        g = cv2.resize(cv2.GaussianBlur(g, (0, 0), 0.6), (W, H), interpolation=cv2.INTER_LINEAR)
        lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
        img = img + (g * 0.028 * (0.35 + 0.65 * (1 - np.abs(lum - 0.45) * 1.6).clip(0.2, 1)))[..., None]
        img = clamp01(img)
        img[:BAR] = 0
        img[H - BAR:] = 0
        return (img * 255 + 0.5).astype(np.uint8)


_FILM = None


def _init():
    global _FILM
    _FILM = Film()


def _work(fi):
    return fi, _FILM.frame(fi)


def render_stills(times, outdir):
    f = Film()
    os.makedirs(outdir, exist_ok=True)
    for t in times:
        fi = int(round(t * FPS))
        cv2.imwrite(f'{outdir}/frame_{t:05.2f}.png', cv2.cvtColor(f.frame(fi), cv2.COLOR_RGB2BGR))
        print('still', t, flush=True)


def render_video(out, start=0, end=NFRAMES, workers=3):
    import multiprocessing as mp
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
           '-pix_fmt', 'yuv420p', '-tune', 'grain', out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with mp.get_context('fork').Pool(workers, initializer=_init) as pool:
        for fi, fr in pool.imap(_work, range(start, end), chunksize=2):
            proc.stdin.write(fr.tobytes())
            if fi % 24 == 0:
                print('frame', fi, flush=True)
    proc.stdin.close()
    proc.wait()


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'still':
        render_stills([float(x) for x in sys.argv[2].split(',')], sys.argv[3] if len(sys.argv) > 3 else f'{ROOT}/keyframe')
    elif mode == 'video':
        s = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        e = int(sys.argv[4]) if len(sys.argv) > 4 else NFRAMES
        render_video(sys.argv[2], s, e)
