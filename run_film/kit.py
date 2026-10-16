"""Shared compositing kit for the Xiangya marathon promo (1080x1920 vertical)."""
import os, math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
A = f'{ROOT}/assets'
F = f'{ROOT}/fonts'
W, H = 1080, 1920

FONTS = {
    'sport': f'{F}/SmileySans-Oblique.ttf',
    'sans': f'{F}/NotoSansSC.ttf',
    'serif': f'{F}/NotoSerifSC.ttf',
    'num': f'{F}/Anton-Regular.ttf',
}

XY_RED = np.array([0.78, 0.06, 0.10], np.float32)
GOLD = np.array([1.0, 0.80, 0.36], np.float32)
WHITE = np.array([1.0, 1.0, 1.0], np.float32)


def clamp01(x):
    return np.clip(x, 0, 1)


def smoothstep(a, b, x):
    t = clamp01((x - a) / (b - a))
    return t * t * (3 - 2 * t)


def font(name, size, weight=None):
    f = ImageFont.truetype(FONTS[name], size)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


def load_plate(name):
    p = f'{A}/{name}_2x.png'
    img = cv2.imread(p) if os.path.exists(p) else cv2.imread(f'{A}/{name}.jpg')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_mask(name, shape):
    p = f'{A}/{name}_mask.png'
    if not os.path.exists(p):
        return np.zeros(shape, np.float32)
    m = cv2.imread(p, 0).astype(np.float32) / 255.0
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def cover(img, z=1.0, cx=0.5, cy=0.5, rot=0.0, out=(W, H), shake=(0, 0)):
    """Sample a plate to fill the output frame with zoom/pan/rotation."""
    ph, pw = img.shape[:2]
    ow, oh = out
    s = max(ow / pw, oh / ph) * z
    M = cv2.getRotationMatrix2D((pw * cx, ph * cy), math.degrees(rot), s)
    M[0, 2] += ow / 2 - pw * cx + shake[0]
    M[1, 2] += oh / 2 - ph * cy + shake[1]
    return cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def text_mask(text, fname, size, tracking=0, weight=None, pad=None):
    f = font(fname, size, weight)
    pad = pad if pad is not None else int(size * 0.35)
    widths = [f.getlength(c) for c in text]
    total = sum(widths) + tracking * max(0, len(text) - 1)
    asc, desc = f.getmetrics()
    w, h = int(total + 2 * pad), int(asc + desc + 2 * pad)
    img = Image.new('L', (max(w, 1), max(h, 1)), 0)
    d = ImageDraw.Draw(img)
    x = pad
    for c, cw in zip(text, widths):
        d.text((x, pad), c, font=f, fill=255)
        x += cw + tracking
    m = np.asarray(img, np.float32) / 255.0
    ys, xs = np.where(m > 0.02)
    if len(ys):
        m = m[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5]
    return m


def paste_mask(canvas_shape, m, cx, cy, scale=1.0, rot=0.0):
    """Place a mask centered at (cx, cy) on a full-frame alpha layer."""
    h, w = m.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), math.degrees(rot), scale)
    M[0, 2] += cx - w / 2
    M[1, 2] += cy - h / 2
    return cv2.warpAffine(m, M, (canvas_shape[1], canvas_shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def over(dst, color, alpha):
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    col = color if isinstance(color, np.ndarray) and color.ndim == 3 else np.asarray(color, np.float32)
    return dst * (1 - a) + col * a


def gradient_fill(alpha, top, bottom, y0=None, y1=None):
    ys, xs = np.where(alpha > 0.02)
    if not len(ys):
        return np.zeros(alpha.shape + (3,), np.float32)
    y0 = ys.min() if y0 is None else y0
    y1 = ys.max() if y1 is None else y1
    t = clamp01((np.arange(alpha.shape[0], dtype=np.float32) - y0) / max(1, y1 - y0))[:, None, None]
    return (np.asarray(top, np.float32) * (1 - t) + np.asarray(bottom, np.float32) * t) * np.ones((1, alpha.shape[1], 1), np.float32)


def extrude(alpha, dx, dy, steps):
    acc = np.zeros_like(alpha)
    for i in range(1, steps + 1):
        M = np.float32([[1, 0, dx * i / steps], [0, 1, dy * i / steps]])
        acc = np.maximum(acc, cv2.warpAffine(alpha, M, (alpha.shape[1], alpha.shape[0])))
    return acc


def glow(img, alpha, color, sigma, strength):
    g = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return img + g[..., None] * np.asarray(color, np.float32) * strength


def radial_blur(img, cx, cy, amount, samples=8):
    acc = img.copy()
    for i in range(1, samples):
        s = 1 + amount * i / samples
        M = cv2.getRotationMatrix2D((cx, cy), 0, s)
        acc += cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return acc / samples


def dir_blur(img, dx, dy, samples=8):
    acc = np.zeros_like(img)
    for i in range(samples):
        u = i / (samples - 1) - 0.5
        M = np.float32([[1, 0, dx * u], [0, 1, dy * u]])
        acc += cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return acc / samples


def light_ribbon(shape, pts_fn, t, width, color, seed=0, n=220):
    """Volumetric light trail along a parametric path; returns additive RGB layer."""
    h, w = shape
    lay = np.zeros((h // 2, w // 2), np.float32)
    rng = np.random.default_rng(seed)
    us = np.linspace(0, 1, n)
    pts = np.array([pts_fn(u, t) for u in us]) * 0.5
    head = (t * 0.9) % 1.6 - 0.3
    for i in range(n - 1):
        u = us[i]
        # comet-like: bright head travelling along the path, long fading tail, gentle flicker
        tail = clamp01(1 - (head - u) / 0.75) * (u <= head + 0.02)
        taper = (math.sin(math.pi * u) ** 0.6) * (0.25 + 0.75 * tail) * (0.85 + 0.15 * math.sin(u * 40 + t * 9))
        wd = max(1, int(width * 0.5 * (0.35 + 0.65 * tail) * math.sin(math.pi * u) ** 0.5))
        cv2.line(lay, tuple(np.int32(pts[i] * 4)), tuple(np.int32(pts[i + 1] * 4)), float(taper), wd * 4,
                 cv2.LINE_AA, 2)
    core = cv2.GaussianBlur(lay, (0, 0), 0.9)
    soft = cv2.GaussianBlur(lay, (0, 0), 7)
    wide = cv2.GaussianBlur(lay, (0, 0), 26)
    a = core * 0.55 + soft * 0.7 + wide * 0.75
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    col = np.asarray(color, np.float32)
    hot = np.clip(a - 0.6, 0, 1)
    return a[..., None] * col + hot[..., None] * np.array([1, 1, 1], np.float32) * 0.8


def sparks(shape, t, seed, n, origin_fn, color, speed=600, life=0.9, size=3.0):
    h, w = shape
    rng = np.random.default_rng(seed)
    lay = np.zeros((h // 2, w // 2), np.float32)
    t0 = rng.random(n) * 4.0
    ang = rng.normal(-math.pi / 2, 0.9, n)
    sp = speed * (0.4 + rng.random(n))
    for i in range(n):
        age = (t - t0[i]) % 4.0
        if age > life:
            continue
        ox, oy = origin_fn(rng.random(), i)
        x = ox + math.cos(ang[i]) * sp[i] * age
        y = oy + math.sin(ang[i]) * sp[i] * age + 300 * age * age
        b = (1 - age / life) ** 1.5
        p0 = (int(x * 0.5 * 4), int(y * 0.5 * 4))
        p1 = (int((x - math.cos(ang[i]) * 18 * b) * 0.5 * 4), int((y - math.sin(ang[i]) * 18 * b) * 0.5 * 4))
        cv2.line(lay, p0, p1, float(b), max(1, int(size * 2)), cv2.LINE_AA, 2)
    a = cv2.GaussianBlur(lay, (0, 0), 0.8) + cv2.GaussianBlur(lay, (0, 0), 5) * 0.8
    a = cv2.resize(a, (w, h))
    return a[..., None] * np.asarray(color, np.float32)


def flare(shape, x, y, strength, color=(1.0, 0.78, 0.45)):
    h, w = shape
    yy, xx = np.mgrid[0:h:2, 0:w:2].astype(np.float32)
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    core = np.exp(-(d / 60) ** 2) * 1.2 + np.exp(-(d / 260) ** 2) * 0.5 + np.exp(-(d / 700) ** 2) * 0.25
    streak = np.exp(-((yy - y) / 6) ** 2) * np.exp(-(np.abs(xx - x) / 520)) * 0.7
    a = cv2.resize((core + streak).astype(np.float32), (w, h))
    out = a[..., None] * np.asarray(color, np.float32)
    # ghosts along the axis through the centre
    for k, (r, s) in enumerate([(0.35, 40), (0.62, 80), (1.25, 30), (1.6, 120)]):
        gx = x + (w / 2 - x) * 2 * r
        gy = y + (h / 2 - y) * 2 * r
        g = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (s * s))) * 0.12
        out += cv2.resize(g.astype(np.float32), (w, h))[..., None] * np.array([0.6, 0.8, 1.0], np.float32)
    return out * strength


def grade(img, lift=(0, 0, 0), gain=(1, 1, 1), gamma=(1, 1, 1), sat=1.0, con=1.0):
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    out = lum[..., None] + (img - lum[..., None]) * sat
    out = np.asarray(lift, np.float32) + clamp01(out) * (np.asarray(gain, np.float32) - np.asarray(lift, np.float32))
    out = np.power(clamp01(out), np.asarray(gamma, np.float32))
    return clamp01(0.5 + (out - 0.5) * con)


def finish(img, fi, grain=0.035, vign=0.35, chroma=0.0):
    h, w = img.shape[:2]
    if chroma > 0:
        r = cv2.warpAffine(img[..., 0], np.float32([[1, 0, chroma], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REFLECT)
        b = cv2.warpAffine(img[..., 2], np.float32([[1, 0, -chroma], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REFLECT)
        img = np.dstack([r, img[..., 1], b])
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    bl = cv2.GaussianBlur(clamp01(small - 0.7) * 2, (0, 0), 12)
    img = img + cv2.resize(bl, (w, h)) * 0.35
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = 1 - vign * clamp01((((xx / w - 0.5) ** 2) * 1.6 + ((yy / h - 0.5) ** 2) * 1.0) * 2.4) ** 1.3
    img = img * v[..., None]
    rng = np.random.default_rng(fi + 7)
    g = cv2.resize(rng.standard_normal((h // 2, w // 2)).astype(np.float32), (w, h))
    img = img + g[..., None] * grain * 0.5
    return (clamp01(img) * 255 + 0.5).astype(np.uint8)
