"""Procedural CG world for the v2 promo: camera rig, sky, textured ground plane and plate camera matching.

Coordinate frame matches runner3d: y up, ground at y=0, runners run along their heading (+z by default).
"""
import math, functools
import numpy as np, cv2


def norm(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


class Cam:
    def __init__(self, pos, target, fov=40.0, roll=0.0, up=(0, 1, 0)):
        self.pos = np.asarray(pos, float)
        self.target = np.asarray(target, float)
        self.fov = float(fov)
        f = norm(self.target - self.pos)
        upv = np.asarray(up, float)
        if abs(np.dot(norm(upv), f)) > 0.999:
            upv = np.array([0.0, 0.0, 1.0])
        r = norm(np.cross(upv, f))
        u = np.cross(f, r)
        if roll:
            c, s = math.cos(roll), math.sin(roll)
            r, u = r * c + u * s, -r * s + u * c
        self.r, self.u, self.f = r, u, f
        self.tanh = math.tan(math.radians(self.fov) / 2)

    @property
    def basis(self):
        return (self.r, self.u, self.f)

    def rays(self, w, h):
        xs = (np.arange(w, dtype=np.float32) + 0.5) / w * 2 - 1
        ys = 1 - (np.arange(h, dtype=np.float32) + 0.5) / h * 2
        X, Y = np.meshgrid(xs * self.tanh * (w / h), ys * self.tanh)
        d = X[..., None] * self.r.astype(np.float32) + Y[..., None] * self.u.astype(np.float32) + self.f.astype(np.float32)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def project(self, p, w, h):
        """World point -> (px, py, depth)."""
        d = np.asarray(p, float) - self.pos
        z = d @ self.f
        if z <= 1e-4:
            return None
        x = d @ self.r / (z * self.tanh * (w / h))
        y = d @ self.u / (z * self.tanh)
        return (x + 1) / 2 * w, (1 - y) / 2 * h, z

    def mirrored(self):
        """Camera reflected through the ground plane; its render, flipped vertically, is the floor reflection."""
        M = np.array([1.0, -1.0, 1.0])
        c = Cam(self.pos * M, self.target * M, self.fov)
        return c


def plate_cam(horizon_y, height, fov=58.0, x=0.0, z=0.0, yaw=0.0):
    """Camera that matches a plate whose horizon sits at fraction horizon_y from the top of the output frame."""
    th = math.tan(math.radians(fov) / 2)
    vh = 1 - 2 * horizon_y
    pitch = math.atan(vh * th)  # positive = looking down
    f = np.array([math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
    pos = np.array([x, height, z])
    return Cam(pos, pos + f, fov)


# ------------------------------------------------------------------ textures
@functools.lru_cache(maxsize=8)
def noise_tex(seed, size=512, octaves=5):
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = 4 * 2 ** o
        g = rng.standard_normal((n, n)).astype(np.float32)
        g = np.tile(g, (3, 3))
        up = cv2.resize(g, (size * 3, size * 3), interpolation=cv2.INTER_CUBIC)[size:2 * size, size:2 * size]
        acc += up * amp
        tot += amp
        amp *= 0.55
    acc /= tot
    return (acc - acc.mean()) / (acc.std() + 1e-6)


@functools.lru_cache(maxsize=8)
def noise_pyr(seed):
    base = noise_tex(seed)
    lv = [base]
    for _ in range(6):
        lv.append(cv2.GaussianBlur(lv[-1], (0, 0), 2 ** len(lv) * 0.6))
    return lv


def sample_noise(x, z, period, fp, seed=1):
    """Tileable noise sampled in world metres with distance-aware prefiltering. fp = pixel footprint (m)."""
    lv = noise_pyr(seed)
    size = lv[0].shape[0]
    u = (x / period % 1.0) * size
    v = (z / period % 1.0) * size
    texel = period / size
    lod = np.clip(np.log2(np.maximum(fp / texel, 1.0)), 0, len(lv) - 1)
    out = np.zeros_like(x, dtype=np.float32)
    lo = np.floor(lod).astype(int)
    fr = (lod - lo).astype(np.float32)
    for k in range(len(lv)):
        sel = (lo == k) | (lo + 1 == k)
        if not sel.any():
            continue
        s = cv2.remap(lv[k], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        wgt = np.where(lo == k, 1 - fr, 0) + np.where(lo + 1 == k, fr, 0)
        out += s * wgt
    atten = 1.0 / (1.0 + (fp / (texel * 64)) ** 2)
    return out * atten


def box_line(d, half, fp):
    """Coverage of the band |d|<half under a box filter of width fp (exact)."""
    fp = np.maximum(fp, 1e-4)
    return (np.clip(d + fp / 2, -half, half) - np.clip(d - fp / 2, -half, half)) / fp


def dashes(z, period, duty, fpz):
    """Coverage of a dashed pattern along z with antialiasing; averages to the duty cycle when minified."""
    ph = (z / period) % 1.0
    edge = np.clip(fpz / period, 1e-3, 1.0)
    on = np.clip((duty - ph) / edge + 0.5, 0, 1) * np.clip(ph / edge + 0.5, 0, 1)
    blend = np.clip(fpz / period * 1.5, 0, 1)
    return on * (1 - blend) + duty * blend


def ground_asphalt(x, z, fpx, fpz, lanes=(), dash=None, tone=(0.11, 0.11, 0.115), wet=0.0, center_line=None):
    fp = np.maximum(fpx, fpz * 0.5)
    n1 = sample_noise(x, z, 3.0, fp, 1)
    n2 = sample_noise(x * 0.31, z * 0.31, 7.0, fp * 0.31, 2)
    base = np.asarray(tone, np.float32) * (1 + 0.16 * n1[..., None] + 0.12 * n2[..., None])
    paint = np.zeros_like(x, dtype=np.float32)
    for lx, half, kind in lanes:
        c = box_line(x - lx, half, fpx)
        if kind == 'dash':
            c = c * dashes(z, 6.0, 0.45, fpz)
        paint = np.maximum(paint, c)
    if center_line is not None:
        lx, half, col = center_line
        c = box_line(x - lx, half, fpx)
        base = base * (1 - c[..., None]) + np.asarray(col, np.float32) * c[..., None]
    wear = 0.85 + 0.15 * np.clip(sample_noise(x * 2, z * 2, 2.0, fp * 2, 3), -1, 1)
    base = base * (1 - paint[..., None]) + np.array([0.78, 0.78, 0.76], np.float32) * (paint * wear)[..., None]
    rough = np.full(x.shape, 1.0 - wet, np.float32)
    return base, rough


def ground_track(x, z, fpx, fpz, lane_w=1.22, x0=0.0):
    fp = np.maximum(fpx, fpz * 0.5)
    n = sample_noise(x * 3, z * 3, 2.0, fp * 3, 4)
    base = np.array([0.50, 0.11, 0.075], np.float32) * (1 + 0.1 * n[..., None])
    rel = (x - x0) / lane_w
    d = (rel - np.round(rel)) * lane_w
    c = box_line(d, 0.025, fpx)
    base = base * (1 - c[..., None]) + np.array([0.86, 0.85, 0.82], np.float32) * c[..., None]
    return base, np.full(x.shape, 0.85, np.float32)


def ground_brick(x, z, fpx, fpz, tone=(0.45, 0.2, 0.13)):
    bw, bh = 0.22, 0.11
    row = np.floor(z / bh)
    xo = x + (row % 2) * bw * 0.5
    col = np.floor(xo / bw)
    h = np.sin(row * 12.9898 + col * 78.233) * 43758.5453
    h = h - np.floor(h)
    mx = (xo / bw - col - 0.5) * bw
    mz = (z / bh - row - 0.5) * bh
    mortar = np.maximum(1 - box_line(mx, bw / 2 - 0.008, fpx), 1 - box_line(mz, bh / 2 - 0.008, fpz))
    fade = np.clip(np.maximum(fpx, fpz) / 0.05, 0, 1)
    mortar = mortar * (1 - fade) + 0.15 * fade
    var = (0.75 + 0.5 * h)[..., None] * (1 - fade[..., None]) + fade[..., None]
    base = np.asarray(tone, np.float32) * var
    base = base * (1 - mortar[..., None]) + np.array([0.3, 0.27, 0.24], np.float32) * mortar[..., None]
    n = sample_noise(x * 4, z * 4, 2.0, np.maximum(fpx, fpz) * 4, 5)
    return base * (1 + 0.08 * n[..., None]), np.full(x.shape, 0.9, np.float32)


def ground_mirror(x, z, fpx, fpz, tone=(0.015, 0.015, 0.02)):
    fp = np.maximum(fpx, fpz * 0.5)
    n = sample_noise(x, z, 4.0, fp, 6)
    base = np.asarray(tone, np.float32) * (1 + 0.3 * n[..., None])
    rough = np.clip(0.15 + 0.1 * n, 0, 1).astype(np.float32)
    return base, rough


def decal(x, z, mask, rect, fpx, fpz):
    """Project a 2D mask onto the ground rect=(x0, z0, x1, z1); mask row 0 is at z1 (far edge) so it reads upright."""
    x0, z0, x1, z1 = rect
    hgt, wid = mask.shape
    u = (x - x0) / (x1 - x0) * wid
    v = (z1 - z) / (z1 - z0) * hgt
    texel = (x1 - x0) / wid
    lod = np.maximum(fpx, fpz) / texel
    blur = cv2.GaussianBlur(mask, (0, 0), 4)
    a = cv2.remap(mask, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    b = cv2.remap(blur, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    k = np.clip((lod - 1) / 6, 0, 1)
    return a * (1 - k) + b * k


# ------------------------------------------------------------------ sky
def skyline_height(az, seed, dens=1.0, hmax=0.12):
    """Elevation (radians) of a procedural city skyline as a function of azimuth."""
    k = az * 60 * dens
    i = np.floor(k)
    h = np.sin(i * 12.9898 + seed * 7.1) * 43758.5453
    h = h - np.floor(h)
    h2 = np.sin(i * 3.17 + seed) * 9182.31
    h2 = h2 - np.floor(h2)
    tall = (h2 > 0.82) * h2 * 1.6
    return (0.15 + 0.85 * h ** 2 + tall) * hmax * 0.6


def mountain_height(az, seed, amp=0.06):
    a = az * 3.0
    return amp * (0.55 + 0.25 * np.sin(a * 2.1 + seed) + 0.15 * np.sin(a * 5.3 + seed * 2) + 0.06 * np.sin(a * 13 + seed * 3))


def sky(rd, top, hor, sun_dir=None, sun_col=(1.0, 0.8, 0.5), sun_size=0.03, glow=0.6, below=None, skyline=None,
        sky_floor=None, halo=0.9):
    """rd: HxWx3 unit ray directions. Returns HxWx3 linear RGB."""
    el = np.arcsin(np.clip(rd[..., 1], -1, 1))
    t = np.clip(el / 0.6, 0, 1) ** 0.6
    col = np.asarray(hor, np.float32) * (1 - t[..., None]) + np.asarray(top, np.float32) * t[..., None]
    if below is not None:
        b = np.clip(-el / 0.4, 0, 1)[..., None]
        col = col * (1 - b) + np.asarray(below, np.float32) * b
    if sun_dir is not None:
        sd = norm(sun_dir).astype(np.float32)
        c = np.clip(rd @ sd, -1, 1)
        ang = np.arccos(c)
        disc = np.clip((sun_size - ang) / (sun_size * 0.08) + 0.5, 0, 1)
        halo = np.exp(-ang / (sun_size * 2.2)) * halo + np.exp(-ang / 0.35) * glow
        col = col + np.asarray(sun_col, np.float32) * (halo[..., None] * 0.9) + np.array([1.0, 0.96, 0.85], np.float32) * \
            disc[..., None] * 6.0
    if skyline is not None:
        for layer in skyline:
            az = np.arctan2(rd[..., 0], rd[..., 2])
            if layer['kind'] == 'city':
                hh = skyline_height(az + layer.get('off', 0), layer.get('seed', 1), layer.get('dens', 1), layer.get('h', 0.12))
            else:
                hh = mountain_height(az + layer.get('off', 0), layer.get('seed', 1), layer.get('h', 0.06))
            hh = hh + layer.get('base', 0.0)
            pix = layer.get('aa', 0.002)
            m = np.clip((hh - el) / pix + 0.5, 0, 1) * (el > layer.get('base', 0.0) - 0.2)
            haze = np.clip(1 - (hh - el) / max(hh.max(), 1e-3), 0, 1)
            lc = np.asarray(layer['col'], np.float32)[None, None] * (1 + layer.get('haze', 0.0) * haze[..., None])
            col = col * (1 - m[..., None]) + lc * m[..., None]
    return col.astype(np.float32)


# ------------------------------------------------------------------ ground compositing
def ground_coords(cam, rd):
    o = cam.pos.astype(np.float32)
    dy = rd[..., 1]
    hit = dy < -1e-4
    t = np.where(hit, -o[1] / np.where(hit, dy, -1), 1e6).astype(np.float32)
    x = o[0] + rd[..., 0] * t
    z = o[2] + rd[..., 2] * t
    return hit, t, x, z


def footprints(cam, rd, t, w, h):
    px = 2 * cam.tanh / h
    fpx = t * px
    fpz = t * px / np.maximum(-rd[..., 1], 0.01)
    return fpx.astype(np.float32), fpz.astype(np.float32)
