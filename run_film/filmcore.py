"""v2 renderer core: 60 s vertical promo, 湘雅追光乐跑团 · 2026 长沙马拉松. 1080x1920 @ 24 fps, 120 BPM.

Every person on screen is a procedurally animated, code-drawn 3D runner (runner3d). Every shot has its own scene:
each photographic plate is used exactly once, the rest are CG stages, and the build fails on any repeat.
"""
import os, sys, math, functools
import numpy as np, cv2
from kit import *
from timeline import *
from fxtext import DRAW, ease, bezier, tmask, dust, confetti, leaves
import runner3d as RN
import world as WD

OUT = f'{ROOT}/out/frames2'
PL = f'{ROOT}/assets/v2'
SHUTTER = 0.5 / FPS
MAGENTA = np.array([1.0, 0.30, 0.55], np.float32)
ORANGE = np.array([1.0, 0.58, 0.16], np.float32)
CYAN = np.array([0.45, 0.85, 1.0], np.float32)
AMBER = np.array([1.0, 0.68, 0.32], np.float32)
SEPIA = np.array([0.95, 0.74, 0.5], np.float32)

PAL = {
    'indigo': dict(g=dict(lift=(0.01, 0.015, 0.05), gain=(0.86, 0.94, 1.08), gamma=(1.04, 1.0, 0.96), sat=0.9, con=1.06),
                   acc=AMBER, acc2=np.array([0.4, 0.6, 1.0], np.float32), flash=(0.7, 0.8, 1.0)),
    'sepia': dict(g=dict(lift=(0.045, 0.025, 0.01), gain=(1.08, 0.96, 0.8), gamma=(0.98, 1.0, 1.08), sat=0.82, con=1.06),
                  acc=GOLD, acc2=XY_RED, flash=(1.0, 0.88, 0.7)),
    'dawn': dict(g=dict(lift=(0.03, 0.015, 0.02), gain=(1.08, 0.98, 0.9), gamma=(0.98, 1.0, 1.03), sat=1.1, con=1.05),
                 acc=GOLD, acc2=ORANGE, flash=(1.0, 0.92, 0.8)),
    'red': dict(g=dict(lift=(0.03, 0.0, 0.01), gain=(1.1, 0.95, 0.92), gamma=(0.98, 1.0, 1.02), sat=1.18, con=1.12),
                acc=XY_RED, acc2=GOLD, flash=(1.0, 1.0, 1.0)),
    'gold': dict(g=dict(lift=(0.04, 0.03, 0.0), gain=(1.12, 1.0, 0.82), gamma=(0.97, 1.0, 1.05), sat=1.1, con=1.06),
                 acc=GOLD, acc2=WHITE, flash=(1.0, 0.95, 0.8)),
    'white': dict(g=dict(lift=(0.05, 0.04, 0.02), gain=(1.1, 1.04, 0.92), gamma=(0.98, 1.0, 1.04), sat=1.0, con=1.04),
                  acc=GOLD, acc2=XY_RED, flash=(1.0, 1.0, 1.0)),
}


def pal_at(t):
    return PAL[section_at(t)[3]]


def lerp(a, b, u):
    return a + (b - a) * u


def tm(x):
    """Soft shoulder for the HDR CG layers."""
    k = 0.72
    return np.where(x < k, x, k + (1 - k) * (1 - np.exp(-(x - k) / (1 - k)))).astype(np.float32)


# ------------------------------------------------------------------ people
@functools.lru_cache(maxsize=1)
def renderer():
    return RN.Renderer(W, H)


class Run:
    """One code-drawn runner: body + where it starts, which way it runs, and its gait."""

    def __init__(self, body, x=0.0, z=0.0, heading=0.0, phase=0.0, freq=1.45):
        self.body, self.x, self.z, self.heading, self.phase, self.freq = body, x, z, heading, phase, freq

    def pose(self, tw):
        return RN.pose_world(self.body, tw, self.freq, (self.x, 0.0, self.z), self.heading, self.phase)

    def pelvis(self, tw):
        return self.pose(tw)[0][0]


def B(h=1.75, f=False, skin=0, kit=0, acc=0, stride=1.0, arm=1.0):
    return RN.Body(h, female=f, skin=skin, kit=kit, acc=acc, stride=stride, arm=arm)


class Layer:
    pass


def people(runs, cam_fn, tw0, slow, light, res=1.0, mb=2, mirror=False, ground=True):
    """Render runners with motion blur. cam_fn(tw) gives the camera at world time tw."""
    rd = renderer()
    w, h = int(round(W * res / 4) * 4), int(round(H * res / 4) * 4)
    acc = np.zeros((h, w, 4), np.float32)
    sh = np.zeros((h, w), np.float32)
    ao = np.zeros((h, w), np.float32)
    gf = np.zeros((h, w), np.float32)
    dep = np.full((h, w), 1e4, np.float32)
    refl = None
    for k in range(mb):
        tw = tw0 + (((k + 0.5) / mb - 0.5) * SHUTTER) * slow
        cam = cam_fn(tw)
        posed = [(r.body,) + tuple(r.pose(tw)[:2]) for r in runs]
        posed.sort(key=lambda p: -np.linalg.norm(p[1][0] - cam.pos))
        c_k = np.zeros((h, w, 4), np.float32)
        s_k, a_k, g_k, d_k = np.zeros((h, w), np.float32), np.ones((h, w), np.float32), np.ones((h, w), np.float32), \
            np.full((h, w), 1e4, np.float32)
        for b0 in range(0, max(len(posed), 1), RN.MAXR):
            batch = posed[b0:b0 + RN.MAXR]
            c, aux = rd.render(batch, cam.pos, cam.target, fov=cam.fov, basis=cam.basis, size=(w, h), ground=ground,
                               **light)
            d = rd.depth
            front = (d < d_k)[..., None]
            c_k = np.where(front, c + (1 - c[..., 3:4]) * c_k, c_k + (1 - c_k[..., 3:4]) * c)
            d_k = np.minimum(d_k, d)
            s_k = 1 - (1 - s_k) * (1 - aux[..., 0])
            a_k = a_k * aux[..., 1]
            g_k = g_k * (aux[..., 3] > 0.5)
        acc += c_k
        sh += s_k
        ao += a_k
        gf += g_k
        dep = np.minimum(dep, d_k)
        if mirror and k == 0:
            mc = cam.mirrored()
            cr, _ = rd.render(posed, mc.pos, mc.target, fov=mc.fov, basis=mc.basis, size=(w, h), ground=False, **light)
            refl = cr[::-1].copy()
    L = Layer()
    up = (lambda a: a) if (w, h) == (W, H) else (lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC))
    L.c = np.clip(up(acc / mb), 0, None)
    L.c[..., 3] = np.clip(L.c[..., 3], 0, 1)
    L.sh, L.ao, L.gf = up(sh / mb), up(ao / mb), up(gf / mb)
    L.depth = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
    L.refl = None if refl is None else np.clip(up(refl), 0, None)
    L.cam = cam_fn(tw0)
    return L


def over_people(img, L, wrap=0.35):
    a = L.c[..., 3:4]
    col = tm(L.c[..., :3] / np.maximum(a, 1e-4)) * a
    if wrap > 0:
        edge = np.clip(cv2.GaussianBlur(a[..., 0], (0, 0), 3) - a[..., 0] * 0.6, 0, 1)[..., None]
        img_b = cv2.GaussianBlur(img, (0, 0), 9)
        col = col + img_b * edge * wrap * a
    return img * (1 - a) + col


def ground_shadow(img, L, k=0.6, ao_k=0.6):
    m = np.clip(L.gf, 0, 1)
    f = (1 - k * L.sh * m) * (1 - ao_k * (1 - L.ao) * m)
    return img * f[..., None]


def floor_reflection(img, L, k=0.35, blur=6, mask=None):
    if L.refl is None:
        return img
    r = L.refl
    a = r[..., 3:4]
    col = tm(r[..., :3] / np.maximum(a, 1e-4)) * a
    if blur:
        col = cv2.GaussianBlur(col, (0, 0), blur)
    m = 1.0 if mask is None else mask[..., None]
    return img + col * k * m * (1 - L.c[..., 3:4])


# ------------------------------------------------------------------ backgrounds
def cg_bg(cam, env, L=None, res=0.75):
    w, h = int(W * res), int(H * res)
    rd = cam.rays(w, h)
    sk = env['sky']
    img = WD.sky(rd, **sk)
    hit, t, x, z = WD.ground_coords(cam, rd)
    if env.get('ground') and hit.any():
        fpx, fpz = WD.footprints(cam, rd, t, w, h)
        kind = env['ground']
        if kind == 'asphalt':
            alb, rough = WD.ground_asphalt(x, z, fpx, fpz, **env.get('gargs', {}))
        elif kind == 'track':
            alb, rough = WD.ground_track(x, z, fpx, fpz, **env.get('gargs', {}))
        elif kind == 'brick':
            alb, rough = WD.ground_brick(x, z, fpx, fpz, **env.get('gargs', {}))
        else:
            alb, rough = WD.ground_mirror(x, z, fpx, fpz, **env.get('gargs', {}))
        for dm, rect, dcol in env.get('decals', []):
            c = WD.decal(x, z, dm, rect, fpx, fpz)[..., None]
            alb = alb * (1 - c) + np.asarray(dcol, np.float32) * c
        sun = WD.norm(env['light']['sun']).astype(np.float32)
        scol = np.asarray(env['light']['sun_col'], np.float32)
        amb = np.asarray(env.get('amb', env['light']['sky']), np.float32)
        if L is not None:
            shd = cv2.resize(L.sh * np.clip(L.gf, 0, 1), (w, h))
            aoo = cv2.resize(1 - (1 - L.ao) * np.clip(L.gf, 0, 1), (w, h))
        else:
            shd, aoo = 0.0, 1.0
        shd = np.asarray(shd, np.float32)
        lit = scol * max(sun[1], 0.0) * 2.0 * env.get('sun_k', 1.0) * (1 - 0.92 * shd)[..., None] + \
            amb * (np.asarray(aoo, np.float32)[..., None] * 0.9 + 0.1)
        g = alb * lit
        if env.get('pool'):
            px_, pz_, pr, pc = env['pool']
            d2 = ((x - px_) ** 2 + (z - pz_) ** 2) / (pr * pr)
            g = g + alb * np.asarray(pc, np.float32) * np.exp(-d2)[..., None] * (1 - 0.92 * shd)[..., None]
        if env.get('spec', 0) > 0:
            rr = rd * np.array([1, -1, 1], np.float32)
            fres = 0.04 + 0.96 * (1 - np.clip(-rd[..., 1], 0, 1)) ** 5
            sref = WD.sky(rr, **sk)
            g = g + sref * (fres * env['spec'] * (1 - rough * 0.7))[..., None]
        if env.get('grid'):
            gs, gw, gc = env['grid']
            cx_ = WD.box_line(((x / gs) - np.round(x / gs)) * gs, gw, fpx)
            cz_ = WD.box_line(((z / gs) - np.round(z / gs)) * gs, gw, fpz)
            g = g + np.asarray(gc, np.float32) * np.maximum(cx_, cz_)[..., None] * np.exp(-t / 25)[..., None]
        fog = env.get('fog')
        if fog:
            fd, fc = fog
            f = (1 - np.exp(-t / fd))[..., None]
            g = g * (1 - f) + np.asarray(fc, np.float32) * f
        img = np.where(hit[..., None], g, img)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC) if res != 1 else img
    if L is not None and env.get('mirror_k'):
        gmask = np.clip(L.gf, 0, 1)
        img = floor_reflection(img, L, env['mirror_k'], env.get('mirror_blur', 4), gmask)
    return tm(np.clip(img, 0, None))


@functools.lru_cache(maxsize=2)
def plate_img(name):
    p = f'{PL}/{name}_2x.png'
    if not os.path.exists(p):
        p = f'{PL}/{name}.jpg' if os.path.exists(f'{PL}/{name}.jpg') else f'{A}/{name}_2x.png'
    img = cv2.imread(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def plate_view(name, hy, camh, z=1.0, cy=0.5, fov0=58.0, shake=(0, 0), rot=0.0):
    """Cover-warp a plate and return (image, matched camera). Zoom is about the frame centre, cy pans vertically."""
    P = plate_img(name)
    ph, pw = P.shape[:2]
    s = max(W / pw, H / ph) * z
    M = cv2.getRotationMatrix2D((pw * 0.5, ph * cy), math.degrees(rot), s)
    M[0, 2] += W / 2 - pw * 0.5 + shake[0]
    M[1, 2] += H / 2 - ph * cy + shake[1]
    img = cv2.warpAffine(P, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    hy_s = (M @ np.array([pw * 0.5, ph * hy, 1.0]))[1] / H
    th = math.tan(math.radians(fov0) / 2) / z
    fov = math.degrees(2 * math.atan(th))
    cam = WD.plate_cam(hy_s, camh, fov)
    return img, cam, M


def sun_xy(cam, sun):
    p = cam.project(cam.pos + WD.norm(sun) * 1e4, W, H)
    return None if p is None else (p[0], p[1])


# lighting presets (runner shader units)
def light(sun, sun_col, sky, gnd, rim=1.0, rim_col=None, exposure=1.0):
    return dict(sun=tuple(sun), sun_col=tuple(sun_col), sky=tuple(sky), gnd=tuple(gnd), rim=rim,
                rim_col=tuple(rim_col if rim_col is not None else sun_col), exposure=exposure)
