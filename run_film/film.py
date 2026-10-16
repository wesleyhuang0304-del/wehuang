"""v2 shot list + frame assembly. Usage: python3 film.py still 1.0 5.5 ... | python3 film.py <f0> <f1> [step]"""
import os, sys, math
import numpy as np, cv2
from filmcore import *

R = math.radians
PI = math.pi

# ------------------------------------------------------------------ registry
SHOTS = []


def shot(t0, scene, plate=None, whip=(1, 0)):
    def deco(fn):
        SHOTS.append(dict(t0=t0, scene=scene, plate=plate, fn=fn, whip=whip))
        return fn
    return deco


class Ctx:
    def __init__(self, t, fi, lt, dur, punch, shk):
        self.t, self.fi, self.lt, self.dur = t, fi, lt, dur
        self.u = lt / dur
        self.punch, self.shk = punch, shk


def track_cam(run, off, look=(0, 0, 0), fov=35.0, roll=0.0):
    off = off if callable(off) else np.asarray(off, float)
    look = np.asarray(look, float)

    def f(tw):
        p = run.track(tw)
        p[1] = 0.95 * run.body.s
        o = off(tw) if callable(off) else off
        return WD.Cam(p + o, p + look, fov, roll)
    return f


def static_cam(cam):
    return lambda tw: cam


def glow_dot(img, x, y, r, col, k=1.0):
    h, w = img.shape[:2]
    lay = np.zeros((h // 4, w // 4), np.float32)
    cv2.circle(lay, (int(x / 4), int(y / 4)), max(1, int(r / 4)), 1.0, -1, cv2.LINE_AA)
    a = cv2.GaussianBlur(lay, (0, 0), max(1.0, r / 6)) + cv2.GaussianBlur(lay, (0, 0), max(2.0, r / 1.5)) * 0.6
    return img + cv2.resize(a, (w, h))[..., None] * np.asarray(col, np.float32) * k


def plate_shot(c, name, hy, camh, runs, lgt, z=(1.0, 1.06), cy=0.5, fov0=58.0, sh_k=0.55, mirror=False, res=0.75,
               mb=2, ease_k='io', sun=None, flare_k=0.0, fx=(), refl_k=0.35, slow=1.0):
    zz = lerp(z[0], z[1], ease(ease_k, c.u))
    img, cam, M = plate_view(name, hy, camh, zz, cy, fov0)
    out = dict(bg=img, fx=list(fx))
    if runs:
        L = people(runs, static_cam(cam), c.lt * slow, slow, lgt, res=res, mb=mb, mirror=mirror)
        img = ground_shadow(img, L, sh_k, 0.5)
        if mirror:
            img = floor_reflection(img, L, refl_k, 5, np.clip(L.gf, 0, 1))
        out['bg'], out['L'] = img, L
    if sun is not None:
        P = plate_img(name)
        p = M @ np.array([sun[0] * P.shape[1], sun[1] * P.shape[0], 1.0])
        out['sun'], out['flare'] = (p[0], p[1]), flare_k
    return out


# ================================================================== 0-8 s  intro (indigo night)
@shot(0.0, 'cg_void_mirror_shoe_macro')
def s01(c):
    run = Run(B(1.78, kit=1), heading=0, phase=0.05)
    slow = 0.3
    lg = light((0.25, 0.4, 1.0), (0.5, 0.72, 1.3), (0.11, 0.14, 0.24), (0.04, 0.05, 0.09), rim=2.4,
               rim_col=(0.5, 0.75, 1.4))
    y0 = 0.95 * 1.78 / 1.75
    cam = track_cam(run, lambda tw: np.array([lerp(0.5, 0.25, c.u), 0.3 - y0, lerp(-4.2, -3.5, c.u)]),
                    (0, 0.85 - y0, 1.5), fov=38)
    tw = c.lt * slow
    L = people([run], cam, tw, slow, lg, res=1.0, mb=2, mirror=True)
    p = run.pelvis(tw)
    env = dict(sky=dict(top=(0, 0, 0.004), hor=(0.008, 0.012, 0.03)), ground='mirror',
               gargs=dict(tone=(0.01, 0.012, 0.02)), light=lg, spec=0.25, mirror_k=0.5, mirror_blur=3,
               pool=(p[0], p[2] + 0.6, 1.8, (0.12, 0.2, 0.4)), grid=(0.5, 0.004, (0.05, 0.08, 0.16)),
               fog=(6, (0.008, 0.012, 0.03)))
    return dict(bg=cg_bg(L.cam, env, L), L=L, fx=[('dust', CYAN * 0.5, 40)])


@shot(2.0, 'plate_night_street_approach', plate='n01_night_street')
def s02(c):
    runs = [Run(B(1.66, f=True, kit=0, acc=1), x=0.45, z=11.5, heading=PI, phase=0.3),
            Run(B(1.8, kit=1, acc=2), x=-0.75, z=16.0, heading=PI, phase=0.72)]
    lg = light((-0.2, 0.7, 1.0), (0.42, 0.6, 0.95), (0.05, 0.07, 0.12), (0.03, 0.035, 0.05), rim=1.9,
               rim_col=(0.55, 0.75, 1.1))
    return plate_shot(c, 'n01_night_street', 0.49, 1.2, runs, lg, z=(1.0, 1.08), sh_k=0.3, mirror=True,
                      refl_k=0.45)


@shot(4.0, 'cg_streetlamp_profile_track', whip=(-1, 0))
def s03(c):
    run = Run(B(1.8, kit=0, acc=0), heading=0, phase=0.4)
    tw = c.lt
    spacing, lx, lh = 9.0, -2.4, 6.0

    def lamp(tw_):
        pz = run.pelvis(tw_)[2]
        k = math.floor((pz + 3.0) / spacing)
        zl = k * spacing + 1.5
        dz = zl - pz
        inten = 4.5 * 34 / (34 + dz * dz)
        return (lx, lh - 1.0, dz), inten, zl

    sdir, inten, zl = lamp(tw)
    amber = np.array([1.0, 0.6, 0.28])
    lg = light(sdir, tuple(amber * inten), (0.03, 0.045, 0.09), (0.03, 0.025, 0.02), rim=1.2,
               rim_col=(0.45, 0.6, 1.0))
    cam = track_cam(run, (4.4, -0.1, 0.5), (0, -0.12, 0.5), fov=38)
    L = people([run], cam, tw, 1.0, lg, res=0.75, mb=2)
    env = dict(sky=dict(top=(0.0, 0.004, 0.015), hor=(0.03, 0.05, 0.1),
                        skyline=[dict(kind='city', col=(0.012, 0.016, 0.03), h=0.16, seed=3, dens=1.4, base=0.0)]),
               ground='asphalt', gargs=dict(tone=(0.05, 0.05, 0.055), lanes=[(-1.6, 0.06, 'dash')], wet=0.4),
               light=lg, sun_k=0.0, amb=(0.02, 0.03, 0.05), spec=0.2,
               pool=(lx, zl, 3.6, tuple(amber * 0.9)), fog=(28, (0.03, 0.05, 0.1)))
    bg = cg_bg(L.cam, env, L)
    out = dict(bg=bg, L=L)

    def post(img):
        for k in range(-2, 4):
            z_ = zl + k * spacing
            p = L.cam.project((lx, lh, z_), W, H)
            if p and -200 < p[0] < W + 200:
                img = glow_dot(img, p[0], max(p[1], -40), 70, amber, 1.3)
            q = L.cam.project((lx - 9, lh, z_ + 4), W, H)
            if q and -200 < q[0] < W + 200:
                img = glow_dot(img, q[0], q[1], 22, amber, 0.8)
        return img
    out['post'] = post
    return out


@shot(6.0, 'plate_dufu_pavilion_away', plate='n12_pavilion_walk')
def s04(c):
    runs = [Run(B(1.75, kit=0, acc=3), x=0.0, z=4.4, heading=0.09, phase=0.2),
            Run(B(1.64, f=True, skin=2, kit=1, acc=1), x=0.73, z=5.3, heading=0.09, phase=0.65)]
    lg = light((0.7, 0.45, 0.4), (1.1, 0.72, 0.42), (0.12, 0.15, 0.26), (0.08, 0.07, 0.07), rim=1.8)
    return plate_shot(c, 'n12_pavilion_walk', 0.46, 2.0, runs, lg, z=(1.22, 1.3), sh_k=0.45)


# ================================================================== 8-20 s  heritage (sepia)
@shot(8.0, 'plate_xiangya_redbuilding_cross', plate='s08_redbuilding')
def s05(c):
    runs = [Run(B(1.76, kit=0, acc=0), x=1.1, z=4.2, heading=-PI / 2, phase=0.1),
            Run(B(1.68, f=True, kit=1, acc=3), x=1.9, z=5.0, heading=-PI / 2, phase=0.55)]
    lg = light((1.0, 0.5, 0.4), (1.0, 0.74, 0.45), (0.3, 0.32, 0.38), (0.25, 0.18, 0.12), rim=1.2)
    return plate_shot(c, 's08_redbuilding', 0.45, 2.7, runs, lg, z=(1.12, 1.04), cy=0.55, sh_k=0.35, slow=0.2,
                      sun=(0.95, 0.2), flare_k=0.3, fx=[('dust', GOLD, 60)])


@shot(10.0, 'plate_colonnade_toward', plate='n02_colonnade', whip=(0, 1))
def s06(c):
    run = Run(B(1.74, kit=0, acc=0), x=0.15, z=9.5, heading=PI, phase=0.3)
    pz = run.pelvis(c.lt)[2]
    shaft = 0.3 + 0.7 * smoothstep(-0.2, 0.6, math.sin(2 * PI * pz / 3.1))
    lg = light((0.85, 0.5, 0.25), tuple(np.array([1.0, 0.74, 0.44]) * (0.4 + 1.0 * shaft)), (0.2, 0.15, 0.1),
               (0.25, 0.18, 0.12), rim=1.3)
    return plate_shot(c, 'n02_colonnade', 0.52, 1.0, [run], lg, z=(1.0, 1.1), sh_k=0.5, fx=[('dust', GOLD, 90)])


@shot(12.0, 'cg_ecg_heartline_silhouette', whip=(1, 0))
def s07(c):
    run = Run(B(1.76, kit=0, acc=0), heading=0, phase=0.15)
    tw = c.lt
    lg = light((-1.0, 0.3, 0.15), (1.0, 0.25, 0.2), (0.015, 0.008, 0.008), (0, 0, 0), rim=2.6,
               rim_col=(1.0, 0.4, 0.32))
    v = RN.speed(run.body, run.freq)

    def cam(tw_):
        zc = 0.8 + 0.85 * v * tw_
        return WD.Cam((6.5, 0.85, zc), (0, 0.85, zc), 34)
    L = people([run], cam, tw, 1.0, lg, res=0.75, mb=2, ground=False)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    bg = (np.array([0.05, 0.005, 0.01], np.float32) * (1 - yy) + np.array([0.012, 0.0, 0.004], np.float32) * yy) * \
        np.ones((1, W, 1), np.float32)
    cm = L.cam
    pz = run.pelvis(tw)[2]
    gy = cm.project((0, 0, pz), W, H)[1]
    zs = np.linspace(pz - 12, pz + 6, 900)
    ys = []
    for z_ in zs:
        beat = 0.5 * v
        d = (z_ % beat) / beat
        amp = 0.0
        if z_ < pz:
            amp = (0.06 * math.exp(-((d - 0.18) / 0.04) ** 2) - 0.05 * math.exp(-((d - 0.33) / 0.012) ** 2) +
                   0.55 * math.exp(-((d - 0.36) / 0.014) ** 2) - 0.12 * math.exp(-((d - 0.4) / 0.014) ** 2) +
                   0.1 * math.exp(-((d - 0.6) / 0.05) ** 2))
        ys.append(amp)
    pts = [cm.project((0, a, z_), W, H) for z_, a in zip(zs, ys)]
    pts = np.array([[p[0], p[1]] for p in pts if p], np.float32)
    lay = np.zeros((H // 2, W // 2), np.float32)
    for i in range(len(pts) - 1):
        xm = pts[i][0]
        px_r = cm.project((0, 0, pz), W, H)[0]
        b = 1.0 if xm <= px_r else 0.25
        b *= clamp01((xm + 50) / 300)
        cv2.line(lay, tuple(np.int32(pts[i] * 2)), tuple(np.int32(pts[i + 1] * 2)), float(b), 4, cv2.LINE_AA, 2)
    a = cv2.GaussianBlur(lay, (0, 0), 0.8) + cv2.GaussianBlur(lay, (0, 0), 6) * 0.9 + cv2.GaussianBlur(lay, (0, 0), 24)
    a = cv2.resize(a, (W, H))
    bg = bg + a[..., None] * np.array([1.0, 0.22, 0.2], np.float32)
    bg = bg + np.clip(a - 1.0, 0, 1)[..., None] * 0.6
    return dict(bg=bg, L=L)


@shot(14.0, 'plate_campus_avenue_away', plate='n03_campus_avenue', whip=(-1, 0))
def s08(c):
    runs = [Run(B(1.77, kit=1, acc=2), x=-0.35, z=2.3, heading=0.02, phase=0.4),
            Run(B(1.62, f=True, skin=1, kit=0, acc=0), x=0.4, z=3.1, heading=0.02, phase=0.9)]
    lg = light((0.0, 0.3, 1.0), (1.05, 0.88, 0.56), (0.25, 0.28, 0.2), (0.18, 0.16, 0.1), rim=2.1)
    return plate_shot(c, 'n03_campus_avenue', 0.6, 0.6, runs, lg, z=(1.0, 1.07), sh_k=0.6,
                      sun=(0.5, 0.5), flare_k=0.2, fx=[('dust', GOLD, 50)])


@shot(16.0, 'cg_brick_path_leg_tracking_slowmo', whip=(0, -1))
def s09(c):
    run = Run(B(1.75, kit=1, acc=0), heading=0, phase=0.0)
    slow = 0.42
    tw = c.lt * slow
    lg = light((-0.4, 0.5, -0.65), (1.0, 0.72, 0.42), (0.35, 0.28, 0.2), (0.22, 0.12, 0.08), rim=1.6)
    cam = track_cam(run, (2.7, 0.3 - 0.95, lerp(0.5, -0.3, c.u)), (0, 0.5 - 0.95, 0.1), fov=33)
    L = people([run], cam, tw, slow, lg, res=1.0, mb=2)
    env = dict(sky=dict(top=(0.5, 0.36, 0.24), hor=(0.88, 0.66, 0.44), sun_dir=(-0.4, 0.12, -0.65),
                        sun_col=(1.0, 0.7, 0.4), sun_size=0.05, glow=0.4,
                        skyline=[dict(kind='mountain', col=(0.62, 0.42, 0.3), h=0.05, seed=2, haze=0.25),
                                 dict(kind='city', col=(0.45, 0.25, 0.16), h=0.05, seed=5, dens=0.7)]),
               ground='brick', light=lg, fog=(10, (0.85, 0.63, 0.43)))
    return dict(bg=cg_bg(L.cam, env, L), L=L, fx=[('dust', SEPIA, 70)])


@shot(18.0, 'cg_gold_studio_120_dolly', whip=(1, 0))
def s10(c):
    run = Run(B(1.78, kit=0, acc=1), heading=0, phase=0.25)
    tw = c.lt
    lg = light((0.0, 0.35, -1.0), (1.4, 1.02, 0.55), (0.4, 0.3, 0.15), (0.2, 0.13, 0.05), rim=2.0)
    cam = track_cam(run, lambda tw_: np.array([0.6, -0.45, lerp(7.6, 6.4, c.u)]), (0, 0.05, 0), fov=30)
    L = people([run], cam, tw, 1.0, lg, res=0.75, mb=2, mirror=True)
    env = dict(sky=dict(top=(0.12, 0.06, 0.02), hor=(0.95, 0.62, 0.25), sun_dir=(0.0, 0.1, -1.0),
                        sun_col=(1.0, 0.7, 0.35), sun_size=0.04, glow=0.9),
               ground='mirror', gargs=dict(tone=(0.07, 0.045, 0.02)), light=lg, spec=0.22, mirror_k=0.4,
               mirror_blur=5, fog=(30, (0.9, 0.58, 0.24)))
    s = sun_xy(L.cam, (0.0, 0.1, -1.0))
    return dict(bg=cg_bg(L.cam, env, L), L=L, sun=s, flare=0.35, fx=[('sparks', GOLD, 80)])


# ================================================================== 20-30 s  city (dawn)
@shot(20.0, 'plate_orange_isle_establish_birds', plate='s03_sunrise_isle')
def s11(c):
    zz = lerp(1.3, 1.04, ease('expo', c.u))
    img, cam, M = plate_view('s03_sunrise_isle', 0.45, 20.0, zz, 0.46)
    P = plate_img('s03_sunrise_isle')
    sp = M @ np.array([0.54 * P.shape[1], 0.41 * P.shape[0], 1.0])

    def post(im):
        rng = np.random.default_rng(11)
        lay = im
        for i in range(26):
            x0, y0 = rng.random() * 500 + 80, rng.random() * 380 + 330
            vx = 160 + rng.random() * 60
            ph = rng.random() * 6
            x = x0 + vx * c.lt + 10 * math.sin(c.lt * 2 + ph)
            y = y0 - 25 * c.lt + 8 * math.sin(c.lt * 3 + ph)
            s_ = 7 + rng.random() * 6
            fl = math.sin(c.lt * 14 + ph)
            pts = np.array([[x - s_, y - fl * s_ * 0.6], [x, y], [x + s_, y - fl * s_ * 0.6]], np.int32)
            cv2.polylines(lay, [pts], False, (0.05, 0.03, 0.03), 2, cv2.LINE_AA)
        return lay
    return dict(bg=img, sun=(sp[0], sp[1]), flare=0.45, post=post)


@shot(22.0, 'plate_xiang_river_road_pack', plate='n04_river_road', whip=(-1, 0))
def s12(c):
    # the road runs 0.2 rad left of the plate's optical axis; lane x0 at the camera plane, x = x0 - 0.2 z
    hd = PI - 0.2
    runs = [Run(B(1.78, kit=0, acc=1), x=-1.80, z=10.5, heading=hd, phase=0.1),
            Run(B(1.65, f=True, kit=1, acc=0), x=-0.92, z=11.6, heading=hd, phase=0.45),
            Run(B(1.82, skin=1, kit=0, acc=2), x=-3.16, z=12.8, heading=hd, phase=0.8),
            Run(B(1.7, kit=1, acc=3), x=-1.96, z=13.8, heading=hd, phase=0.3)]
    lg = light((0.15, 0.12, 1.0), (1.3, 0.9, 0.52), (0.35, 0.3, 0.28), (0.25, 0.18, 0.12), rim=2.3)
    return plate_shot(c, 'n04_river_road', 0.43, 2.4, runs, lg, z=(1.0, 1.06), sh_k=0.6,
                      sun=(0.62, 0.41), flare_k=0.35)


@shot(24.0, 'plate_yuelu_maple_climb', plate='n05_yuelu_path', whip=(0, 1))
def s13(c):
    runs = [Run(B(1.7, f=True, kit=0, acc=3), x=0.1, z=9.0, heading=PI, phase=0.2, freq=1.4)]
    lg = light((0.2, 0.4, 1.0), (1.1, 0.72, 0.42), (0.3, 0.2, 0.15), (0.3, 0.12, 0.08), rim=2.0)
    return plate_shot(c, 'n05_yuelu_path', 0.55, 0.5, runs, lg, z=(1.06, 1.0), sh_k=0.5, fx=[('leaves', None, 60)])


@shot(26.0, 'plate_juzizhou_bridge_away', plate='n07_bridge_deck', whip=(1, 0))
def s14(c):
    runs = [Run(B(1.76, kit=1, acc=1), x=-0.6, z=2.2, heading=0, phase=0.0),
            Run(B(1.66, f=True, kit=0, acc=0), x=0.35, z=2.9, heading=0, phase=0.5),
            Run(B(1.8, skin=1, kit=0, acc=3), x=1.1, z=3.8, heading=0, phase=0.25)]
    lg = light((-0.3, 0.15, 1.0), (1.0, 0.72, 0.62), (0.45, 0.42, 0.5), (0.3, 0.28, 0.3), rim=1.8)
    return plate_shot(c, 'n07_bridge_deck', 0.52, 0.5, runs, lg, z=(1.0, 1.05), sh_k=0.5)


@shot(28.0, 'gfx_changsha_route_map', whip=(0, -1))
def s15(c):
    return dict(bg=route_map(c), fx=[])


def route_map(c):
    w, h = W, H
    img = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img[:] = np.array([0.02, 0.03, 0.06], np.float32)
    grid = ((xx % 60) < 1.5) | ((yy % 60) < 1.5)
    img[grid] += 0.03
    river = [(560, -50), (600, 400), (520, 800), (560, 1200), (640, 1600), (600, 1990)]
    rpts = np.array([bezier(river, u) for u in np.linspace(0, 1, 200)], np.int32)
    lay = np.zeros((h, w), np.float32)
    cv2.polylines(lay, [rpts], False, 1.0, 120, cv2.LINE_AA)
    img = over(img, np.array([0.06, 0.16, 0.3], np.float32), cv2.GaussianBlur(lay, (0, 0), 3))
    isle = np.array([[548, 560], [575, 700], [580, 900], [562, 1120], [548, 1180], [540, 1100], [536, 860],
                     [538, 650]], np.int32)
    il = np.zeros((h, w), np.float32)
    cv2.fillPoly(il, [isle], 1.0, cv2.LINE_AA)
    img = over(img, np.array([0.35, 0.55, 0.25], np.float32), il * 0.85)
    route = [(860, 1500), (820, 1180), (700, 1000), (560, 900), (400, 760), (300, 560), (420, 420), (700, 380),
             (840, 600), (880, 900), (760, 1350), (560, 1500)]
    u = ease('io', clamp01(c.lt / (c.dur * 0.9)))
    n = 260
    pts = np.array([bezier(route[:6], v) for v in np.linspace(0, 1, n // 2)] +
                   [bezier(route[5:], v) for v in np.linspace(0, 1, n // 2)], np.float32)
    k = max(2, int(n * u))
    rl = np.zeros((h, w), np.float32)
    cv2.polylines(rl, [pts[:k].astype(np.int32)], False, 1.0, 10, cv2.LINE_AA)
    img = over(img, XY_RED, rl)
    img = img + cv2.GaussianBlur(rl, (0, 0), 14)[..., None] * XY_RED * 0.8
    hx, hy_ = pts[k - 1]
    img = glow_dot(img, hx, hy_, 26, GOLD, 1.6)
    marks = [((860, 1500), '湘雅 · 起点'), ((560, 900), '橘子洲'), ((300, 560), '岳麓山'), ((700, 380), '杜甫江阁'),
             ((560, 1500), '贺龙体育场 · 终点')]
    for (mx, my), name in marks:
        idx = int(np.argmin(np.sum((pts - np.array([mx, my], np.float32)) ** 2, 1)))
        if idx > k + 2:
            continue
        img = glow_dot(img, mx, my, 16, WHITE, 1.0)
        m = tmask(name, 'sans', 40, 2, 700)
        img = over(img, WHITE, paste_mask((h, w), m, mx + (m.shape[1] / 2 + 30) * (1 if mx < 700 else -1), my))
    tilt = 1 - 0.18 * (1 - ease('out', c.u))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = (1 - tilt) * w * 0.5
    dst = np.float32([[d, 120 * (1 - tilt) / 0.18], [w - d, 120 * (1 - tilt) / 0.18], [w, h], [0, h]])
    Mp = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, Mp, (w, h), borderValue=(0.02, 0.03, 0.06))


# ================================================================== 30-44 s  drop (red)
@shot(30.0, 'cg_red_strobe_studio_sprint')
def s16(c):
    run = Run(B(1.8, kit=0, acc=0, stride=1.15), heading=0, phase=0.1, freq=1.6)
    tw = c.lt
    on = math.exp(-((c.t % BEAT) / 0.12))
    lg = light((0.4, 0.5, 1.0), (1.6 * on + 0.2, 0.08, 0.06), (0.03, 0.005, 0.005), (0.02, 0, 0), rim=2.4,
               rim_col=(1.0, 0.9, 0.85))
    cam = track_cam(run, (-4.6, -0.1, 0.4), (0, -0.1, 0.4), fov=36)
    L = people([run], cam, tw, 1.0, lg, res=0.75, mb=2, mirror=True)
    env = dict(sky=dict(top=(0, 0, 0), hor=(0.08 * on + 0.02, 0.0, 0.005)), ground='mirror',
               gargs=dict(tone=(0.012, 0.005, 0.005)), light=lg, spec=0.2, mirror_k=0.5, mirror_blur=3,
               grid=(1.5, 0.012, (0.5 * on + 0.1, 0.02, 0.02)))
    return dict(bg=cg_bg(L.cam, env, L), L=L)


@shot(31.0, 'plate_elevated_expressway_trio', plate='n06_overpass', whip=(-1, 0))
def s17(c):
    runs = [Run(B(1.78, kit=0, acc=2), x=-0.5, z=6.5, heading=PI, phase=0.2, freq=1.6),
            Run(B(1.66, f=True, kit=1, acc=0), x=0.7, z=7.4, heading=PI, phase=0.6, freq=1.6),
            Run(B(1.84, skin=1, kit=0, acc=0), x=0.0, z=9.0, heading=PI, phase=0.9, freq=1.6)]
    lg = light((0.0, 0.12, 1.0), (1.3, 0.75, 0.55), (0.4, 0.3, 0.35), (0.25, 0.18, 0.15), rim=2.2)
    return plate_shot(c, 'n06_overpass', 0.48, 1.2, runs, lg, z=(1.05, 1.15), sh_k=0.6, ease_k='expo',
                      sun=(0.5, 0.48), flare_k=0.3)


@shot(32.0, 'cg_red_track_shoe_tracking', whip=(0, 1))
def s18(c):
    run = Run(B(1.74, kit=1, acc=0), heading=0, phase=0.3, freq=1.6)
    slow = 0.6
    tw = c.lt * slow
    lg = light((0.5, 0.45, -0.7), (1.3, 0.85, 0.55), (0.35, 0.3, 0.3), (0.3, 0.1, 0.06), rim=1.5)
    cam = track_cam(run, (2.0, 0.16 - 0.95 * 1.74 / 1.75, 0.5), (0, 0.32 - 0.95 * 1.74 / 1.75, 0.0), fov=36)
    L = people([run], cam, tw, slow, lg, res=1.0, mb=2)
    env = dict(sky=dict(top=(0.4, 0.3, 0.32), hor=(0.95, 0.6, 0.4),
                        skyline=[dict(kind='city', col=(0.55, 0.32, 0.28), h=0.06, seed=8, dens=0.6)]),
               ground='track', gargs=dict(x0=0.61), light=lg, fog=(12, (0.9, 0.55, 0.4)))
    return dict(bg=cg_bg(L.cam, env, L), L=L)


@shot(33.0, 'plate_race_boulevard_pack_into_sun', plate='n09_boulevard', whip=(1, 0))
def s19(c):
    rng = np.random.default_rng(19)
    runs = []
    for k in range(6):
        runs.append(Run(B(1.62 + 0.22 * rng.random(), f=k % 3 == 1, skin=k % 3, kit=k % 2, acc=k % 4),
                        x=-1.4 + 0.55 * k + rng.normal(0, 0.12), z=2.6 + (k * 1.3) % 3.2, heading=0.03,
                        phase=rng.random(), freq=1.55 + 0.1 * rng.random()))
    lg = light((0.1, 0.12, 1.0), (1.5, 0.95, 0.5), (0.2, 0.17, 0.17), (0.25, 0.13, 0.08), rim=2.8,
               rim_col=(1.0, 0.7, 0.4))
    return plate_shot(c, 'n09_boulevard', 0.5, 0.5, runs, lg, z=(1.12, 1.0), sh_k=0.6, ease_k='expo',
                      sun=(0.55, 0.43), flare_k=0.4)


@shot(34.0, 'cg_overhead_asphalt_pack', whip=(0, -1))
def s20(c):
    rng = np.random.default_rng(20)
    runs = [Run(B(1.65 + 0.2 * rng.random(), f=k % 2 == 1, skin=k % 3, kit=k % 2, acc=(1, 0, 3, 0, 1, 2, 0)[k]),
                x=-1.8 + 0.62 * k + rng.normal(0, 0.12), z=-0.6 + 1.25 * ((k * 3) % 5) + rng.normal(0, 0.25),
                heading=rng.normal(0, 0.03), phase=rng.random(), freq=1.58 + 0.06 * rng.random()) for k in range(7)]
    tw = c.lt
    v = RN.speed(runs[0].body, 1.6)

    def cam(tw_):
        zc = v * tw_ + 2.4
        return WD.Cam((0.2, 7.5, zc), (0.2, 0.0, zc + 0.001), 52, up=(0, 0, 1))
    lg = light((-1.0, 0.7, 0.35), (1.4, 0.8, 0.5), (0.3, 0.25, 0.3), (0.2, 0.1, 0.08), rim=0.8)
    L = people(runs, cam, tw, 1.0, lg, res=0.75, mb=2)
    dm = tmask('42.195', 'num', 260, 10)
    env = dict(sky=dict(top=(0.3, 0.3, 0.4), hor=(0.6, 0.5, 0.5)), ground='asphalt',
               gargs=dict(tone=(0.1, 0.095, 0.1), lanes=[(-2.4, 0.07, 'solid'), (2.8, 0.07, 'solid')],
                          center_line=(0.2, 0.09, (0.15, 0.35, 0.95))),
               decals=[(dm, (-1.1, 4.4, 1.5, 5.4), (0.9, 0.88, 0.85))], light=lg)
    return dict(bg=cg_bg(L.cam, env, L), L=L)


@shot(35.0, 'plate_tunnel_silhouettes_emerging', plate='n08_tunnel', whip=(1, 0))
def s21(c):
    runs = [Run(B(1.78, kit=0, acc=0), x=-0.55, z=8.6, heading=PI, phase=0.1, freq=1.6),
            Run(B(1.66, f=True, kit=1, acc=0), x=0.6, z=9.8, heading=PI, phase=0.6, freq=1.6)]
    lg = light((0.0, 0.08, 1.0), (1.6, 1.2, 0.7), (0.06, 0.08, 0.12), (0.05, 0.05, 0.07), rim=2.8,
               rim_col=(1.0, 0.8, 0.5))
    return plate_shot(c, 'n08_tunnel', 0.49, 1.5, runs, lg, z=(1.0, 1.1), sh_k=0.6, ease_k='expo',
                      sun=(0.5, 0.49), flare_k=0.5)


@shot(36.0, 'cg_telephoto_arm_swing_closeup', whip=(-1, 0))
def s22(c):
    run = Run(B(1.8, skin=1, kit=0, acc=2), heading=0, phase=0.2, freq=1.6)
    tw = c.lt
    lg = light((0.4, 0.55, 1.0), (1.3, 0.8, 0.5), (0.3, 0.15, 0.12), (0.25, 0.08, 0.05), rim=2.2,
               rim_col=(1.0, 0.5, 0.35))
    cam = track_cam(run, (4.6, 0.0, lerp(-0.3, 0.3, c.u)), (0, 0.02, 0), fov=15)
    L = people([run], cam, tw, 1.0, lg, res=1.0, mb=3)
    env = dict(sky=dict(top=(0.35, 0.03, 0.05), hor=(1.0, 0.4, 0.15), sun_dir=(-1.0, 0.08, 0.2),
                        sun_col=(1.0, 0.5, 0.2), sun_size=0.02, glow=0.6), ground='asphalt',
               gargs=dict(tone=(0.12, 0.08, 0.07)), light=lg, fog=(8, (0.95, 0.4, 0.18)))
    return dict(bg=cg_bg(L.cam, env, L), L=L)


@shot(37.0, 'cg_magenta_skyline_female_silhouette', whip=(1, 0))
def s23(c):
    run = Run(B(1.66, f=True, kit=0, acc=0), heading=0, phase=0.5, freq=1.6)
    tw = c.lt
    lg = light((-1.0, 0.12, 0.1), (1.5, 0.45, 0.6), (0.06, 0.02, 0.06), (0.02, 0.0, 0.02), rim=3.0,
               rim_col=(1.0, 0.45, 0.65))
    cam = track_cam(run, (7.0, -0.55, 0.0), (0, 0.15, 0), fov=24)
    L = people([run], cam, tw, 1.0, lg, res=0.75, mb=2)
    env = dict(sky=dict(top=(0.12, 0.02, 0.15), hor=(1.0, 0.35, 0.5), sun_dir=(-1.0, 0.06, 0.25),
                        sun_col=(1.0, 0.4, 0.5), sun_size=0.03, glow=0.8,
                        skyline=[dict(kind='city', col=(0.25, 0.05, 0.15), h=0.1, seed=4, dens=1.2, haze=0.5),
                                 dict(kind='city', col=(0.05, 0.0, 0.03), h=0.07, seed=9, dens=0.8, off=0.3)]),
               ground='asphalt', gargs=dict(tone=(0.03, 0.02, 0.025)), light=lg, sun_k=0.3,
               amb=(0.05, 0.02, 0.05), spec=0.3)
    s = sun_xy(L.cam, (-1.0, 0.06, 0.25))
    return dict(bg=cg_bg(L.cam, env, L), L=L, sun=s, flare=0.3)


@shot(38.0, 'cg_ground_level_runner_passes_lens', whip=(0, 1))
def s24(c):
    run = Run(B(1.78, kit=1, acc=1, stride=1.1), x=0.0, z=-6.2, heading=0, phase=0.0, freq=1.6)
    cam = WD.Cam((0.42, 0.07, 0.0), (0.0, 1.1, -3.5), 74)
    lg = light((0.1, 0.35, -1.0), (1.5, 0.9, 0.45), (0.45, 0.35, 0.3), (0.3, 0.15, 0.1), rim=2.2)
    L = people([run], static_cam(cam), c.lt, 1.0, lg, res=0.75, mb=3)
    env = dict(sky=dict(top=(0.2, 0.25, 0.45), hor=(1.0, 0.62, 0.3), sun_dir=(0.1, 0.35, -1.0),
                        sun_col=(1.0, 0.7, 0.35), sun_size=0.03, glow=0.8),
               ground='asphalt', gargs=dict(tone=(0.12, 0.1, 0.1), lanes=[(-1.2, 0.06, 'dash')]), light=lg,
               fog=(60, (1.0, 0.62, 0.32)))
    s = sun_xy(cam, (0.1, 0.35, -1.0))
    return dict(bg=cg_bg(cam, env, L), L=L, sun=s, flare=0.4)


@shot(39.0, 'cg_telephoto_pack_heat_haze', whip=(-1, 0))
def s25(c):
    rng = np.random.default_rng(25)
    runs = [Run(B(1.62 + 0.22 * rng.random(), f=k % 3 == 2, skin=k % 3, kit=(k // 2) % 2, acc=k % 4),
                x=-2.2 + 0.62 * k + rng.normal(0, 0.1), z=-(k * 2.3) % 7.0 + rng.normal(0, 0.4), heading=0,
                phase=rng.random(), freq=1.5 + 0.15 * rng.random()) for k in range(8)]
    v = RN.speed(runs[0].body, 1.55)

    def cam(tw_):
        zc = v * tw_ + 48.0
        return WD.Cam((0.2, 1.0, zc), (0.0, 1.0, zc - 48.0), 7.0)
    lg = light((0.1, 0.18, -1.0), (1.4, 0.75, 0.42), (0.4, 0.28, 0.25), (0.3, 0.15, 0.1), rim=2.4)
    L = people(runs, cam, c.lt, 1.0, lg, res=0.75, mb=2)
    env = dict(sky=dict(top=(0.7, 0.35, 0.25), hor=(1.0, 0.6, 0.32), sun_dir=(0.05, 0.03, -1.0),
                        sun_col=(1.0, 0.6, 0.3), sun_size=0.012, glow=0.6,
                        skyline=[dict(kind='city', col=(0.75, 0.42, 0.3), h=0.03, seed=12, dens=4)]),
               ground='asphalt', gargs=dict(tone=(0.16, 0.12, 0.11), lanes=[(-3.0, 0.08, 'solid'),
                                                                           (3.0, 0.08, 'solid')]),
               light=lg, fog=(110, (1.0, 0.6, 0.35)))
    bg = cg_bg(L.cam, env, L)

    def post(img):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        k = smoothstep(700, 1500, yy) * 3.0
        dx = np.sin(yy * 0.09 + c.t * 22) * k
        dy = np.sin(xx * 0.05 + c.t * 17) * k * 0.6
        return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return dict(bg=bg, L=L, post=post)


@shot(40.0, 'cg_slowmo_female_frontal_backlit', whip=(0, -1))
def s26(c):
    run = Run(B(1.67, f=True, skin=2, kit=1, acc=2), heading=0, phase=0.15, freq=1.5)
    slow = 0.6
    tw = c.lt * slow
    lg = light((0.15, 0.3, -1.0), (1.5, 1.0, 0.55), (0.42, 0.34, 0.3), (0.3, 0.18, 0.1), rim=2.6)
    cam = track_cam(run, lambda tw_: np.array([lerp(0.7, 0.25, c.u), 0.25, lerp(4.6, 4.0, c.u)]), (0, 0.12, 0), fov=32)
    L = people([run], cam, tw, slow, lg, res=1.0, mb=2)
    env = dict(sky=dict(top=(0.28, 0.3, 0.45), hor=(1.0, 0.68, 0.38), sun_dir=(0.15, 0.22, -1.0),
                        sun_col=(1.0, 0.72, 0.4), sun_size=0.03, glow=1.0,
                        skyline=[dict(kind='mountain', col=(0.55, 0.38, 0.32), h=0.07, seed=1, haze=0.3)]),
               ground='asphalt', gargs=dict(tone=(0.13, 0.11, 0.1), lanes=[(-1.4, 0.06, 'dash')]), light=lg,
               fog=(25, (1.0, 0.68, 0.4)))
    s = sun_xy(L.cam, (0.15, 0.22, -1.0))
    return dict(bg=cg_bg(L.cam, env, L), L=L, sun=s, flare=0.4, fx=[('sparks', GOLD, 60)])


@shot(42.0, 'cg_orbit_grid_stage_bullet_time', whip=(1, 0))
def s27(c):
    run = Run(B(1.8, kit=0, acc=1), heading=0, phase=0.35)
    tw = c.lt * 0.8
    lg = light((0.35, 1.0, 0.25), (1.4, 1.05, 0.7), (0.05, 0.1, 0.12), (0.01, 0.03, 0.04), rim=2.6,
               rim_col=(0.35, 0.85, 1.0))

    def off(tw_):
        a = R(lerp(35, 150, ease('io', c.u)))
        return np.array([3.8 * math.sin(a), 0.25, 3.8 * math.cos(a)])
    cam = track_cam(run, off, (0, 0.12, 0), fov=40)
    L = people([run], cam, tw, 0.8, lg, res=0.75, mb=2, mirror=True)
    p = run.pelvis(tw)
    env = dict(sky=dict(top=(0.0, 0.03, 0.05), hor=(0.04, 0.16, 0.2)), ground='mirror',
               gargs=dict(tone=(0.012, 0.03, 0.035)), light=lg, spec=0.25, mirror_k=0.6, mirror_blur=5,
               pool=(p[0], p[2], 1.8, (0.5, 0.36, 0.18)), fog=(12, (0.02, 0.1, 0.13)))
    return dict(bg=cg_bg(L.cam, env, L), L=L, fx=[('sparks', GOLD, 70)])


# ================================================================== 44-54 s  chase the light (gold)
@shot(44.0, 'cg_giant_sun_disc_silhouette_cross')
def s28(c):
    run = Run(B(1.78, kit=0, acc=0), x=0, z=-1.75, heading=0, phase=0.2)
    sd = (1.0, 0.024, 0.0)
    cam = WD.Cam((-32.0, 0.9, 0.0), (0.0, 1.05, 0.0), 9.0)
    lg = light(sd, (1.6, 0.9, 0.45), (0.12, 0.07, 0.05), (0.05, 0.03, 0.02), rim=2.8, rim_col=(1.0, 0.65, 0.3))
    L = people([run], static_cam(cam), c.lt * 0.3, 0.3, lg, res=0.75, mb=2, ground=False)
    env = dict(sky=dict(top=(0.3, 0.1, 0.05), hor=(0.8, 0.33, 0.12), sun_dir=sd, sun_col=(1.0, 0.55, 0.22),
                        sun_size=0.027, glow=0.08, halo=0.3,
                        skyline=[dict(kind='mountain', col=(0.32, 0.12, 0.06), h=0.012, seed=6, base=0.0)]),
               ground='asphalt', gargs=dict(tone=(0.05, 0.03, 0.02)), light=lg, sun_k=0.1, amb=(0.15, 0.07, 0.03),
               fog=(400, (1.0, 0.5, 0.2)))
    s = sun_xy(cam, sd)
    return dict(bg=cg_bg(cam, env, L), L=L, sun=s, flare=0.22)


@shot(46.0, 'plate_helong_stadium_approach', plate='n10_stadium_ext', whip=(-1, 0))
def s29(c):
    runs = [Run(B(1.76, kit=0, acc=1), x=-0.6, z=2.6, heading=0.12, phase=0.1),
            Run(B(1.64, f=True, kit=1, acc=0), x=0.3, z=3.2, heading=0.12, phase=0.55),
            Run(B(1.82, skin=1, kit=1, acc=2), x=1.0, z=4.4, heading=0.12, phase=0.3),
            Run(B(1.7, f=True, skin=2, kit=0, acc=3), x=-1.3, z=4.8, heading=0.12, phase=0.85)]
    lg = light((-0.45, 0.2, 1.0), (1.4, 0.95, 0.55), (0.35, 0.35, 0.45), (0.25, 0.2, 0.15), rim=2.2)
    return plate_shot(c, 'n10_stadium_ext', 0.49, 0.6, runs, lg, z=(1.08, 1.0), sh_k=0.6,
                      sun=(0.3, 0.43), flare_k=0.35)


@shot(48.0, 'plate_stadium_track_final_straight', plate='n11_stadium_track', whip=(1, 0))
def s30(c):
    runs = [Run(B(1.78, kit=1, acc=1, stride=1.1), x=0.0, z=10.5, heading=PI, phase=0.3, freq=1.6)]
    lg = light((0.0, 0.3, 1.0), (1.4, 0.95, 0.55), (0.35, 0.3, 0.3), (0.35, 0.12, 0.08), rim=2.4)
    return plate_shot(c, 'n11_stadium_track', 0.322, 2.1, runs, lg, z=(1.08, 1.25), sh_k=0.6, res=1.0, slow=0.6,
                      sun=(0.6, 0.31), flare_k=0.4)


@shot(50.0, 'cg_finish_tape_break_confetti', whip=(0, 1))
def s31(c):
    run = Run(B(1.78, kit=0, acc=1), x=0.0, z=-5.0, heading=0, phase=0.0)
    tz = -1.2
    cam = WD.Cam((1.0, 0.95, 5.2), (0.0, 1.15, -2.0), 40)
    lg = light((0.0, 0.3, -1.0), (1.5, 1.05, 0.6), (0.45, 0.38, 0.35), (0.35, 0.15, 0.1), rim=2.2)
    L = people([run], static_cam(cam), c.lt, 1.0, lg, res=0.75, mb=2)
    fin = tmask('FINISH', 'num', 220, 30)
    env = dict(sky=dict(top=(0.3, 0.32, 0.48), hor=(1.0, 0.7, 0.4), sun_dir=(0.0, 0.18, -1.0),
                        sun_col=(1.0, 0.72, 0.4), sun_size=0.03, glow=0.9,
                        skyline=[dict(kind='city', col=(0.6, 0.38, 0.3), h=0.05, seed=14, dens=1.3, haze=0.3)]),
               ground='track', light=lg, decals=[(np.ones((8, 200), np.float32), (-3.0, tz - 0.1, 3.0, tz + 0.1),
                                                  (0.95, 0.95, 0.92)),
                                                 (fin[::-1, ::-1].copy(), (-1.8, tz + 0.3, 1.8, tz + 1.4),
                                                  (0.95, 0.95, 0.92))],
               fog=(45, (1.0, 0.7, 0.42)))
    bg = cg_bg(cam, env, L)
    chest = run.pose(c.lt)[0][1]
    s = sun_xy(cam, (0.0, 0.18, -1.0))

    def post(img):
        # gantry
        for x_ in (-3.2, 3.2):
            a = cam.project((x_, 0, tz), W, H)
            b = cam.project((x_, 2.9, tz), W, H)
            if a and b:
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0.6, 0.04, 0.06), int(1400 / a[2] * 0.12),
                         cv2.LINE_AA)
        tl, tr = cam.project((3.4, 2.9, tz), W, H), cam.project((-3.4, 2.9, tz), W, H)
        bl, br = cam.project((3.4, 2.35, tz), W, H), cam.project((-3.4, 2.35, tz), W, H)
        quad = np.float32([[tl[0], tl[1]], [tr[0], tr[1]], [br[0], br[1]], [bl[0], bl[1]]])
        cv2.fillConvexPoly(img, quad.astype(np.int32), (0.62, 0.04, 0.07), cv2.LINE_AA)
        m = tmask('终点 FINISH', 'sport', 160, 6)
        hh, ww = m.shape
        P = cv2.getPerspectiveTransform(np.float32([[-ww * 0.9, -hh * 0.2], [ww * 1.9, -hh * 0.2],
                                                    [ww * 1.9, hh * 1.2], [-ww * 0.9, hh * 1.2]]), quad)
        T = cv2.warpPerspective(m, P, (W, H))
        img = over(img, WHITE, T)
        # tape: straight until the chest reaches it, then a V that snaps and flutters away
        dt = c.lt - (tz - (-5.0) - 0.15) / RN.speed(run.body, run.freq)
        ys = 1.2
        if dt < 0:
            pts3 = [(x_, ys, tz) for x_ in np.linspace(-3.2, 3.2, 30)]
            segs = [pts3]
        else:
            push = min(dt, 0.12) * 4.0
            if dt < 0.12:
                pts3 = [(x_, ys, tz + push * max(0, 1 - abs(x_ - chest[0]) / 3.2)) for x_ in np.linspace(-3.2, 3.2, 30)]
                segs = [pts3]
            else:
                e = dt - 0.12
                segs = []
                for side in (-1, 1):
                    seg = []
                    for k in range(16):
                        f = k / 15
                        x_ = side * 3.2 * (1 - f) + side * 0.15 * f
                        sway = math.sin(e * 9 + f * 4) * 0.25 * f
                        seg.append((x_ * (1 - 0.35 * f * min(e * 2, 1)), ys - 0.9 * f * min(e * 1.5, 1) ** 2 + sway * 0.3,
                                    tz + 0.5 * f + sway))
                    segs.append(seg)
        lay = np.zeros((H, W), np.float32)
        dl = np.full((H, W), 1e4, np.float32)
        for seg in segs:
            pr = [cam.project(p, W, H) for p in seg]
            for a, b in zip(pr[:-1], pr[1:]):
                if a and b:
                    th = max(2, int(0.06 * 1400 / a[2]))
                    cv2.line(lay, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1.0, th, cv2.LINE_AA)
                    cv2.line(dl, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), float(a[2]), th + 2)
        vis = lay * (dl < L.depth + 0.05)
        img = over(img, np.array([0.85, 0.05, 0.08], np.float32), vis)
        return img
    fx = [('confetti_burst', None, 220)] if c.lt > 1.05 else []
    return dict(bg=bg, L=L, sun=s, flare=0.35, post=post, post_over=True, fx=fx, conf_t0=50.0 + 1.05)


@shot(52.0, 'cg_finish_line_feet_high_angle_slowmo', whip=(-1, 0))
def s32(c):
    run = Run(B(1.74, kit=1, acc=0), x=0.0, z=-1.85, heading=0, phase=0.6)
    slow = 0.3
    cam = WD.Cam((0.35, 3.3, 3.4), (0.0, 0.25, -0.2), 36)
    lg = light((-0.5, 0.45, -0.8), (1.5, 1.05, 0.6), (0.45, 0.4, 0.38), (0.35, 0.15, 0.1), rim=1.4)
    L = people([run], static_cam(cam), c.lt * slow, slow, lg, res=1.0, mb=2)
    fin = tmask('FINISH', 'num', 220, 30)
    num_ = tmask('42.195KM', 'num', 140, 12)
    env = dict(sky=dict(top=(0.4, 0.4, 0.5), hor=(1.0, 0.75, 0.5)), ground='track', light=lg,
               decals=[(np.ones((8, 200), np.float32), (-3.0, -0.07, 3.0, 0.07), (0.95, 0.95, 0.92)),
                       (fin[::-1, ::-1].copy(), (-1.5, 0.3, 1.5, 1.25), (0.95, 0.95, 0.92)),
                       (num_[::-1, ::-1].copy(), (-1.2, -1.15, 1.2, -0.5), (0.95, 0.85, 0.5))])
    return dict(bg=cg_bg(cam, env, L, res=1.0), L=L, fx=[('confetti', None, 150)])


# ================================================================== 54-60 s  end card
@shot(54.0, 'cg_endcard_road_into_sunrise_crane')
def s33(c):
    runs = [Run(B(1.76, kit=0, acc=1), x=-0.55, z=1.5, heading=0, phase=0.1, freq=1.4),
            Run(B(1.65, f=True, kit=1, acc=0), x=0.25, z=2.1, heading=0, phase=0.55, freq=1.4),
            Run(B(1.82, skin=1, kit=0, acc=3), x=0.95, z=1.2, heading=0, phase=0.3, freq=1.4)]
    u = ease('io', c.u)
    cam = WD.Cam((0.2, lerp(0.6, 3.2, u), lerp(-1.8, -3.0, u)), (0.2, lerp(1.0, 0.7, u), 20.0), 46)
    sd = (0.0, 0.06, 1.0)
    lg = light(sd, (1.5, 1.0, 0.55), (0.45, 0.38, 0.32), (0.3, 0.18, 0.1), rim=2.6)
    L = people(runs, static_cam(cam), c.lt, 1.0, lg, res=0.75, mb=2)
    env = dict(sky=dict(top=(0.25, 0.3, 0.5), hor=(1.0, 0.72, 0.4), sun_dir=sd, sun_col=(1.0, 0.75, 0.42),
                        sun_size=0.035, glow=1.1,
                        skyline=[dict(kind='mountain', col=(0.62, 0.42, 0.32), h=0.035, seed=3, haze=0.3, off=0.4),
                                 dict(kind='city', col=(0.5, 0.32, 0.26), h=0.035, seed=21, dens=2.0, haze=0.3)]),
               ground='asphalt', gargs=dict(tone=(0.13, 0.11, 0.1), lanes=[(-2.2, 0.07, 'solid'), (2.6, 0.07, 'solid'),
                                                                          (0.2, 0.06, 'dash')]),
               light=lg, fog=(70, (1.0, 0.72, 0.42)))
    s = sun_xy(cam, sd)
    return dict(bg=cg_bg(cam, env, L), L=L, sun=s, flare=0.45, fx=[('dust', GOLD, 60)], shade=True)


SHOTS.sort(key=lambda s: s['t0'])
assert [s['t0'] for s in SHOTS] == CUTS, 'shot list out of sync with timeline.CUTS'
_scenes = [s['scene'] for s in SHOTS]
assert len(set(_scenes)) == len(_scenes), 'duplicate scene in shot list'
_plates = [s['plate'] for s in SHOTS if s['plate']]
assert len(set(_plates)) == len(_plates), 'a plate is used more than once'


def shot_at(t):
    i = max(k for k, s in enumerate(SHOTS) if s['t0'] <= t)
    t1 = SHOTS[i + 1]['t0'] if i + 1 < len(SHOTS) else DUR
    return i, SHOTS[i], t1


# ------------------------------------------------------------------ typography
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
en(0.4, 1.95, 'CHANGSHA · 05:30 AM', 540, 360, 44, color=CYAN)
line(2.3, 3.95, '长沙 · 清晨五点半', 540, 330, 66, weight=700)
line(4.3, 5.95, '这座城 还在沉睡', 540, 330, 76, weight=700)
line(6.2, 7.95, '有人 已经出发', 540, 330, 80, weight=900, color=AMBER)
en(6.6, 7.95, 'DU FU RIVER PAVILION', 540, 440, 34, color=GOLD, track=(2, 6))
# heritage
num(8.05, 9.95, 540, 420, 300, count=(1906, 2026, 8.05, 9.6, '{:d}'), stroke=GOLD)
en(8.3, 9.95, 'SINCE 1906 · XIANGYA', 540, 620, 44, color=GOLD)
for k, (t0_, word, x_) in enumerate([(10.1, '公勇勤慎', 900), (12.1, '诚爱谦廉', 180), (14.1, '求真求确', 900),
                                     (16.1, '必邃必专', 180)]):
    vert(t0_, t0_ + 1.85, word, x_, 300, 120, color=GOLD if k % 2 == 0 else WHITE)
en(12.3, 13.95, 'HEART OF XIANGYA', 540, 1640, 40, color=(1.0, 0.45, 0.4))
slam(18.0, 19.95, '120', 540, 640, 560, font='num', rot=-3, behind=True)
line(18.4, 19.95, '湘雅 · 一百二十年', 540, 1620, 72, bar=XY_RED, track=6)
en(18.7, 19.95, '1906 — 2026', 540, 1740, 46, color=GOLD)
# city
vert(20.3, 21.95, '橘子洲', 150, 300, 110, color=WHITE)
en(20.6, 21.95, 'ORANGE ISLE', 540, 1700, 50, color=GOLD)
line(22.2, 23.95, '湘江北去', 540, 300, 96, font='serif', weight=900)
en(22.5, 23.95, 'XIANG RIVER', 540, 410, 40, color=GOLD)
line(24.2, 25.95, '岳麓山', 540, 300, 110, font='serif', weight=900, color=WHITE)
en(24.5, 25.95, 'YUELU MOUNTAIN', 540, 420, 40, color=GOLD)
line(26.2, 27.95, '橘子洲大桥', 540, 300, 96, bar=XY_RED)
en(26.5, 27.95, '10KM · JUZIZHOU BRIDGE', 540, 420, 38, color=GOLD)
num(28.1, 29.74, 540, 230, 180, text='11.15', stroke=XY_RED)
en(28.4, 29.74, '2026 CHANGSHA MARATHON', 540, 360, 38, color=GOLD)
num(28.6, 29.74, 540, 1720, 150, count=(0.0, 42.195, 28.6, 29.6, '{:.3f}'), stroke=GOLD)
# drop
for t0_, word, fnt, sz, x_, y_, rot_, beh in [
        (30.0, '42.195', 'num', 300, 540, 380, -3, False), (31.0, '长沙', 'sport', 340, 540, 330, 3, False),
        (32.0, '马拉松', 'sport', 320, 540, 380, -4, False), (33.0, '11.15', 'num', 300, 540, 330, 3, False),
        (34.0, '2026', 'num', 340, 540, 400, -4, False), (35.0, 'GO!', 'num', 400, 540, 400, 4, False),
        (36.0, '每一步', 'sport', 280, 540, 380, -3, False), (37.0, '都算数', 'sport', 280, 540, 380, 3, False),
        (38.0, '超越', 'sport', 400, 540, 420, -4, False), (39.0, '自我', 'sport', 400, 540, 420, 4, False)]:
    slam(t0_, t0_ + 0.96, word, x_, y_, sz, font=fnt, rot=rot_, behind=beh)
line(40.3, 41.95, '湘雅追光乐跑团', 540, 300, 104, bar=XY_RED, track=8)
en(40.6, 41.95, 'XIANGYA · CHASE THE LIGHT RUNNING CLUB', 540, 420, 32, color=GOLD, track=(3, 8))
slam(42.0, 43.74, '全力以赴', 540, 255, 195, rot=-3)
# chase
slam(44.0, 45.95, '追光', 540, 1580, 360, rot=-3)
en(44.3, 45.95, 'CHASE THE LIGHT', 545, 1800, 58, track=(2, 22))
line(46.2, 47.95, '贺龙体育场', 540, 300, 96, bar=XY_RED)
en(46.5, 47.95, 'HELONG STADIUM · FINISH', 540, 420, 38, color=GOLD)
num(48.0, 49.95, 540, 1600, 230, count=(40.0, 42.195, 48.0, 49.6, '{:.3f}'), stroke=ORANGE)
en(48.3, 49.95, 'KM  TO  THE  LIGHT', 545, 1770, 46, color=GOLD, track=(6, 14))
line(50.35, 51.97, '致敬每一位', 540, 1540, 88, weight=900)
slam(50.75, 51.97, '追光者', 540, 1735, 230, rot=-3)
# end card
vert(54.6, 60.0, '我们追光', 130, 260, 96, color=WHITE)
vert(55.0, 60.0, '也成为光', 950, 260, 96, color=GOLD)
slam(54.0, 60.0, '湘雅追光乐跑团', 540, 1330, 128, font='sans', weight=900, rot=-2, ext=XY_RED)
en(54.3, 60.0, 'XIANGYA · CHASE THE LIGHT RUNNING CLUB', 540, 1465, 34, color=GOLD, track=(3, 8), rot=-2)
line(54.6, 60.0, '2026 长沙马拉松 · 11.15 · 42.195KM', 540, 1565, 50, font='sport', weight=None, bar=XY_RED, track=2)
line(55.4, 60.0, '公勇勤慎  诚爱谦廉  求真求确  必邃必专', 540, 1660, 38, font='serif', weight=700, color=GOLD, track=3)
TX.append(dict(k='badge', t0=56.0, t1=60.0, text='湘雅 1906—2026', behind=False))


# ------------------------------------------------------------------ rhythm
def beat_env(t, period, decay, phase=0.0):
    return math.exp(-((t - phase) % period) / decay)


def rhythm(t):
    sec = section_at(t)[0]
    punch, shake = 0.0, 0.0
    if sec == 'intro':
        for hb in heartbeats():
            if 0 <= t - hb < 0.4:
                punch += 0.012 * math.exp(-(t - hb) / 0.08)
    elif sec == 'heritage':
        punch = 0.012 * beat_env(t, 2 * BEAT, 0.1)
        shake = 2 * beat_env(t, BEAT, 0.1)
    elif sec == 'city':
        punch = 0.016 * beat_env(t, BEAT, 0.1)
        shake = 3 * beat_env(t, BEAT, 0.1)
        if t >= 28.0:
            punch += 0.02 * beat_env(t, BEAT / 2, 0.05) * (t - 28.0) / 1.75
            shake += 6 * (t - 28.0) / 1.75
    elif sec == 'drop':
        punch = 0.035 * beat_env(t, BEAT, 0.09)
        shake = 11 * beat_env(t, BEAT, 0.1)
    elif sec == 'chase':
        punch = 0.03 * beat_env(t, 2 * BEAT, 0.12)
        shake = 4 * beat_env(t, BEAT, 0.1)
    for tm_ in IMPACTS:
        if 0 <= t - tm_ < 0.6:
            shake += 40 * math.exp(-(t - tm_) / 0.16)
            punch += 0.08 * math.exp(-(t - tm_) / 0.12)
    return punch, shake


def shake_vec(t, amp):
    if amp < 0.3:
        return (0.0, 0.0)
    rng = np.random.default_rng(int(t * FPS) * 13 + 5)
    return tuple(rng.normal(0, amp, 2))


def bug(img, t):
    m = tmask('湘雅追光乐跑团', 'sans', 30, 2, 800)
    m2 = tmask('2026 长马 · 11.15', 'sport', 30, 1)
    x0, y0 = 54, 130
    w1, w2 = m.shape[1] + 30, m2.shape[1] + 26
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


# ------------------------------------------------------------------ frame
def xform(img, punch, sh):
    if punch < 1e-4 and abs(sh[0]) + abs(sh[1]) < 0.3:
        return img
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + punch)
    M[0, 2] += sh[0]
    M[1, 2] += sh[1]
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def render_frame(fi):
    t = fi / FPS
    i, S, t1 = shot_at(t)
    pal = pal_at(t)
    lt = t - S['t0']
    punch, shk = rhythm(t)
    c = Ctx(t, fi, lt, t1 - S['t0'], punch, shk)
    res = S['fn'](c)
    img = res['bg']
    behind = [x for x in TX if x['t0'] <= t < x['t1'] and x.get('behind') and res.get('L') is not None]
    front = [x for x in TX if x['t0'] <= t < x['t1'] and not (x.get('behind') and res.get('L') is not None)]
    for x in behind:
        img = DRAW[x['k']](img, x, t, pal, None)
    if res.get('L') is not None:
        img = over_people(img, res['L'])
    if res.get('post'):
        img = res['post'](img)
    if res.get('shade'):
        yy = np.arange(H, dtype=np.float32)
        img = img * (1 - 0.45 * smoothstep(1150, 1600, yy))[:, None, None]
    img = grade(np.clip(img, 0, 1), **pal['g'])
    img = xform(img, punch, shake_vec(t, shk))
    if lt < 0.13 and i > 0:
        w_ = 1 - lt / 0.13
        img = dir_blur(img, S['whip'][0] * 160 * w_, S['whip'][1] * 160 * w_, 9)
    for x in front:
        img = DRAW[x['k']](img, x, t, pal, None)
    for kind, col, n in res.get('fx', []):
        if kind == 'dust':
            img = img + dust(t, i, n, col)
        elif kind == 'sparks':
            img = img + sparks((H, W), lt + 1.0, i, n, lambda r, j: (r * W, H + 20 - (j % 9) * 8), col, speed=560,
                               life=0.85)
        elif kind == 'confetti':
            img = confetti(img, t, i, n, S['t0'] - 0.4)
        elif kind == 'confetti_burst':
            img = confetti(img, t, i, n, res.get('conf_t0', S['t0']) - 0.35)
        elif kind == 'leaves':
            img = leaves(img, t, i, n)
    if res.get('sun') and res.get('flare', 0) > 0:
        sx, sy = res['sun']
        if -400 < sx < W + 400 and -400 < sy < H + 400:
            pulse = 1 + 0.35 * punch / 0.035
            fl_lay = flare((H, W), sx, sy, res['flare'] * min(pulse, 2.0), tuple(0.6 + 0.4 * np.asarray(pal['acc2'])))
            if res.get('L') is not None:
                fl_lay = fl_lay * (1 - 0.8 * res['L'].c[..., 3:4])
            img = img + fl_lay
    if 30.0 <= t < 54.0 and not any(a <= t < b for a, b in GAPS):
        img = bug(img, t)
    fl = 0.0
    fcol = np.asarray(pal['flash'], np.float32)
    for tm_ in IMPACTS:
        if 0 <= t - tm_ < 0.5:
            fl = max(fl, 0.95 * math.exp(-(t - tm_) / 0.09))
    sec = section_at(t)[0]
    if sec in ('drop', 'chase') and lt < 1.0 / FPS + 1e-6 and S['t0'] not in IMPACTS:
        fl = max(fl, 0.35)
        fcol = np.asarray(pal['acc'] if (i % 2) else pal['flash'], np.float32)
    if fl > 0:
        img = img * (1 - fl * 0.5) + fcol * fl
    chroma = 1.4 + 9 * (fl if fl > 0.2 else 0) + shk * 0.08
    for a, b in GAPS:
        if a <= t < b:
            g = (t - a) / (b - a)
            img = img * 0.04
            yy = np.arange(H, dtype=np.float32)[:, None]
            xx = np.arange(W, dtype=np.float32)[None, :]
            streak = np.exp(-((yy - 960) / (3 + 10 * g)) ** 2) * np.exp(-np.abs(xx - 540) / (80 + 900 * g))
            img = img + streak[..., None] * np.asarray(pal['flash'], np.float32) * (0.6 + 1.2 * g)
    if t > 59.3:
        img = img * (1 - ease('io', (t - 59.3) / 0.7))
    if t < 0.4:
        img = img * ease('io', t / 0.4)
    return finish(img, fi, grain=0.03, vign=0.3, chroma=chroma)


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
        os.makedirs(f'{ROOT}/out/stills2', exist_ok=True)
        for ts in sys.argv[2:]:
            fi = int(round(float(ts) * FPS))
            out = render_frame(fi)
            cv2.imwrite(f'{ROOT}/out/stills2/s_{float(ts):05.2f}.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            print('still', ts, flush=True)
    else:
        main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
