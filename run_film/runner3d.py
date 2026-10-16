"""Procedural 3D runners: anatomical SDF bodies driven by a biomechanical run cycle, ray-marched on the GPU
(moderngl / EGL). Produces premultiplied RGBA plus a ground-shadow layer for compositing over scenery."""
import math, functools
import numpy as np
import moderngl

NJ = 18          # joints per runner
MAXR = 8         # runners per draw call

# ----------------------------------------------------------------------------- run cycle
# phase 0 = right foot contact. angles in degrees: hip flexion (+ = thigh forward), knee flexion, ankle (+ = toes up)
_PH = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_HIP = np.array([30, 16, 0, -14, -8, 8, 26, 42, 46, 38])
_KNEE = np.array([20, 40, 32, 22, 58, 98, 112, 92, 55, 24])
_ANK = np.array([8, 12, 2, -28, -14, 2, 8, 12, 12, 10])


def _periodic(tab, u):
    """Periodic Catmull-Rom through 10 evenly spaced keys."""
    n = len(tab)
    x = (u % 1.0) * n
    i = int(math.floor(x))
    f = x - i
    p0, p1, p2, p3 = tab[(i - 1) % n], tab[i % n], tab[(i + 1) % n], tab[(i + 2) % n]
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Body:
    def __init__(self, height=1.75, female=False, skin=0, hair=0, kit=0, stride=1.0, arm=1.0, acc=0):
        s = height / 1.75
        self.s, self.female, self.skin, self.hair, self.kit = s, female, skin, hair, kit
        self.thigh, self.shin, self.uarm, self.farm = 0.44 * s, 0.43 * s, 0.29 * s, 0.26 * s
        self.hipw = (0.095 if female else 0.088) * s
        self.shw = (0.175 if female else 0.2) * s
        self.spine = 0.5 * s
        self.stride, self.arm, self.acc = stride, arm, acc


def _leg(body, phase, side):
    u = phase + (0.0 if side > 0 else 0.5)
    h = math.radians(_periodic(_HIP, u) * body.stride)
    k = math.radians(_periodic(_KNEE, u) * (0.85 + 0.15 * body.stride))
    a = math.radians(_periodic(_ANK, u))
    tdir = np.array([0, -math.cos(h), math.sin(h)])
    sa = h - k
    sdir = np.array([0, -math.cos(sa), math.sin(sa)])
    fa = sa + math.pi / 2 + a
    fdir = np.array([0, -math.cos(fa), math.sin(fa)])
    return tdir, sdir, fdir


def _pose_local(body, phase):
    """Joints in runner space with the pelvis at origin (y up, z forward, x right)."""
    s = body.s
    lean = math.radians(9)
    rot_p = math.radians(6) * math.sin(2 * math.pi * phase)       # pelvis yaw
    rot_c = -math.radians(9) * math.sin(2 * math.pi * phase)      # thorax counter-rotation
    P = _ry(rot_p)
    C = _ry(rot_c) @ _rx(-lean)
    J = np.zeros((NJ, 3))
    J[0] = 0
    J[1] = C @ np.array([0, body.spine, 0.0])
    J[2] = J[1] + C @ np.array([0, 0.035 * s, 0.01 * s])
    Hm = _ry(rot_c * -0.6) @ _rx(-lean * 0.4)
    J[3] = J[2] + Hm @ np.array([0, 0.17 * s, 0.03 * s])
    for side, (i_s, i_e, i_w, i_h, i_k, i_a, i_t) in ((1, (7, 8, 9, 14, 15, 16, 17)), (-1, (4, 5, 6, 10, 11, 12, 13))):
        # legs
        tdir, sdir, fdir = _leg(body, phase, side)
        hip = P @ np.array([side * body.hipw, -0.03 * s, 0])
        knee = hip + P @ (tdir * body.thigh)
        ank = knee + P @ (sdir * body.shin)
        toe = ank + P @ (fdir * 0.17 * s)
        J[i_h], J[i_k], J[i_a], J[i_t] = hip, knee, ank, toe
        # arms swing opposite to the same-side leg
        u = phase + (0.0 if side > 0 else 0.5)
        sw = -math.radians(_periodic(_HIP, u) - 14) * 0.95 * body.arm
        el = math.radians(80 + 14 * math.sin(2 * math.pi * u + 1.2))
        sh = J[1] + C @ np.array([side * body.shw, -0.035 * s, -0.01 * s])
        udir = C @ (_rx(-sw) @ _rz(side * math.radians(8)) @ np.array([0, -1.0, 0]))
        elb = sh + udir * body.uarm
        fdir_a = C @ (_rx(-sw - el) @ _rz(side * math.radians(-14)) @ np.array([0, -1.0, 0]))
        wr = elb + fdir_a * body.farm
        J[i_s], J[i_e], J[i_w] = sh, elb, wr
    frames = np.stack([P, C, Hm])
    return J, frames


def _foot_low(body, J, frames):
    """Lowest sole point (heel or toe) of each foot in pelvis space."""
    out = []
    for ia, it in ((16, 17), (12, 13)):
        a, t = J[ia], J[it]
        fd = (t - a) / (np.linalg.norm(t - a) + 1e-9)
        heel = a - fd * 0.06 * body.s + np.array([0, -0.055 * body.s, 0])
        toe = t + np.array([0, -0.025 * body.s, 0])
        out.append(min(heel[1], toe[1]))
    return out


@functools.lru_cache(maxsize=32)
def _cycle_tables(key):
    """Pelvis height and forward speed (per unit phase) that keep stance feet planted."""
    body = Body(*key)
    n = 400
    ph = np.arange(n) / n
    lows = np.array([_foot_low(body, *_pose_local(body, p)) for p in ph])  # (n,2) right, left
    stance_r = (ph % 1.0) < 0.31
    stance_l = ((ph + 0.5) % 1.0) < 0.31
    y = np.full(n, np.nan)
    y[stance_r] = -lows[stance_r, 0]
    y[stance_l] = -lows[stance_l, 1]
    # ballistic flight between stances
    idx = np.arange(n)
    known = ~np.isnan(y)
    yk = np.interp(idx, idx[known], y[known], period=n)
    flight = ~known
    bump = np.zeros(n)
    runs = []
    i = 0
    while i < n:
        if flight[i]:
            j = i
            while j < n and flight[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    for i0, i1 in runs:
        L = i1 - i0
        u = (np.arange(L) + 0.5) / L
        bump[i0:i1] = 4 * u * (1 - u) * 0.045 * body.s
    y = yk + bump
    y = np.convolve(np.r_[y[-6:], y, y[:6]], np.ones(5) / 5, 'same')[6:-6]
    # speed: stance foot backwards displacement per unit phase
    def ankz(p, ia):
        J, _ = _pose_local(body, p)
        return J[ia][2]
    dz = ankz(0.0, 16) - ankz(0.3, 16)
    dz2 = ankz(0.5, 12) - ankz(0.8, 12)
    v_per_phase = (dz + dz2) / 2 / 0.3   # metres per unit phase
    return y, v_per_phase


def pose_world(body, t, freq=1.45, origin=(0, 0, 0), heading=0.0, phase0=0.0, timescale=1.0):
    """Joint positions in world space. The runner advances along its heading so the stance foot stays planted."""
    key = (body.s * 1.75, body.female, body.skin, body.hair, body.kit, body.stride, body.arm)
    ytab, vpp = _cycle_tables(key)
    phase = (phase0 + t * freq * timescale) % 1.0
    J, frames = _pose_local(body, phase)
    y = ytab[int(phase * len(ytab)) % len(ytab)]
    dist = (phase0 + t * freq * timescale) * vpp
    R = _ry(heading)
    off = np.array(origin, float) + R @ np.array([0, y, dist])
    Jw = (R @ J.T).T + off
    Fw = np.stack([R @ f for f in frames])
    return Jw, Fw, phase


def speed(body, freq=1.45):
    key = (body.s * 1.75, body.female, body.skin, body.hair, body.kit, body.stride, body.arm)
    return _cycle_tables(key)[1] * freq


# ----------------------------------------------------------------------------- GPU renderer
VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main(){ v_uv = in_pos; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAG = open(__file__.replace('runner3d.py', 'runner3d.glsl')).read()


class Renderer:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.ctx = moderngl.create_standalone_context(backend='egl')
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        vbo = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4').tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, '2f', 'in_pos')])
        self.c0 = self.ctx.texture((w, h), 4, dtype='f4')
        self.c1 = self.ctx.texture((w, h), 4, dtype='f4')
        self.c2 = self.ctx.texture((w, h), 4, dtype='f4')
        self.fbo = self.ctx.framebuffer(color_attachments=[self.c0, self.c1, self.c2])

    def set(self, name, value):
        if name in self.prog:
            self.prog[name].value = value

    def render(self, runners, cam_pos, cam_target, fov=40.0, sun=(0.3, 0.35, -0.9), sun_col=(1.0, 0.75, 0.45),
               sky=(0.35, 0.45, 0.6), gnd=(0.25, 0.18, 0.12), rim=1.0, rim_col=(1.0, 0.7, 0.4), ground=True,
               exposure=1.0, roll=0.0, kits=None, basis=None, size=None):
        """runners: list of (Body, joints(NJ,3), frames(3,3,3))."""
        n = min(len(runners), MAXR)
        J = np.zeros((MAXR * NJ, 3), 'f4')
        F = np.zeros((MAXR * 3, 3, 3), 'f4')
        prm = np.zeros((MAXR, 4), 'f4')
        for i, (b, Jw, Fw) in enumerate(runners[:n]):
            J[i * NJ:(i + 1) * NJ] = Jw
            F[i * 3:(i + 1) * 3] = Fw
            prm[i] = (b.s, 1.0 if b.female else 0.0, b.skin + 0.1 * b.hair, b.kit + 2 * b.acc)
        self.prog['uJ'].write(J.tobytes())
        self.prog['uF'].write(np.ascontiguousarray(F.transpose(0, 2, 1)).tobytes())  # column-major mat3
        self.prog['uP'].write(prm.tobytes())
        self.set('uN', n)
        if basis is not None:
            r, u, f = (np.asarray(v, float) for v in basis)
        else:
            f = np.array(cam_target, float) - np.array(cam_pos, float)
            f /= np.linalg.norm(f)
            up = np.array([math.sin(roll), math.cos(roll), 0.0])
            r = np.cross(up, f)
            r /= np.linalg.norm(r)
            u = np.cross(f, r)
        self.set('uCamPos', tuple(map(float, cam_pos)))
        self.prog['uCamRot'].write(np.stack([r, u, f], 1).astype('f4').T.tobytes())
        self.set('uTanHalf', math.tan(math.radians(fov) / 2))
        w, h = size if size is not None else (self.w, self.h)
        self.set('uRes', (float(w), float(h)))
        sd = np.array(sun, float)
        self.set('uSun', tuple(sd / np.linalg.norm(sd)))
        self.set('uSunCol', tuple(map(float, sun_col)))
        self.set('uSky', tuple(map(float, sky)))
        self.set('uGnd', tuple(map(float, gnd)))
        self.set('uRim', float(rim))
        self.set('uRimCol', tuple(map(float, rim_col)))
        self.set('uGround', 1.0 if ground else 0.0)
        self.set('uExposure', float(exposure))
        self.fbo.use()
        self.fbo.viewport = (0, 0, w, h)
        self.fbo.clear(0, 0, 0, 0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        def rd(k):
            buf = self.fbo.read(viewport=(0, 0, w, h), components=4, dtype='f4', attachment=k)
            return np.frombuffer(buf, 'f4').reshape(h, w, 4)[::-1].copy()
        a, b = rd(0), rd(1)
        self.depth = rd(2)[..., 0]
        return a, b
