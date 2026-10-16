"""Style keyframe: the 'drop' hero shot - text behind subject, light ribbons, flare."""
import math, sys
import numpy as np, cv2
from kit import *


def bezier(p, u):
    p = np.asarray(p, np.float32)
    n = len(p) - 1
    return sum(math.comb(n, i) * (1 - u) ** (n - i) * u ** i * p[i] for i in range(n + 1))


def render(t=1.42):
    plate = load_plate('hero_runner')
    mask = load_mask('hero_runner', plate.shape[:2])
    z, cx, cy = 1.06, 0.52, 0.52
    bg = cover(plate, z, cx, cy)
    m = cover(mask[..., None].repeat(3, 2), z, cx, cy)[..., 0]
    m = clamp01((m - 0.15) / 0.7)
    sun = (330, 690)
    # speed: radial blur on background only, centred on the sun
    bgb = radial_blur(bg, sun[0], sun[1], 0.05, 10)
    img = grade(bgb, lift=(0.02, 0.01, 0.03), gain=(1.04, 0.96, 0.86), gamma=(0.98, 1.0, 1.06), sat=1.08, con=1.06)

    # giant title behind the runner
    tm = text_mask('追光', 'sport', 560)
    T = paste_mask(img.shape[:2], tm, 560, 520, 1.0, math.radians(-3))
    ext = extrude(T, 16, 22, 14)
    img = over(img, XY_RED * 0.85, clamp01(ext - T) * 0.95)
    for k, (dx, a) in enumerate([(-70, 0.18), (-38, 0.32)]):
        echo = cv2.warpAffine(T, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H))
        img = over(img, GOLD, echo * a)
    fill = gradient_fill(T, (1.0, 1.0, 0.98), (1.0, 0.80, 0.42))
    img = over(img, fill, T)
    img = glow(img, T, (1.0, 0.7, 0.35), 24, 0.35)

    en = text_mask('CHASE THE LIGHT', 'num', 70, tracking=16)
    E = paste_mask(img.shape[:2], en, 545, 215, 1.0, math.radians(-3))
    img = over(img, WHITE, E * 0.95)

    # light ribbons sweeping behind the runner toward the sun
    rib1 = light_ribbon((H, W), lambda u, t: bezier([(-80, 1500), (250, 1250), (120, 860), (sun[0], sun[1])], u),
                        t, 11, (1.0, 0.62, 0.22), seed=1)
    rib2 = light_ribbon((H, W), lambda u, t: bezier([(1200, 1650), (820, 1300), (980, 820), (sun[0] + 40, sun[1] - 30)], u),
                        t, 8, (1.0, 0.30, 0.25), seed=2)
    img = img + (rib1 + rib2 * 0.9) * 1.25

    # runner, sharp, with a warm rim from the backlight
    fg = grade(bg, lift=(0.0, 0.0, 0.02), gain=(1.05, 0.98, 0.9), sat=1.1, con=1.1)
    edge = clamp01(m - cv2.GaussianBlur(m, (0, 0), 6)) * 2.5
    fg = fg + edge[..., None] * np.array([1.0, 0.7, 0.35], np.float32) * 0.6
    img = img * (1 - m[..., None]) + fg * m[..., None]

    img = img + sparks((H, W), 1.3, 3, 140, lambda r, i: (300 + r * 260, 1700 + (i % 7) * 10), (1.0, 0.7, 0.3),
                       speed=520, life=0.8)
    img = img + flare((H, W), sun[0], sun[1], 0.38)

    # lower-third lockup
    band = np.zeros((H, W), np.float32)
    poly = np.array([[0, 1555], [W, 1500], [W, 1690], [0, 1745]], np.int32)
    cv2.fillPoly(band, [poly], 1.0, cv2.LINE_AA)
    img = img * (1 - band[..., None] * 0.35)
    stripe = np.zeros((H, W), np.float32)
    cv2.fillPoly(stripe, [np.array([[0, 1555], [W, 1500], [W, 1512], [0, 1567]], np.int32)], 1.0, cv2.LINE_AA)
    img = over(img, XY_RED, stripe)
    name = text_mask('湘雅追光乐跑团', 'sans', 96, tracking=6, weight=900)
    N = paste_mask(img.shape[:2], name, 540, 1600, 1.0, math.radians(-2.9))
    img = over(img, WHITE, N)
    sub = text_mask('XIANGYA  ·  CHASE THE LIGHT  RUNNING CLUB', 'num', 30, tracking=6)
    img = over(img, GOLD, paste_mask(img.shape[:2], sub, 540, 1678, 1.0, math.radians(-2.9)))
    info = text_mask('2026 长沙马拉松   11.15   42.195KM', 'sport', 52, tracking=3)
    img = over(img, WHITE, paste_mask(img.shape[:2], info, 540, 1800, 1.0, 0))

    tag = text_mask('湘雅 1906—2026', 'sans', 34, tracking=4, weight=700)
    badge = np.zeros((H, W), np.float32)
    cv2.rectangle(badge, (60, 92), (60 + tag.shape[1] + 40, 92 + tag.shape[0] + 26), 1.0, -1)
    img = over(img, XY_RED, badge * 0.92)
    img = over(img, WHITE, paste_mask(img.shape[:2], tag, 80 + tag.shape[1] / 2, 105 + tag.shape[0] / 2))
    return finish(img, 1, grain=0.03, vign=0.28, chroma=1.5)


if __name__ == '__main__':
    out = render()
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else 'keyframe/keyframe.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print('ok')
