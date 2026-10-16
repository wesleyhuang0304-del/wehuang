"""Overlay a projected ground grid and runner foot tracks on a plate to check camera matching.

usage: python3 tools/plate_dbg.py PLATE HY CAMH [ZOOM] [CY] -- writes out/dbg_PLATE.jpg
       runner tracks: pass x,z,heading triples via RUNS="x,z,h;x,z,h"
"""
import os, sys, math
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filmcore import plate_view, W, H, ROOT  # noqa: E402

name, hy, camh = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
zoom = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
cy = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
img, cam, _ = plate_view(name, hy, camh, zoom, cy)
out = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def seg(p, q, col, th=2):
    a, b = cam.project(p, W, H), cam.project(q, W, H)
    if a and b:
        cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, th, cv2.LINE_AA)


for x in range(-6, 7):
    seg((x, 0, 1.0), (x, 0, 80), (0, 255, 0) if x else (0, 255, 255), 1)
for z in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50):
    seg((-8, 0, z), (8, 0, z), (0, 200, 0), 1)
    a = cam.project((-5.5, 0, z), W, H)
    if a:
        cv2.putText(out, f'z{z}', (int(a[0]), int(a[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
for x in filter(None, os.environ.get('LINES', '').split(',')):
    seg((float(x), 0, 0.5), (float(x), 0, 120), (255, 255, 0), 2)
for spec in filter(None, os.environ.get('RUNS', '').split(';')):
    x, z, h = map(float, spec.split(','))
    d = np.array([math.sin(h), 0, math.cos(h)])
    p0 = np.array([x, 0, z])
    p1 = p0 + d * 3.2 * 2.0
    seg(p0, p1, (0, 0, 255), 4)
    for p, hh in ((p0, 1.75), (p1, 1.75)):
        seg(p, p + np.array([0, hh, 0]), (255, 0, 255), 3)
cv2.imwrite(f'{ROOT}/out/dbg_{name}.jpg', cv2.resize(out, (540, 960)))
print('ok', cam.pos, cam.fov)
