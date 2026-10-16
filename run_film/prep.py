"""Upscale generated plates 2x (EDSR) and cut out subjects (rembg) for layered compositing."""
import os, sys, glob
import numpy as np, cv2
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))

A = f'{ROOT}/assets'


def upscale(img, sr, tile=256, pad=16):
    h, w = img.shape[:2]
    out = np.zeros((h * 2, w * 2, 3), np.float32)
    wsum = np.zeros((h * 2, w * 2, 1), np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(h, y + tile + pad), min(w, x + tile + pad)
            up = sr.upsample(img[y0:y1, x0:x1]).astype(np.float32)
            th, tw = up.shape[:2]
            wy = np.minimum(np.arange(th) + 1, th - np.arange(th)).astype(np.float32)
            wx = np.minimum(np.arange(tw) + 1, tw - np.arange(tw)).astype(np.float32)
            wgt = np.minimum(np.minimum(wy[:, None], 2 * pad), np.minimum(wx[None, :], 2 * pad))[..., None]
            out[y0 * 2:y1 * 2, x0 * 2:x1 * 2] += up * wgt
            wsum[y0 * 2:y1 * 2, x0 * 2:x1 * 2] += wgt
    return np.clip(out / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)
EDSR = f'{A}/EDSR_x2.pb'


def main(names):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model = os.environ.get('SR_MODEL', 'fsrcnn')
    sr.readModel(EDSR if model == 'edsr' else f'{A}/FSRCNN_x2.pb')
    sr.setModel(model, 2)
    from rembg import remove, new_session
    sess = new_session('isnet-general-use')
    for n in names:
        src = f'{A}/{n}.jpg'
        up_p = f'{A}/{n}_2x.png'
        if not os.path.exists(up_p):
            img = cv2.imread(src)
            img = cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 5, 15)
            up = upscale(img, sr)
            blur = cv2.GaussianBlur(up, (0, 0), 1.2)
            up = cv2.addWeighted(up, 1.3, blur, -0.3, 0)
            cv2.imwrite(up_p, up)
        m_p = f'{A}/{n}_mask.png'
        if not os.path.exists(m_p):
            m = remove(Image.open(src).convert('RGB'), session=sess, only_mask=True)
            m.save(m_p)
        print('prepared', n, flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
