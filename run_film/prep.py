"""Upscale generated plates 2x (EDSR) and cut out subjects (rembg) for layered compositing."""
import os, sys, glob
import numpy as np, cv2
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..', 'film'))
from prep import upscale  # noqa: E402

A = f'{ROOT}/assets'
EDSR = os.path.join(ROOT, '..', 'film', 'assets', 'EDSR_x2.pb')


def main(names):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(EDSR)
    sr.setModel('edsr', 2)
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
