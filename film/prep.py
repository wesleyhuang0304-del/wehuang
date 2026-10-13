"""Upscale generated plates 2x (EDSR, tiled) and estimate depth maps for 2.5D parallax."""
import os, sys, shutil, glob
import numpy as np, cv2

SRC = '/opt/cursor/artifacts/assets'
DST = '/workspace/film/assets'
NAMES = ['plate_ch1_humans', 'plate_ch2_elves', 'plate_ch3_kingdom', 'plate_ch4_dwarves',
         'plate_ch5_isle', 'plate_ch6_desert', 'map_parchment', 'parchment_blank']


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


def main(stage):
    if stage in ('all', 'up'):
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(f'{DST}/EDSR_x2.pb')
        sr.setModel('edsr', 2)
        for n in NAMES:
            out = f'{DST}/{n}_2x.png'
            if os.path.exists(out):
                continue
            img = cv2.imread(f'{SRC}/{n}.jpg')
            img = cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 5, 15)
            up = upscale(img, sr)
            blur = cv2.GaussianBlur(up, (0, 0), 1.2)
            up = cv2.addWeighted(up, 1.35, blur, -0.35, 0)
            cv2.imwrite(out, up)
            print('up', n, up.shape, flush=True)
    if stage in ('all', 'depth'):
        import torch
        from transformers import pipeline
        torch.set_num_threads(4)
        pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device='cpu')
        from PIL import Image
        for n in NAMES[:6]:
            out = f'{DST}/{n}_depth.png'
            if os.path.exists(out):
                continue
            im = Image.open(f'{SRC}/{n}.jpg').convert('RGB')
            d = np.asarray(pipe(im)['predicted_depth'].squeeze().numpy(), np.float32)
            d = cv2.resize(d, (2560, 1440), interpolation=cv2.INTER_CUBIC)
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)
            cv2.imwrite(out, (d * 65535).astype(np.uint16))
            print('depth', n, flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
