#!/usr/bin/env python3
"""
semcl_halftone_floyd.py
Floyd–Steinberg error-diffusion dithering for SEM-CL images.

What it does:
1) Loads a TIF (any bit depth), keeps full precision internally.
2) Converts to grayscale (luminance) if needed.
3) Robust intensity normalization + local contrast (CLAHE).
4) Optional tiny edge boost (unsharp-like) before dithering.
5) Floyd–Steinberg error-diffusion with serpentine scan → binary PNG (0/255).

How to use (IDE):
- Edit INPUT_PATH and OUTPUT_PATH below and press Run.

Deps:
    pip install numpy imageio scikit-image
"""

from pathlib import Path
import numpy as np
import imageio.v3 as iio
from skimage import img_as_float32, exposure
from scipy.ndimage import gaussian_filter

# adjust to your repo layout as before
repo_root = Path(__file__).resolve().parent.parent

# =========================
# CONFIG — EDIT THESE
# =========================
INPUT_PATH  = repo_root / "data/RV/denoised_ebsd.png"   # put your input TIF here
OUTPUT_PATH = repo_root / "data/RV/binary_ebsd_halfstone_floyd.png"  # output PNG (binary 0/255)

# Optional preprocessing knobs (usually fine as-is)
CLAHE_CLIP   = 0.01   # lower if dots look too dense (e.g., 0.006)
CLAHE_NBINS  = 256
EDGE_BOOST   = 0.0    # 0..1; tiny edge boost before dithering (e.g., 0.2). 0 disables.
EDGE_SIGMA   = 1.0    # Gaussian radius used for the edge boost

# =========================
# Helpers
# =========================
def _to_gray01(arr):
    """Convert to float [0,1] grayscale (luminance if RGB)."""
    a = img_as_float32(arr)
    if a.ndim == 3:
        if a.shape[-1] == 4:
            a = a[..., :3]
        if a.shape[-1] >= 3:
            # luminance
            a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        else:
            a = np.median(a, axis=-1)
    # robust normalize with 1–99th percentiles
    p1, p99 = np.percentile(a, (1, 99))
    if p99 > p1:
        a = np.clip((a - p1) / (p99 - p1 + 1e-8), 0, 1)
    else:
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a.astype(np.float32)

def _preprocess_gray01(g):
    """Local contrast (CLAHE) + optional tiny edge boost, keep in [0,1]."""
    g = exposure.equalize_adapthist(g, clip_limit=CLAHE_CLIP, nbins=CLAHE_NBINS)
    if EDGE_BOOST and EDGE_BOOST > 0:
        blur = gaussian_filter(g, EDGE_SIGMA)
        g = np.clip(g + EDGE_BOOST * (g - blur), 0, 1)
    return g.astype(np.float32)

def floyd_steinberg_serpentine(img01):
    """
    Floyd–Steinberg error diffusion with serpentine scan.
    Input: float32 image in [0,1]; Output: uint8 {0,255}.
    """
    a = img01.copy()
    h, w = a.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        # serpentine: left→right on even rows, right→left on odd rows
        x_range = range(w) if (y % 2 == 0) else range(w - 1, -1, -1)
        direction = 1 if (y % 2 == 0) else -1

        for x in x_range:
            old = a[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = 255 if new > 0 else 0
            err = old - new

            # diffuse error to neighbors (clipping keeps values stable)
            if direction == 1:
                # left→right
                if x + 1 < w: a[y, x + 1] += err * (7 / 16)
                if y + 1 < h:
                    if x - 1 >= 0: a[y + 1, x - 1] += err * (3 / 16)
                    a[y + 1, x] += err * (5 / 16)
                    if x + 1 < w: a[y + 1, x + 1] += err * (1 / 16)
            else:
                # right→left (mirror the neighbors horizontally)
                if x - 1 >= 0: a[y, x - 1] += err * (7 / 16)
                if y + 1 < h:
                    if x + 1 < w: a[y + 1, x + 1] += err * (3 / 16)
                    a[y + 1, x] += err * (5 / 16)
                    if x - 1 >= 0: a[y + 1, x - 1] += err * (1 / 16)

        # optional row-wise clip for numerical stability
        a[y, :] = np.clip(a[y, :], 0.0, 1.0)

    return out

# =========================
# Main
# =========================
def main():
    in_path = Path(INPUT_PATH)
    out_path = Path(OUTPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    img = iio.imread(str(in_path))
    gray01 = _to_gray01(img)
    prep = _preprocess_gray01(gray01)
    bin_img = floyd_steinberg_serpentine(prep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out_path), bin_img)
    print(f"[OK] Floyd–Steinberg halftone saved → {out_path}")

if __name__ == "__main__":
    main()
