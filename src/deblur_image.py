import numpy as np
import imageio.v3 as iio
from skimage import img_as_float32, img_as_ubyte
from skimage.filters import unsharp_mask
from skimage.restoration import richardson_lucy, wiener, denoise_bilateral, denoise_tv_chambolle
from scipy.ndimage import gaussian_filter
from pathlib import Path

# =========================
# PARAMS – EDIT HERE
# =========================
PARAMS = {
    # method: "unsharp", "rl", "wiener", "rl_tv", or "wiener_rl"
    "method": "rl",

    # --- PSF (used by rl & wiener) ---
    # If psf_size is None, we'll auto-size it to ~6*sigma+1 (odd).
    "psf_size": 15,         # e.g., 15, 31, 51 ... or None for auto
    "psf_sigma": 5,         # blur radius in pixels; try 6–12 before 30+

    # --- RL (Richardson–Lucy) ---
    "rl_iters": 10,           # 10–20 w/ TV is usually enough
    "rl_post_gauss": 0.3,     # gentle Gaussian to tame halos; set 0 to disable
    "pre_denoise": True,      # light bilateral before RL
    "pre_sigma_color": 0.03,  # <= 0.04 keeps faint small grains
    "pre_sigma_spatial": 2,

    # --- TV denoise (for rl_tv / wiener_rl) ---
    # "auto" chooses weight from noise estimate; or give a float like 0.12
    "tv_weight": "auto",

    # --- Wiener (standalone or pre-pass) ---
    "wiener_balance": 0.01,   # larger = smoother, smaller = sharper/noisier

    # --- Output bit depth ---
    # 16 (recommended for SEM) or 8
    "save_bitdepth": 16,

    # --- I/O ---
    "input_path": None,
    "output_path": None
}

# (Optional) Simple CLI overrides: python sem_cl_deblur_easy.py rl 8 16
# method, psf_sigma, rl_iters
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2: PARAMS["method"] = sys.argv[1]
    if len(sys.argv) >= 3: PARAMS["psf_sigma"] = float(sys.argv[2])
    if len(sys.argv) >= 4: PARAMS["rl_iters"] = int(sys.argv[3])

# --------------------- helpers --------------------- #

def gaussian_psf(size: int, sigma: float):
    """Centered Gaussian PSF (normalized)."""
    size = int(size) | 1  # force odd
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()
    return psf

def gaussian_psf_auto(sigma: float):
    """Choose PSF size based on sigma (~6*sigma coverage), ensure odd."""
    size = int(np.ceil(6.0 * float(sigma)))
    size = size | 1
    size = max(size, 7)  # keep it reasonable
    return gaussian_psf(size, sigma)

def _estimate_noise_sigma01(img01: np.ndarray) -> float:
    """Robust noise estimate in [0,1] via high-pass MAD."""
    hp = img01 - gaussian_filter(img01, 1.0)
    return 1.4826 * np.median(np.abs(hp))

def _rl_call(image: np.ndarray, psf: np.ndarray, iters: int):
    """Safe RL call across scikit-image versions (num_iter vs iterations)."""
    try:
        return richardson_lucy(image, psf, num_iter=iters, clip=False)
    except TypeError:
        return richardson_lucy(image, psf, iterations=iters, clip=False)

def _maybe_bilateral(x, is_gray, enable, s_color, s_spatial):
    if not enable:
        return x
    return denoise_bilateral(
        x,
        sigma_color=s_color,
        sigma_spatial=s_spatial,
        channel_axis=None if is_gray else -1
    )

def _apply_tv_auto(x01: np.ndarray, tv_weight):
    if isinstance(tv_weight, str) and tv_weight.lower() == "auto":
        sigma_n = _estimate_noise_sigma01(x01)
        # map noise to a gentle TV weight
        w = float(np.clip(0.08 + 0.8 * sigma_n, 0.06, 0.25))
    else:
        w = float(tv_weight)
    return denoise_tv_chambolle(x01, weight=w)

def _save_image(out_path: Path, img01: np.ndarray, bitdepth: int):
    img01 = np.clip(img01, 0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if int(bitdepth) == 16:
        out_u16 = (img01 * 65535.0 + 0.5).astype(np.uint16)
        iio.imwrite(str(out_path), out_u16)
    else:
        out_u8 = img_as_ubyte(img01)
        iio.imwrite(str(out_path), out_u8)

# --------------------- main pipeline --------------------- #

def run():
    p = PARAMS

    # --- paths ---
    inp = Path(p["input_path"])
    outp = Path(p["output_path"])
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    img = iio.imread(str(inp))  # pass str(...) for Windows/imageio compatibility
    is_gray = (img.ndim == 2) or (img.ndim == 3 and img.shape[-1] == 1)
    if not is_gray and img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # drop alpha
    img_f = img_as_float32(img)

    method = p["method"].lower()

    # --- choose PSF ---
    if p["psf_size"] is None:
        psf = gaussian_psf_auto(p["psf_sigma"])
    else:
        psf = gaussian_psf(p["psf_size"], p["psf_sigma"])

    if method == "unsharp":
        if is_gray:
            out = unsharp_mask(img_f, radius=p["unsharp_radius"], amount=p["unsharp_amount"], preserve_range=True)
        else:
            out = np.stack([
                unsharp_mask(img_f[..., c], radius=p["unsharp_radius"], amount=p["unsharp_amount"], preserve_range=True)
                for c in range(img_f.shape[-1])
            ], axis=-1)
        out = np.clip(out, 0, 1)

    elif method == "rl":
        work = _maybe_bilateral(img_f, is_gray, p["pre_denoise"], p["pre_sigma_color"], p["pre_sigma_spatial"])
        if is_gray:
            out = _rl_call(work, psf, p["rl_iters"])
        else:
            out = np.stack([_rl_call(work[..., c], psf, p["rl_iters"]) for c in range(work.shape[-1])], axis=-1)
        if p["rl_post_gauss"] and p["rl_post_gauss"] > 0:
            out = gaussian_filter(out, sigma=p["rl_post_gauss"])
        out = np.clip(out, 0, 1)

    elif method == "wiener":
        if is_gray:
            out = wiener(img_f, psf, balance=p["wiener_balance"], clip=False)
        else:
            out = np.stack([wiener(img_f[..., c], psf, balance=p["wiener_balance"], clip=False)
                            for c in range(img_f.shape[-1])], axis=-1)
        out = np.clip(out, 0, 1)

    elif method == "rl_tv":
        # Bilateral (light) -> short RL -> optional tiny Gaussian -> TV (auto)
        work = _maybe_bilateral(img_f, is_gray, p["pre_denoise"], p["pre_sigma_color"], p["pre_sigma_spatial"])
        short_iters = int(max(8, min(p["rl_iters"], 20)))  # keep it moderate
        if is_gray:
            out = _rl_call(work, psf, short_iters)
        else:
            out = np.stack([_rl_call(work[..., c], psf, short_iters) for c in range(work.shape[-1])], axis=-1)
        if p["rl_post_gauss"] and p["rl_post_gauss"] > 0:
            out = gaussian_filter(out, sigma=p["rl_post_gauss"])
        out = np.clip(out, 0, 1)
        out = _apply_tv_auto(out, p["tv_weight"])  # noise-aware cleanup
        out = np.clip(out, 0, 1)

    elif method == "wiener_rl":
        # Wiener pre-denoise -> short RL -> TV
        if is_gray:
            base = wiener(img_f, psf, balance=p["wiener_balance"], clip=False)
        else:
            base = np.stack([wiener(img_f[..., c], psf, balance=p["wiener_balance"], clip=False)
                             for c in range(img_f.shape[-1])], axis=-1)
        base = np.clip(base, 0, 1)
        short_iters = int(max(6, min(p["rl_iters"], 16)))
        if is_gray:
            out = _rl_call(base, psf, short_iters)
        else:
            out = np.stack([_rl_call(base[..., c], psf, short_iters) for c in range(base.shape[-1])], axis=-1)
        if p["rl_post_gauss"] and p["rl_post_gauss"] > 0:
            out = gaussian_filter(out, sigma=p["rl_post_gauss"])
        out = np.clip(out, 0, 1)
        out = _apply_tv_auto(out, p["tv_weight"])  # final tidy
        out = np.clip(out, 0, 1)

    else:
        raise ValueError("method must be 'unsharp', 'rl', 'wiener', 'rl_tv', or 'wiener_rl'")

    # --- save ---
    _save_image(outp, out, p["save_bitdepth"])
    print(f"[{method.upper()}] sigma={p['psf_sigma']} iters={p['rl_iters']} "
          f"bitdepth={p['save_bitdepth']} saved -> {outp}")

if __name__ == "__main__":
    # adjust to your repo layout as before
    repo_root = Path(__file__).resolve().parent.parent
    # 👉 PUT YOUR FILE PATHS HERE (NO trailing commas!)
    PARAMS["input_path"]  = repo_root / "data/RV/EBSD raw cropped.tif"
    PARAMS["output_path"] = repo_root / "data/RV/denoised_ebsd.png"
    run()
