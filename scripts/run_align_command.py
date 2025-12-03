"""Helper script to launch the alignment step from an IDE or terminal.

Edit the path constants below so they point to your dataset, then run the
module from your IDE or a terminal (``python scripts/run_align_command.py``).
The script simply forwards the call to ``python -m src.align`` using the same
interpreter that executed the file, which guarantees the package imports
resolve correctly as long as you start it from inside the repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

# =============================================================================
#     seg_ref_path = repo_root / "data/AM718/segment/AM718_segment.png"
#     ebsd_ref_path = repo_root / "data/AM718/ebsd/AM718_ebsd.jpg"
#     config_path = repo_root / "conf/AM718.align.conf"
#     align_dir = repo_root / "data/AM718/segment.align"
#     out_dir = repo_root / "data/AM718/out"
# =============================================================================
# =============================================================================
#     seg_ref_path = repo_root / "data/RV/binary_semcl_halfstone_floyd_cropped_compress.png"
#     ebsd_ref_path = repo_root / "data/RV/binary_ebsd_halfstone_floyd_maskout_black_compress.png"
# =============================================================================
    seg_ref_path = repo_root / "data/RV/denoised_SEMCL_psfsize51_sigma15_iter20_cropped_threshold 170-255 compressed.png"
    ebsd_ref_path = repo_root / "data/RV/EBSD threshold 184-255 compressed.png"
    config_path = repo_root / "conf/RV.align.conf"
    align_dir = repo_root / "data/RV/segment.align"
    out_dir = repo_root / "data/RV/out"    
    xp_id = 0
    alpha_foreground = 0.4 # transparency between 0-1
    alpha_background = 1 # transparency between 0-1

    align_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "src.align",
        "-seg_ref_path",
        str(seg_ref_path),
        "-ebsd_ref_path",
        str(ebsd_ref_path),
        "-conf_path",
        str(config_path),
        "-align_dir",
        str(align_dir),
        "-out_dir",
        str(out_dir),
        "-id_xp",
        str(xp_id),
        "--overlay_foreground_alpha",
        str(alpha_foreground),
        "--overlay_background_alpha",
        str(alpha_background),
    ]

    print(f"Configuration template: {config_path}")
    print("Running:", " ".join(command))

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise SystemExit(
            "align failed. Review the error output above (missing Python "
            "packages such as numpy/opencv is the most common cause)."
        )


if __name__ == "__main__":
    main()
