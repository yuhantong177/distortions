import os
import cv2
import json
import numpy as np

from matplotlib import pyplot as plt, cm
from matplotlib.patches import Polygon
from src.misc.tools import Aligner, compute_score
import argparse


def get_range(conf):
    """Return an inclusive numpy array based on ``start``/``end``/``step``.

    ``np.arange`` drops the upper bound which confused users when the best
    configuration sat near the ``end`` value (e.g. rescale sweeps stopping at
    ``0.95`` instead of reaching ``1.0``).  The helper below keeps the historical
    semantics for positive steps while guaranteeing the upper bound is included
    when the loop finishes.
    """

    start = float(conf["start"])
    end = float(conf["end"])
    step = float(conf["step"])

    if step == 0:
        raise ValueError("Step must be non-zero in grid-search ranges.")

    ascending = step > 0
    if ascending and start > end:
        raise ValueError("Start must be <= end when step is positive.")
    if not ascending and start < end:
        raise ValueError("Start must be >= end when step is negative.")

    values = []
    current = start
    tol = abs(step) * 1e-9

    # Cap the number of iterations to avoid accidental infinite loops when the
    # user supplies inconsistent ranges.
    max_iter = 1_000_000
    iter_count = 0

    compare = (lambda a, b: a <= b + tol) if ascending else (lambda a, b: a >= b - tol)

    while compare(current, end):
        values.append(round(current, 6))
        current += step
        iter_count += 1
        if iter_count > max_iter:
            raise ValueError(
                "Range produced more than {max_iter} steps. Check the configuration for infinite loops.".format(
                    max_iter=max_iter
                )
            )

    if not values:
        # Should not happen thanks to the earlier guards, but keep a fallback to
        # help debugging odd configurations.
        values.append(round(start, 6))

    return np.array(values, dtype=float)


def get_rescale_values(axis_conf):
    """Return a numpy array of rescale factors for one axis.

    The configuration accepts either a single float value (historical
    behaviour) or a dictionary with ``start``, ``end`` and ``step`` keys to
    describe a grid-search range.  Ranges are inclusive so the ``end`` value is
    always evaluated alongside the intermediate steps.
    """

    if isinstance(axis_conf, dict):
        return get_range(axis_conf)

    return np.array([axis_conf], dtype=float)


def get_rescale_values(axis_conf):
    """Return a numpy array of rescale factors for one axis.

    The configuration accepts either a single float value (historical
    behaviour) or a dictionary with ``start``, ``end`` and ``step`` keys to
    describe a grid-search range.  ``np.arange`` is used internally which means
    the upper bound is exclusive, matching the behaviour already used for the
    translation/rotation sweeps.
    """

    if isinstance(axis_conf, dict):
        return get_range(axis_conf)

    return np.array([axis_conf], dtype=float)


def __main__(args=None):

    if args is None:

        parser = argparse.ArgumentParser('Image align input!')

        parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
        parser.add_argument("-ebsd_ref_path", type=str, required=True, help="Path to the grain image")
        parser.add_argument("-conf_path", type=str, required=True, help="Path to the configuration file")
        parser.add_argument("-align_dir", type=str, required=True, help="Output directory (segment crop)")
        parser.add_argument("-out_dir", type=str, required=True, help="Output directory (image overlap + affine.pkl)")

        parser.add_argument("-id_xp", type=int, default=0, help="Define the xp id to save the results")
        parser.add_argument(
            "-score_normalization",
            type=str,
            default="dice",
            choices=["dice", "min"],
            help=(
                "Overlap score normalization. Use 'min' to normalize by the smaller mask "
                "and reduce scale bias when the two images cover different areas."
            ),
        )
        parser.add_argument(
            "--overlay_background_alpha",
            type=float,
            default=1.0,
            help="Alpha (0-1) for the SEMCL/segmented background when saving the overlay image.",
        )
        parser.add_argument(
            "--overlay_background_cmap",
            type=str,
            default="gray",
            help="Matplotlib colormap name for the SEMCL/segmented background on the overlay image.",
        )
        parser.add_argument(
            "--overlay_foreground_alpha",
            type=float,
            default=0.5,
            help="Alpha (0-1) for the EBSD foreground when saving the overlay image.",
        )
        parser.add_argument(
            "--overlay_foreground_cmap",
            type=str,
            default="jet",
            help="Matplotlib colormap name for the EBSD foreground on the overlay image.",
        )
        parser.add_argument(
            "--overlay_mask_threshold",
            type=float,
            default=None,
            help=(
                "Mask pixels at or below this threshold in both overlay layers "
                "to avoid coloring binary background pixels. Use 0 to hide zeros."
            ),
        )
        parser.add_argument(
            "--overlay_background_color_min",
            type=float,
            default=None,
            help=(
                "Minimum normalized mean intensity (0-1) required to keep the "
                "background overlay colormap. Darker images fall back to grayscale."
            ),
        )
        parser.add_argument(
            "--overlay_foreground_color_min",
            type=float,
            default=None,
            help=(
                "Minimum normalized mean intensity (0-1) required to keep the "
                "foreground overlay colormap. Darker images fall back to grayscale."
            ),
        )
        parser.add_argument(
            "--overlay_segment_outline",
            action="store_true",
            help=(
                "Draw an orange rectangle around the aligned segmentation image on the overlay output. "
                "The outline is diagnostic only and does not change the saved aligned PNG or the score."
            ),
        )
        parser.add_argument(
            "--hide_axes",
            action="store_true",
            help="Hide the X/Y axes when saving matplotlib figures.",
        )
        parser.add_argument(
            "--figure_dpi",
            type=int,
            default=300,
            help="Resolution (DPI) used when saving matplotlib figures.",
        )

        args = parser.parse_args()

    # Ensure output folders exist before attempting to write images later on.
    os.makedirs(args.align_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load configuration
    with open(args.conf_path, "r") as conf_file:
        conf = json.loads(conf_file.read())

    def _load_grayscale_image(path, label):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "{label} not found at '{path}'. Verify the CLI paths supplied to align.".format(
                    label=label, path=path
                )
            )

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(
                (
                    "OpenCV could not load the {label} from '{path}'. "
                    "Check that the file is a readable image (e.g. PNG/TIF) and not locked by another application."
                ).format(label=label, path=path)
            )
        return image

    # Load for the segment/esbd

    segment_raw = _load_grayscale_image(args.seg_ref_path, "segmented reference")

    ebsd = _load_grayscale_image(args.ebsd_ref_path, "EBSD reference")


    print(
        "Note: only the segmented image is rescaled during the grid search; "
        "the EBSD reference always stays at its original pixel size."
    )

    def _validate_alpha(value, label):
        if not 0.0 <= value <= 1.0:
            raise ValueError("{label} must be between 0 and 1 inclusive.".format(label=label))
        return value

    def _resolve_cmap(name, label):
        try:
            return cm.get_cmap(name)
        except ValueError as exc:
            raise ValueError(
                "{label} must be a valid Matplotlib colormap name (got '{name}').".format(
                    label=label, name=name
                )
            ) from exc

    def _validate_color_min(value, label):
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            raise ValueError("{label} must be between 0 and 1 inclusive.".format(label=label))
        return float(value)

    def _copy_cmap(cmap):
        if hasattr(cmap, "copy"):
            return cmap.copy()
        return cm.get_cmap(cmap.name)

    def _normalized_mean(image):
        max_val = float(np.max(image))
        if max_val == 0.0:
            return 0.0
        return float(np.mean(image)) / max_val

    def _maybe_use_gray(cmap, image, min_mean):
        if min_mean is None:
            return cmap
        if _normalized_mean(image) <= min_mean:
            return cm.get_cmap("gray")
        return cmap

    def _apply_overlay_mask(image, threshold, cmap):
        if threshold is None:
            return image, cmap
        masked = np.ma.masked_where(image <= threshold, image)
        cmap = _copy_cmap(cmap)
        cmap.set_bad(color=(0.0, 0.0, 0.0, 0.0))
        return masked, cmap

    background_alpha = _validate_alpha(args.overlay_background_alpha, "overlay_background_alpha")
    foreground_alpha = _validate_alpha(args.overlay_foreground_alpha, "overlay_foreground_alpha")
    background_cmap = _resolve_cmap(args.overlay_background_cmap, "overlay_background_cmap")
    foreground_cmap = _resolve_cmap(args.overlay_foreground_cmap, "overlay_foreground_cmap")
    overlay_mask_threshold = args.overlay_mask_threshold
    background_color_min = _validate_color_min(
        args.overlay_background_color_min, "overlay_background_color_min"
    )
    foreground_color_min = _validate_color_min(
        args.overlay_foreground_color_min, "overlay_foreground_color_min"
    )

    best_score = -1
    best_val, best_segment = None, None

    rescale_conf = conf.get("rescale", {"x": 1.0, "y": 1.0})
    rescale_x_values = get_rescale_values(rescale_conf.get("x", 1.0))
    rescale_y_values = get_rescale_values(rescale_conf.get("y", 1.0))

    rotate_values = get_range(conf["grid_search"]["rotate"])
    translate_x_values = get_range(conf["grid_search"]["translate_x"])
    translate_y_values = get_range(conf["grid_search"]["translate_y"])

    search_conf = conf.get("search", {})
    search_mode = search_conf.get("mode", "grid").lower()
    
    # Look for best alignment
    print("Look for best alignment ({mode} search)...".format(mode=search_mode))
    
    if search_mode not in {"grid", "random"}:
        raise ValueError(
            "Unsupported search mode '{mode}'. Expected 'grid' or 'random'.".format(
                mode=search_mode
            )
        )

    rng = None
    if search_mode == "random":
        samples = int(search_conf.get("samples", 500))
        if samples <= 0:
            raise ValueError("Random search requires a strictly positive 'samples' value.")
        rng = np.random.default_rng(search_conf.get("seed"))
        print(
            "Random search will evaluate {samples} combinations drawn from the configured ranges.".format(
                samples=samples
            )
        )
    else:
        samples = None
        total_candidates = (
            len(rescale_x_values)
            * len(rescale_y_values)
            * len(rotate_values)
            * len(translate_x_values)
            * len(translate_y_values)
        )
        print(
            "Grid search will evaluate {total} combinations.".format(
                total=total_candidates
            )
        )

    def _update_best(score, align_segment, scale_x, scale_y, angle, tx, ty):
        nonlocal best_score, best_segment, best_val

        if score > best_score:
            best_score = score
            best_segment = np.copy(align_segment)
            best_val = {
                "score": float(best_score),
                "translate": (int(tx), int(ty)),
                "angle": float(angle),
                "rescale": (float(scale_x), float(scale_y)),
            }
            print(
                "Score: {0:.4f}, tx: {1}, ty: {2}, angle: {3}, rescale: ({4:.4f}, {5:.4f})".format(
                    float(best_score),
                    tx,
                    ty,
                    angle,
                    scale_x,
                    scale_y,
                )
            )

    if search_mode == "grid":
        for scale_x in rescale_x_values:
            for scale_y in rescale_y_values:
                segment = Aligner.rescale(segment_raw, rescale=(scale_x, scale_y))

                for angle in rotate_values:
                    rot_segment = Aligner.rotate(np.copy(segment), angle)

                    for tx in translate_x_values:
                        for ty in translate_y_values:
                            align_segment = Aligner.translate(
                                segment=rot_segment,
                                tx=tx,
                                ty=ty,
                                shape=ebsd.shape[::-1],
                            )

                            score = compute_score(
                                segment=align_segment,
                                ebsd=ebsd,
                                normalization=args.score_normalization,
                            )
                            _update_best(score, align_segment, scale_x, scale_y, angle, tx, ty)
    else:
        # Random search samples discrete values drawn from the configured ranges.
        for sample_idx in range(samples):
            scale_x = float(rescale_x_values[0]) if len(rescale_x_values) == 1 else float(rng.choice(rescale_x_values))
            scale_y = float(rescale_y_values[0]) if len(rescale_y_values) == 1 else float(rng.choice(rescale_y_values))
            angle = float(rotate_values[0]) if len(rotate_values) == 1 else float(rng.choice(rotate_values))
            tx = int(translate_x_values[0]) if len(translate_x_values) == 1 else int(rng.choice(translate_x_values))
            ty = int(translate_y_values[0]) if len(translate_y_values) == 1 else int(rng.choice(translate_y_values))

            segment = Aligner.rescale(segment_raw, rescale=(scale_x, scale_y))
            rot_segment = Aligner.rotate(segment, angle)
            align_segment = Aligner.translate(
                segment=rot_segment,
                tx=tx,
                ty=ty,
                shape=ebsd.shape[::-1],
            )

            score = compute_score(
                segment=align_segment,
                ebsd=ebsd,
                normalization=args.score_normalization,
            )
            _update_best(score, align_segment, scale_x, scale_y, angle, tx, ty)

    if best_val is None:
        raise SystemExit(
            "align did not evaluate any configuration. Verify the grid-search ranges and rescale settings."
        )

    # Display results
    tx, ty = best_val["translate"]
    angle = best_val["angle"]
    rescale = best_val["rescale"]
    print("----------------------------------------------")
    print("Best score: ", best_score)
    print(" - tx  :  ", tx)
    print(" - ty  :  ", ty)
    print(" - rot : ", angle)
    print(" - sx  :  ", rescale[0])
    print(" - sy  :  ", rescale[1])
    print("----------------------------------------------")

    # Plot results
    out_image = os.path.join(args.out_dir, "overlap.align.{}.png".format(args.id_xp))
    fig = plt.figure(figsize=(15, 8))
    background_cmap = _maybe_use_gray(background_cmap, best_segment, background_color_min)
    foreground_cmap = _maybe_use_gray(foreground_cmap, ebsd, foreground_color_min)
    background_overlay, background_cmap = _apply_overlay_mask(
        best_segment, overlay_mask_threshold, background_cmap
    )
    foreground_overlay, foreground_cmap = _apply_overlay_mask(
        ebsd, overlay_mask_threshold, foreground_cmap
    )
    plt.imshow(
        background_overlay,
        interpolation='nearest',
        cmap=background_cmap,
        alpha=background_alpha,
    )
    plt.imshow(
        foreground_overlay,
        interpolation='nearest',
        cmap=foreground_cmap,
        alpha=foreground_alpha,
    )
    if getattr(args, "overlay_segment_outline", False):
        ax = plt.gca()
        # Track the transformed bounds of the original segment canvas by
        # applying the same rescale/rotate/translate sequence to its corner
        # coordinates. Drawing the resulting quadrilateral preserves the
        # orientation of the original image instead of collapsing to an axis-
        # aligned bounding box.
        def _transformed_corners(shape, rescale_vals, rot_angle, trans_x, trans_y):
            height, width = shape
            corners = np.array(
                [
                    [0, 0],
                    [width, 0],
                    [width, height],
                    [0, height],
                ],
                dtype=np.float32,
            )

            # Rescale the canvas corners.
            corners[:, 0] *= rescale_vals[0]
            corners[:, 1] *= rescale_vals[1]
            scaled_width = width * rescale_vals[0]
            scaled_height = height * rescale_vals[1]

            if rot_angle != 0:
                center_rotation = (scaled_width / 2.0, scaled_height / 2.0)
                m_rot = cv2.getRotationMatrix2D(center_rotation, rot_angle, 1)

                abs_cos = abs(m_rot[0, 0])
                abs_sin = abs(m_rot[0, 1])
                bound_w = float(np.ceil(scaled_height * abs_sin + scaled_width * abs_cos))
                bound_h = float(np.ceil(scaled_height * abs_cos + scaled_width * abs_sin))

                m_rot[0, 2] += bound_w / 2.0 - center_rotation[0]
                m_rot[1, 2] += bound_h / 2.0 - center_rotation[1]

                corners = cv2.transform(corners[None, :, :], m_rot)[0]
            else:
                bound_w, bound_h = scaled_width, scaled_height

            # Translate onto the EBSD canvas.
            corners[:, 0] += trans_x
            corners[:, 1] += trans_y

            return corners

        outline = _transformed_corners(segment_raw.shape, rescale, angle, tx, ty)
        ax.add_patch(
            Polygon(
                outline,
                closed=True,
                fill=False,
                edgecolor="orange",
                linewidth=0.8,
            )
        )
    if getattr(args, "hide_axes", False):
        plt.axis("off")
    fig.savefig(
        out_image,
        dpi=getattr(args, "figure_dpi", None),
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    # Store align segment
    filename_out = os.path.join(args.align_dir, "segment.align.{}.png".format(args.id_xp))
    cv2.imwrite(filename_out, best_segment)

    # Dump affine.pkl with all the required information to perform the affine transformation
    with open(os.path.join(args.out_dir, "affine.{}.json".format(args.id_xp)), "wb") as f:
        data = dict(score=best_score,
                    ebsd=os.path.basename(args.ebsd_ref_path),
                    segment=os.path.basename(args.seg_ref_path),
                    segment_shape=list(segment_raw.shape[:2]),
                    conf=conf,
                    rescale=rescale,
                    translate=[int(tx), int(ty)],
                    angle=float(angle))
        results_json = json.dumps(data)

        f.write(results_json.encode('utf8', 'replace'))

    return best_score, best_segment, data


if __name__ == "__main__":
    __main__()
