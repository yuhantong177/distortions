import os
import cv2
import json
import numpy as np

from matplotlib import pyplot as plt, cm
from src.misc.tools import Aligner, compute_score
import argparse


def get_range(conf):
    return np.arange(conf["start"], conf["end"], conf["step"])


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

    # Look for best alignment
    print("Look for best alignment...")
    best_score = -1
    best_val, best_segment = None, None

    rescale_conf = conf.get("rescale", {"x": 1.0, "y": 1.0})
    rescale_x_values = get_rescale_values(rescale_conf.get("x", 1.0))
    rescale_y_values = get_rescale_values(rescale_conf.get("y", 1.0))

    for scale_x in rescale_x_values:
        for scale_y in rescale_y_values:
            segment = Aligner.rescale(segment_raw, rescale=(scale_x, scale_y))

            for angle in get_range(conf["grid_search"]["rotate"]):

                # use a private method to avoid rotating at each iteration
                init_segment = np.copy(segment)
                rot_segment = Aligner.rotate(init_segment, angle)

                for tx in get_range(conf["grid_search"]["translate_x"]):
                    for ty in get_range(conf["grid_search"]["translate_y"]):

                        align_segment = Aligner.translate(segment=rot_segment,
                                                          tx=tx, ty=ty,
                                                          shape=ebsd.shape[::-1])

                        score = compute_score(segment=align_segment, ebsd=ebsd)

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
    plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
    plt.imshow(ebsd, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(out_image)

    # Store align segment
    filename_out = os.path.join(args.align_dir, "segment.align.{}.png".format(args.id_xp))
    cv2.imwrite(filename_out, best_segment)

    # Dump affine.pkl with all the required information to perform the affine transformation
    with open(os.path.join(args.out_dir, "affine.{}.json".format(args.id_xp)), "wb") as f:
        data = dict(score=best_score,
                    ebsd=os.path.basename(args.ebsd_ref_path),
                    segment=os.path.basename(args.seg_ref_path),
                    conf=conf,
                    rescale=rescale,
                    translate=[int(tx), int(ty)],
                    angle=float(angle))
        results_json = json.dumps(data)

        f.write(results_json.encode('utf8', 'replace'))

    return best_score, best_segment, data


if __name__ == "__main__":
    __main__()
