"""Stage 1 interdot detection — THE file the autoresearch agent edits.

Pipeline (baseline):
    1. robust thresholding of dark pixels (median − k·MAD) on a lightly smoothed image;
    2. probabilistic Hough transform restricted to angles around the stick orientation;
    3. segments drawn, dilated, intersected with a looser threshold (``k_fill``);
    4. angle filter: connected components whose main axis is not aligned with the
       sticks (e.g. fragments of charge-transition lines) are deleted.

Run (prints a grep-able summary, the key line is ``val_dice:``)::

    uv run python autoresearch/detect.py > autoresearch/run.log 2>&1
"""

from __future__ import annotations

import time

import numpy as np
from prepare import TIME_BUDGET, evaluate_dice, load_train, parse_batch_size  # noqa: F401
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.draw import line as draw_line
from skimage.measure import label, regionprops
from skimage.transform import probabilistic_hough_line

STICK_THETA = np.pi / 4  # expected stick orientation [rad], image displayed with origin="lower"

# Parameters found by random search (300 trials) on a separate tuning set.
PARAMS = {
    "sigma": 0.29,
    "k": 2.72,
    "k_fill": 5.01,
    "theta_tol": 0.26,
    "threshold": 5,
    "line_length": 2,
    "line_gap": 2,
    "dilate": 1,
}
ANGLE_TOL = 0.6  # max deviation [rad] between a component's main axis and STICK_THETA
ANGLE_MIN_LEN = 3.0  # components shorter than this [px] have no reliable angle: kept


def hough_mask(
    image,
    sigma=0.5,
    k=4.0,
    k_fill=4.0,
    theta_tol=0.4,
    threshold=3,
    line_length=2,
    line_gap=1,
    dilate=1,
    seed=0,
):
    """Predict an interdot mask with a probabilistic Hough transform.

    Args:
        image: Raw 2D CSD image (interdots are dark dips).
        sigma: Std of the Gaussian smoothing applied before thresholding.
        k: Detection threshold, in MADs below the median, for the Hough input.
        k_fill: Threshold (in MADs) of pixels kept around detected segments.
        theta_tol: Angular tolerance around ``STICK_THETA`` [rad].
        threshold: Minimum number of Hough accumulator votes.
        line_length: Minimum segment length [px].
        line_gap: Maximum gap between pixels of a same segment [px].
        dilate: Dilation of the drawn segments [px] before the ``k_fill`` threshold.
        seed: Seed of ``probabilistic_hough_line``'s random sampling.

    Returns:
        Boolean mask with the same shape as ``image``.
    """
    smooth = gaussian_filter(np.asarray(image, dtype=float), sigma)
    med = np.median(smooth)
    mad = 1.4826 * np.median(np.abs(smooth - med))
    depth = (med - smooth) / mad  # dip depth, in noise units

    # skimage parametrises a line by the angle of its normal: theta_stick - pi/2.
    normal = STICK_THETA - np.pi / 2
    thetas = np.linspace(normal - theta_tol, normal + theta_tol, 41)
    segments = probabilistic_hough_line(
        depth > k,
        threshold=threshold,
        line_length=line_length,
        line_gap=line_gap,
        theta=thetas,
        rng=seed,
    )

    lines = np.zeros(image.shape, dtype=bool)
    for (x0, y0), (x1, y1) in segments:
        rr, cc = draw_line(y0, x0, y1, x1)
        lines[rr, cc] = True
    if dilate > 0:
        lines = binary_dilation(lines, iterations=dilate)
    return lines & (depth > k_fill)


def filter_by_angle(mask, stick_theta=STICK_THETA, angle_tol=ANGLE_TOL, min_len=ANGLE_MIN_LEN):
    """Delete connected components whose orientation does not match the sticks.

    Args:
        mask: Boolean mask (rows = y, cols = x, origin="lower").
        stick_theta: Expected stick orientation [rad].
        angle_tol: Max allowed deviation of a component's main axis [rad].
        min_len: Components with a main axis shorter than this [px] are kept as is.

    Returns:
        Filtered boolean mask.
    """
    labels = label(mask, connectivity=2)
    out = mask.copy()
    for region in regionprops(labels):
        if region.axis_major_length < min_len:
            continue
        # regionprops orientation is measured from the row axis; convert to an x-y angle.
        phi = np.arctan2(np.cos(region.orientation), np.sin(region.orientation)) % np.pi
        deviation = abs((phi - stick_theta + np.pi / 2) % np.pi - np.pi / 2)
        if deviation > angle_tol:
            out[labels == region.label] = False
    return out


def predict(image):
    """Full pipeline: raw image -> binary interdot mask (uint8)."""
    mask = hough_mask(image, **PARAMS)
    mask = filter_by_angle(mask)
    return mask.astype(np.uint8)


if __name__ == "__main__":
    batch_size = parse_batch_size()  # --batch-size N (fixed for a whole series)
    t0 = time.time()
    # Optional tuning on load_train(batch_size) goes here; it must stop within TIME_BUDGET s.
    tune_seconds = time.time() - t0

    val_dice = evaluate_dice(predict, batch_size)
    print("---")
    print(f"val_dice:        {val_dice:.6f}")
    print(f"batch_size:      {batch_size}")
    print(f"tune_seconds:    {tune_seconds:.1f}")
    print(f"total_seconds:   {time.time() - t0:.1f}")
