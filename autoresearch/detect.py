"""Stage 1 interdot detection — THE file the autoresearch agent edits.

Pipeline:
    1. per-row offset removal (scan-line drift), depth in noise units;
    2. seeds: pixels whose depth averaged along the stick direction (matched filter)
       is significant;
    3. around the seeds, keep pixels above half of their component's peak depth (FWHM);
    4. angle filter: connected components whose main axis is not aligned with the
       sticks (e.g. fragments of charge-transition lines) are deleted.

Run (prints a grep-able summary, the key line is ``val_dice:``)::

    uv run python autoresearch/detect.py > autoresearch/run.log 2>&1
"""

from __future__ import annotations

import time

import numpy as np
from prepare import TIME_BUDGET, evaluate_dice, load_train, parse_batch_size  # noqa: F401
from scipy.ndimage import binary_dilation, correlate
from scipy.ndimage import label as ndlabel
from scipy.ndimage import maximum as ndmaximum
from skimage.measure import label, regionprops

STICK_THETA = np.pi / 4  # expected stick orientation [rad], image displayed with origin="lower"

# Parameters tuned on load_train() (grid / coordinate search).
PARAMS = {
    "k_line": 3.5,
    "line_len": 5,
    "dilate": 2,
    "k_low": 2.0,
    "half": 0.5,
    "k_peak": 3.5,
}
ANGLE_TOL = 0.6  # max deviation [rad] between a component's main axis and STICK_THETA
ANGLE_MIN_LEN = 2.0  # components shorter than this [px] have no reliable angle: kept


def robust_depth(a):
    """Dip depth in noise units: (median − a) / (1.4826·MAD)."""
    med = np.median(a)
    return (med - a) / (1.4826 * np.median(np.abs(a - med)))


def dip_mask(
    image,
    k_line=3.5,
    line_len=5,
    dilate=2,
    k_low=2.0,
    half=0.5,
    k_peak=3.5,
):
    """Predict an interdot mask from the dark dips of a CSD.

    Args:
        image: Raw 2D CSD image (interdots are dark dips).
        k_line: Seed threshold (in MADs) on the depth averaged along the stick.
        line_len: Length [px] of the averaging along the stick (matched filter).
        dilate: Dilation of the seeds [px] before the depth thresholds.
        k_low: Minimum depth (in MADs) of kept pixels.
        half: Fraction of its component's peak depth a pixel must reach (0.5 = FWHM).
        k_peak: Minimum peak depth (in MADs) of a kept component.

    Returns:
        Boolean mask with the same shape as ``image``.
    """
    image = np.asarray(image, dtype=float)
    # Charge-sensor drift between scan lines adds a per-row offset: remove it.
    image = image - np.median(image, axis=1, keepdims=True)
    depth = robust_depth(image)

    # Matched filter: averaging along the stick direction raises the SNR of faint
    # interdots (white noise averages out, the dip does not).
    along = robust_depth(correlate(image, np.eye(line_len) / line_len, mode="nearest"))
    seeds = along > k_line
    if dilate > 0:
        seeds = binary_dilation(seeds, iterations=dilate)
    cand = seeds & (depth > k_low)

    # An interdot dip has a roughly uniform amplitude along its length, while the
    # charge-transition lines attached to its ends are shallower and its thermal /
    # tunnel broadening forms tails across it: keep, in each component, the pixels
    # above half of its peak depth (full width at half maximum, in every direction).
    labels, n = ndlabel(cand, structure=np.ones((3, 3)))
    peak = np.concatenate([[0.0], ndmaximum(depth, labels, np.arange(1, n + 1))])[labels]
    return cand & (depth > half * peak) & (peak > k_peak)


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
    mask = dip_mask(image, **PARAMS)
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
