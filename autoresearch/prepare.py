"""Fixed constants, one-time data prep and evaluation for the Stage 1 autoresearch loop.

DO NOT MODIFY (agent). This file is the ground truth: datasets, time budget and the
``evaluate_dice`` metric. The agent only edits ``autoresearch/detect.py``.

Usage (one-time, generates the datasets if missing)::

    uv run python autoresearch/prepare.py
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from csd import generate_dataset, load_dataset

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------
TRAIN_DIR = REPO / "data" / "ar_train"  # tuning set: images + masks usable by detect.py
TRAIN_N, TRAIN_SEED = 64, 1
VAL_DIR = REPO / "data" / "ar_val"  # evaluation set: masks reserved to evaluate_dice
VAL_N, VAL_SEED = 128, 2
TIME_BUDGET = 60.0  # seconds allowed for any tuning/search inside detect.py


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def _ensure(out_dir: pathlib.Path, n: int, seed: int) -> None:
    """Generate a dataset unless a complete one (with ``meta.json``) already exists."""
    if (out_dir / "meta.json").exists():
        print(f"ok      {out_dir} (déjà présent)")
        return
    print(f"génère  {out_dir} (n={n}, seed={seed}) ...")
    generate_dataset(n=n, out_dir=out_dir, seed=seed, overwrite=True)


def prepare() -> None:
    """Generate the train and validation datasets if they are missing."""
    _ensure(TRAIN_DIR, TRAIN_N, TRAIN_SEED)
    _ensure(VAL_DIR, VAL_N, VAL_SEED)


# ---------------------------------------------------------------------------
# Runtime utilities
# ---------------------------------------------------------------------------
def load_train() -> tuple[np.ndarray, np.ndarray]:
    """Load the tuning set.

    Returns:
        ``(images, masks)`` as in-memory arrays of shape ``(n, 150, 150)``.
    """
    ds = load_dataset(TRAIN_DIR)
    return np.asarray(ds["images"]), np.asarray(ds["masks"])


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice score between two binary masks (1.0 when both are empty).

    Args:
        pred: Predicted mask.
        target: Target mask.

    Returns:
        ``2·|P∩C| / (|P| + |C|)``.
    """
    pred, target = pred > 0, target > 0
    total = pred.sum() + target.sum()
    return 1.0 if total == 0 else float(2 * (pred & target).sum() / total)


def evaluate_dice(predict_fn: Callable[[np.ndarray], np.ndarray]) -> float:
    """Ground-truth metric: mean per-image Dice of ``predict_fn`` on the validation set.

    Args:
        predict_fn: Function mapping a raw 2D image to a binary mask of the same shape.

    Returns:
        Mean Dice over the ``VAL_N`` validation images (higher is better).
    """
    ds = load_dataset(VAL_DIR)
    images, masks = np.asarray(ds["images"]), np.asarray(ds["masks"])
    scores = []
    for image, target in zip(images, masks):
        pred = np.asarray(predict_fn(image.copy()))
        if pred.shape != target.shape:
            raise ValueError(f"predict_fn returned shape {pred.shape}, expected {target.shape}")
        scores.append(dice(pred, target))
    return float(np.mean(scores))


if __name__ == "__main__":
    prepare()
