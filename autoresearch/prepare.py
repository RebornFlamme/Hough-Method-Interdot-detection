"""Fixed constants, one-time data prep and evaluation for the Stage 1 autoresearch loop.

DO NOT MODIFY (agent). This file is the ground truth: datasets, time budget, batch size
handling and the ``evaluate_dice`` metric. The agent only edits ``autoresearch/detect.py``.

Usage (one-time, generates the datasets if missing or too small)::

    uv run python autoresearch/prepare.py [--batch-size N]
"""

from __future__ import annotations

import argparse
import json
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
TRAIN_SEED = 1
VAL_DIR = REPO / "data" / "ar_val"  # evaluation set: masks reserved to evaluate_dice
VAL_SEED = 2
DEFAULT_BATCH_SIZE = 128  # number of images used per evaluation (and per tuning set)
TIME_BUDGET = 60.0  # seconds allowed for any tuning/search inside detect.py


# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------
def parse_batch_size(argv: list[str] | None = None) -> int:
    """Read ``--batch-size N`` from the command line.

    Args:
        argv: Arguments to parse (defaults to ``sys.argv[1:]``).

    Returns:
        Number of images used per evaluation (``DEFAULT_BATCH_SIZE`` if not given).
    """
    parser = argparse.ArgumentParser(description="Stage 1 autoresearch")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"images par évaluation (défaut {DEFAULT_BATCH_SIZE})",
    )
    batch_size = parser.parse_args(argv).batch_size
    if batch_size < 1:
        parser.error("--batch-size doit être >= 1")
    return batch_size


def _available(out_dir: pathlib.Path) -> int:
    """Number of images in a complete dataset (0 if missing or partial)."""
    meta = out_dir / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8"))["n"] if meta.exists() else 0


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def _ensure(out_dir: pathlib.Path, n: int, seed: int) -> None:
    """Generate a dataset unless a complete one with at least ``n`` images exists.

    Images are drawn sequentially from ``np.random.seed(seed)``, so the first ``n``
    images of a bigger dataset are exactly a dataset of size ``n``: scores stay
    comparable whatever size was generated.
    """
    have = _available(out_dir)
    if have >= n:
        print(f"ok      {out_dir} ({have} images >= {n})")
        return
    print(f"génère  {out_dir} (n={n}, seed={seed}) ...")
    generate_dataset(n=n, out_dir=out_dir, seed=seed, overwrite=True)


def prepare(batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """Generate the train and validation datasets if missing or too small."""
    _ensure(TRAIN_DIR, batch_size, TRAIN_SEED)
    _ensure(VAL_DIR, batch_size, VAL_SEED)


# ---------------------------------------------------------------------------
# Runtime utilities
# ---------------------------------------------------------------------------
def _load(out_dir: pathlib.Path, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the first ``batch_size`` images and masks of a dataset."""
    have = _available(out_dir)
    if have < batch_size:
        raise RuntimeError(
            f"{out_dir} contient {have} images < batch_size={batch_size}. Lance d'abord :\n"
            f"  uv run python autoresearch/prepare.py --batch-size {batch_size}"
        )
    ds = load_dataset(out_dir)
    return np.asarray(ds["images"][:batch_size]), np.asarray(ds["masks"][:batch_size])


def load_train(batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Load the tuning set.

    Args:
        batch_size: Number of images to load.

    Returns:
        ``(images, masks)`` as in-memory arrays of shape ``(batch_size, 150, 150)``.
    """
    return _load(TRAIN_DIR, batch_size)


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


def evaluate_dice(
    predict_fn: Callable[[np.ndarray], np.ndarray], batch_size: int = DEFAULT_BATCH_SIZE
) -> float:
    """Ground-truth metric: mean per-image Dice of ``predict_fn`` on the validation set.

    Args:
        predict_fn: Function mapping a raw 2D image to a binary mask of the same shape.
        batch_size: Number of validation images evaluated (the first ones of ``ar_val``).

    Returns:
        Mean Dice over the ``batch_size`` validation images (higher is better).
    """
    images, masks = _load(VAL_DIR, batch_size)
    scores = []
    for image, target in zip(images, masks):
        pred = np.asarray(predict_fn(image.copy()))
        if pred.shape != target.shape:
            raise ValueError(f"predict_fn returned shape {pred.shape}, expected {target.shape}")
        scores.append(dice(pred, target))
    return float(np.mean(scores))


if __name__ == "__main__":
    prepare(parse_batch_size())
