"""Feature extraction helpers for the RF_CT_Pipeline project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import sobel

from config import PipelineConfig


@dataclass
class FeatureSet:
    features: np.ndarray
    labels: np.ndarray | None = None
    ids: list[str] | None = None


def _safe_moment(values: np.ndarray, order: int) -> float:
    centered = values - values.mean()
    std = float(values.std(ddof=0))
    if std == 0:
        return 0.0
    normalized = centered / std
    return float(np.mean(normalized**order))


def extract_features(signal: Iterable[float], config: PipelineConfig | None = None) -> dict[str, float]:
    values = np.asarray(list(signal), dtype=float)
    if values.size == 0:
        raise ValueError("signal must contain at least one value")

    q25, q75 = np.percentile(values, [25, 75])

    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "q25": float(q25),
        "q75": float(q75),
        "range": float(values.max() - values.min()),
        "energy": float(np.sum(values ** 2)),
        "skew": _safe_moment(values, 3),
        "kurtosis": _safe_moment(values, 4) - 3.0,
    }


def extract_pixel_features(arr: np.ndarray) -> np.ndarray:
    """Extract the 11 per-pixel CT features used by the Colab workflow."""

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("CT slices must be 2D arrays")

    g1 = gaussian_filter(arr, sigma=1)
    g2 = gaussian_filter(arr, sigma=2)
    g3 = gaussian_filter(arr, sigma=3)
    gx = sobel(arr, axis=1)
    gy = sobel(arr, axis=0)
    local_contrast = arr - uniform_filter(arr, size=5)
    h_elems = hessian_matrix(arr, sigma=2, order="rc")
    eigs = hessian_matrix_eigvals(h_elems)

    stack = np.stack(
        [
            arr,
            g1,
            g2,
            g3,
            np.sqrt(gx**2 + gy**2),
            g1 - g3,
            g1 - g2,
            local_contrast,
            eigs[0],
            eigs[1],
            np.sqrt(eigs[0] ** 2 + eigs[1] ** 2),
        ],
        axis=-1,
    )
    return stack.reshape(-1, stack.shape[-1])


def extract_feature_matrix(signals: Iterable[Iterable[float]], config: PipelineConfig | None = None) -> tuple[np.ndarray, list[str]]:
    config = config or PipelineConfig()
    rows = [extract_features(signal, config=config) for signal in signals]
    if not rows:
        return np.empty((0, len(config.feature_names))), list(config.feature_names)

    matrix = np.asarray([[row[name] for name in config.feature_names] for row in rows], dtype=float)
    return matrix, list(config.feature_names)
