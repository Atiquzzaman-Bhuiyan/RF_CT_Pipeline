"""Validation strategies for the RF_CT_Pipeline project."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold


def run_validation(
    samples: Sequence[object],
    mode: str = "loso",
    fold_count: int = 5,
    labels: Sequence[object] | None = None,
    groups: Sequence[object] | None = None,
) -> dict[str, object]:
    if mode not in {"loso", "kfold"}:
        raise ValueError("mode must be 'loso' or 'kfold'")

    sample_count = len(samples)
    indices = np.arange(sample_count)
    if sample_count == 0:
        return {"mode": mode, "fold_count": fold_count, "sample_count": 0, "folds": []}

    if sample_count == 1:
        return {"mode": mode, "fold_count": fold_count, "sample_count": 1, "folds": [{"train": [], "test": [0]}]}

    if mode == "loso":
        resolved_groups = np.asarray(groups if groups is not None else indices)
        splitter = LeaveOneGroupOut()
        folds = [
            {"train": train_idx.tolist(), "test": test_idx.tolist()}
            for train_idx, test_idx in splitter.split(indices, labels, resolved_groups)
        ]
    else:
        if labels is not None:
            label_array = np.asarray(labels)
            class_counts = np.unique(label_array, return_counts=True)[1]
            n_splits = max(2, min(fold_count, sample_count, int(class_counts.min())))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            folds = [
                {"train": train_idx.tolist(), "test": test_idx.tolist()}
                for train_idx, test_idx in splitter.split(indices, labels)
            ]
        else:
            n_splits = max(2, min(fold_count, sample_count))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            folds = [
                {"train": train_idx.tolist(), "test": test_idx.tolist()}
                for train_idx, test_idx in splitter.split(indices)
            ]

    return {
        "mode": mode,
        "fold_count": fold_count,
        "sample_count": sample_count,
        "folds": folds,
    }
