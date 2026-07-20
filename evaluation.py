"""Model evaluation helpers for the RF_CT_Pipeline project."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(y_true, y_pred, y_proba=None) -> dict[str, float | list[list[int]]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba.ravel()))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics
