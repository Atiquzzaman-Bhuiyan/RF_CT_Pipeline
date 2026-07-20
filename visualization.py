"""Visualization helpers for the RF_CT_Pipeline project."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_roc_curve(y_true, y_score, class_names=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    figure, axis = plt.subplots(figsize=(6, 5))

    if y_score.ndim == 1 or y_score.shape[1] == 1:
        fpr, tpr, _ = roc_curve(y_true, y_score.ravel())
        axis.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        for index in range(y_score.shape[1]):
            fpr, tpr, _ = roc_curve((y_true == index).astype(int), y_score[:, index])
            label = class_names[index] if class_names and index < len(class_names) else f"Class {index}"
            axis.plot(fpr, tpr, label=f"{label} (AUC = {auc(fpr, tpr):.3f})")

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(image, ax=axis)
    ticks = np.arange(matrix.shape[0])
    labels = class_names if class_names is not None else [str(tick) for tick in ticks]
    axis.set(xticks=ticks, yticks=ticks, xticklabels=labels, yticklabels=labels, ylabel="True label", xlabel="Predicted label", title="Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, format(matrix[row, col], "d"), ha="center", va="center", color="white" if matrix[row, col] > threshold else "black")

    figure.tight_layout()
    return figure
