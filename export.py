"""Export helpers for the RF_CT_Pipeline project."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.figure
import numpy as np


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def export_results(output_dir: str | Path, results: dict) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "results.json"
    target.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    return target


def export_figure(output_dir: str | Path, figure: matplotlib.figure.Figure, filename: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / filename
    figure.savefig(target, bbox_inches="tight")
    return target
