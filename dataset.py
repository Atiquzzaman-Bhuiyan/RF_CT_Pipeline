"""Dataset discovery helpers for the RF_CT_Pipeline project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    label: str
    subject_id: str
    slice_num: int


def _infer_label(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return path.parent.name or "unknown"


def _infer_subject_id(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_")[0]
    if "-" in stem:
        return stem.split("-")[0]
    return stem


def _infer_slice_num(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else -1


def discover_slices(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def load_signal(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return np.asarray(Image.open(file_path), dtype=float).ravel()
    raise ValueError(f"Unsupported file type: {suffix}")


def load_ct_slice(slice_num: int, ct_dir: str | Path) -> np.ndarray:
    ct_root = Path(ct_dir)
    fpath = ct_root / f"slice_{slice_num:06d}.tif"
    arr = np.asarray(Image.open(fpath), dtype=np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    return arr


def load_annotation(slice_num: int, ann_dir: str | Path, min_blob_px: int = 5) -> np.ndarray:
    ann_root = Path(ann_dir)
    s = f"{slice_num:04d}"
    particle_arr = np.asarray(Image.open(ann_root / f"annotation_slice_{s}_particle.tif"))
    background_arr = np.asarray(Image.open(ann_root / f"annotation_slice_{s}_background.tif"))

    def to_mask(array: np.ndarray) -> np.ndarray:
        if array.dtype in [np.float32, np.float64]:
            return array > 0.001
        if array.max() > 1:
            return array > 127
        return array == 1

    particle_mask = to_mask(particle_arr)
    background_mask = to_mask(background_arr)

    from scipy.ndimage import label as nd_label

    labeled_blobs, _ = nd_label(particle_mask)
    blob_sizes = np.bincount(labeled_blobs.flatten())
    too_small = np.where(blob_sizes < min_blob_px)[0]
    too_small = too_small[too_small != 0]
    for blob_id in too_small:
        particle_mask[labeled_blobs == blob_id] = False

    labels = np.full(particle_arr.shape, -1, dtype=np.int8)
    labels[background_mask] = 0
    labels[particle_mask] = 1
    return labels


def load_slice_data(slice_num: int, ct_dir: str | Path, ann_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from feature_extraction import extract_pixel_features

    arr = load_ct_slice(slice_num, ct_dir)
    labels = load_annotation(slice_num, ann_dir)
    feats = extract_pixel_features(arr)
    return arr, labels, feats


def load_records(root_dir: str | Path) -> list[SliceRecord]:
    root = Path(root_dir)
    records: list[SliceRecord] = []
    for path in discover_slices(root):
        records.append(
            SliceRecord(
                path=path,
                label=_infer_label(path, root),
                subject_id=_infer_subject_id(path),
                slice_num=_infer_slice_num(path),
            )
        )
    return records
