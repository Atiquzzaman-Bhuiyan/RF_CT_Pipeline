"""
Methods A, B, E â extracted from full_volume_run_v2.py, plus a specimen-mask fix
====================================================================================
Base algorithms are unchanged from full_volume_run_v2.py (Otsu watershed,
intensity-gradient watershed, morphological reconstruction watershed), and
the original full_volume_run_v2.py itself is never touched -- only this
extracted copy.

ADDED (2026-08-11): normalize_volume() and sample_mask(). Root cause
investigation found that ~14% of slices in the source CT TIFF stack (e.g.
350, 700; confirmed by sampling every 20th slice 0-980) have a handful of
extreme outlier pixels -- as few as 16 pixels (0.006% of the image) on
slice 350 -- that drag a naive per-slice min()/max() normalization onto a
completely different scale than the other ~86% of "normal" slices (e.g.
raw min -0.494 vs the bulk-distribution 0.01st percentile of 0.012, which
matches normal slices almost exactly). On an affected slice, per-slice
min-max normalization then rescales background/matrix pixels up into the
middle of the OTSU_LOWER/OTSU_UPPER band instead of leaving them near 0,
so background gets segmented as particle material, AND (for Method B) the
seed-finding step -- which depends on particles standing out as isolated
local maxima against darker surrounding matrix -- collapses into a handful
of giant regions because matrix and particles are no longer separated by
the fixed threshold.

normalize_volume() fixes this at the source: it uses per-slice percentile
bounds (0.01/99.99) instead of true min/max, which is insensitive to a
few outlier pixels while leaving "normal" slices' effective range almost
unchanged (verified: their 0.01/99.99 percentiles nearly equal their true
min/max already). sample_mask() is a second, independent layer of defense:
it finds the specimen disc per slice using an *adaptive* per-slice Otsu
split (skimage.filters.threshold_otsu), which only depends on that slice's
own bimodal histogram shape, not a fixed absolute cutoff, so it isn't
fooled by any remaining scale mismatch. The largest connected foreground
component (holes filled) is taken as the specimen region; everything
outside it is excluded before any of the three methods' own
thresholding/watershed logic runs. Verified visually on slices 150/350/800.

Methods C and D (RF-based) are intentionally excluded â already present,
and untouched, elsewhere in the project (ZoneAnalysis/pipeline_common.py,
pipeline_method_d.py). These fixes only apply here.

Source: full_volume_run_v2.py
"""

import numpy as np
from scipy.ndimage import (distance_transform_edt, label,
                           maximum_filter, gaussian_filter,
                           binary_dilation, binary_fill_holes, uniform_filter)
from skimage.morphology import (remove_small_objects,
                                ball, opening, closing)
from skimage.segmentation import watershed
from skimage.filters import sobel, threshold_otsu

# ============================================================
# CONFIGURATION (shared constants used by methods A, B, E)
# ============================================================
VOXEL_SIZE_UM   = 19.8
OTSU_LOWER      = 0.40
OTSU_UPPER      = 0.85
MIN_DIAMETER_UM = 100.0
MIN_VOXELS      = int((MIN_DIAMETER_UM / VOXEL_SIZE_UM) ** 3 * (np.pi / 6))


def compute_diameters(labeled):
    diams = []
    counts = np.bincount(labeled.flatten())
    for region_id, voxels in enumerate(counts):
        if region_id == 0:
            continue
        if voxels >= MIN_VOXELS:
            d = ((6 * voxels / np.pi) ** (1/3)) * VOXEL_SIZE_UM
            diams.append(d)
    return np.array(diams)


def normalize_volume(volume, lo_pct=0.01, hi_pct=99.99):
    """Per-slice min-max to 0-1, using robust percentile bounds instead of
    the true min/max. See module docstring for why this is needed."""
    out = np.empty(volume.shape, dtype=np.float32)
    for i in range(volume.shape[0]):
        arr = volume[i].astype(np.float32)
        lo, hi = np.percentile(arr, lo_pct), np.percentile(arr, hi_pct)
        out[i] = np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(arr)
    return out


def sample_mask(volume):
    """Per-slice specimen-disc mask, robust to slices whose raw values are
    stored on a shifted scale. See module docstring for why this is needed.
    """
    mask = np.zeros(volume.shape, dtype=bool)
    for i in range(volume.shape[0]):
        sl = volume[i]
        if sl.max() <= sl.min():
            continue
        t = threshold_otsu(sl)
        fg = sl > t
        labeled_fg, n = label(fg)
        if n == 0:
            continue
        sizes = np.bincount(labeled_fg.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        mask[i] = binary_fill_holes(labeled_fg == largest)
    return mask


# ============================================================
# METHODS A, B, E (same parameters as zone comparison)
# ============================================================

def run_method_a(volume):
    binary = (volume > OTSU_LOWER) & (volume < OTSU_UPPER) & sample_mask(volume)
    binary = opening(binary, ball(1))
    binary = closing(binary, ball(1))
    dist = distance_transform_edt(binary)
    dist_max = maximum_filter(dist, size=15)
    peak_mask = (dist == dist_max) & (dist > 3) & binary
    markers, _ = label(peak_mask)
    labeled = watershed(-dist, markers, mask=binary)
    return remove_small_objects(labeled, min_size=MIN_VOXELS)


def run_method_b(volume):
    binary = (volume > OTSU_LOWER) & (volume < OTSU_UPPER) & sample_mask(volume)
    gx = sobel(volume, axis=2)
    gy = sobel(volume, axis=1)
    gz = sobel(volume, axis=0)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)
    dist = distance_transform_edt(binary)
    combined = 0.6 * grad + 0.4 * (1.0 / (dist + 1.0))
    smoothed = gaussian_filter(volume, sigma=2)
    smooth_max = maximum_filter(smoothed * binary, size=13)
    peak_mask = ((smoothed * binary == smooth_max) &
                 (smoothed > OTSU_LOWER + 0.08) & binary)
    if peak_mask.sum() < 100:
        peak_mask = ((smoothed * binary == smooth_max) &
                     (smoothed > OTSU_LOWER + 0.04) & binary)
    markers, _ = label(peak_mask)
    labeled = watershed(combined, markers, mask=binary)
    return remove_small_objects(labeled, min_size=MIN_VOXELS)


def run_method_e(volume):
    binary = (volume > OTSU_LOWER) & (volume < OTSU_UPPER) & sample_mask(volume)
    min_radius = max(3, int((MIN_DIAMETER_UM / VOXEL_SIZE_UM) / 2))
    opened = opening(binary, ball(min_radius))
    reconstructed = opened.copy()
    for _ in range(50):
        expanded = binary_dilation(reconstructed, structure=ball(1))
        new = expanded & binary
        if np.array_equal(new, reconstructed):
            break
        reconstructed = new
    dist = distance_transform_edt(reconstructed)
    dist_max = maximum_filter(dist, size=13)
    peak_mask = (dist == dist_max) & (dist > 2) & reconstructed
    markers, _ = label(peak_mask)
    labeled = watershed(-dist, markers, mask=reconstructed)
    min_vox_e = int((150.0 / VOXEL_SIZE_UM) ** 3 * (np.pi / 6))
    return remove_small_objects(labeled, min_size=min_vox_e)
