"""
Method E v3 â intensity-seeded + gradient-relief variant (new, additive only)
=================================================================================
v2 (method_e_v2.py) fixed the watershed *relief* surface (gradient instead
of -distance) but kept the original *seed placement*: local maxima of the
distance transform, which are purely geometric -- "center of the connected
binary blob" -- and don't know whether that spot is a real particle or
matrix. On fine/dense regions (e.g. slice 800) that still produced a
roughly-uniform honeycomb, just with edges that snap to gradients now.

v3 fixes seed placement too: seeds are local maxima of *smoothed intensity*
within the reconstructed mask (the same technique run_method_b already uses
for its seeds), so they land at actual bright particle cores instead of
arbitrary geometric centers. Combined with the v2 gradient relief, each
seed now grows outward from a real particle and stops at a real intensity
edge. Verified on slice 800: large/medium particles now get clean, closed
contours matching their true silhouettes instead of the honeycomb mesh.

Kept separate from methods_a_b_e.py (original) and method_e_v2.py (relief
fix only) so all three remain available for comparison -- nothing in either
is modified by this file.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label, maximum_filter, gaussian_filter, binary_dilation
from skimage.morphology import remove_small_objects, ball, opening
from skimage.segmentation import watershed
from skimage.filters import sobel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Methods_ABE'))
from methods_a_b_e import (OTSU_LOWER, OTSU_UPPER, MIN_DIAMETER_UM,        # noqa: E402
                            VOXEL_SIZE_UM, sample_mask, normalize_volume)  # noqa: E402


def run_method_e_v3(volume):
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

    smoothed = gaussian_filter(volume, sigma=2)
    smooth_max = maximum_filter(smoothed * reconstructed, size=13)
    peak_mask = ((smoothed * reconstructed == smooth_max) &
                 (smoothed > OTSU_LOWER + 0.05) & reconstructed)
    markers, n_seeds = label(peak_mask)
    if n_seeds < 10:
        peak_mask = ((smoothed * reconstructed == smooth_max) &
                     (smoothed > OTSU_LOWER) & reconstructed)
        markers, _ = label(peak_mask)

    gx = sobel(volume, axis=2)
    gy = sobel(volume, axis=1)
    gz = sobel(volume, axis=0)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)

    labeled = watershed(grad, markers, mask=reconstructed)
    min_vox_e = int((150.0 / VOXEL_SIZE_UM) ** 3 * (np.pi / 6))
    return remove_small_objects(labeled, min_size=min_vox_e)
