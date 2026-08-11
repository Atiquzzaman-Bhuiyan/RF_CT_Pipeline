"""
Method E v2 â gradient-relief variant (new, does not replace the original)
=============================================================================
The original run_method_e (Methods_ABE/methods_a_b_e.py, untouched by this
file) seeds watershed from local maxima of the binary mask's distance
transform, then floods using -dist as the relief surface. That means cell
walls form at a roughly fixed geometric spacing determined by the
maximum_filter window, regardless of where real particle edges are -- on
fine/densely-packed matrix regions (e.g. slice 800) the resulting
tessellation is close to arbitrary, since distance-transform peaks don't
know where one grain ends and its neighbor begins.

This variant keeps everything about run_method_e identical -- same
specimen mask, same OTSU band, same morphological-reconstruction
preprocessing, same seed placement (still from distance-transform local
maxima) -- and changes only the watershed relief surface: instead of
flooding by -dist, it floods by the intensity gradient magnitude (the same
technique run_method_b already uses). Watershed then grows each seed until
it hits an actual intensity edge instead of a purely geometric ridge, so
boundaries should track real particle silhouettes rather than a roughly
uniform honeycomb.

Kept separate from methods_a_b_e.py so the original, verbatim-derived
Method E remains available for comparison -- nothing in Methods_ABE/ is
modified by this file.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label, maximum_filter, binary_dilation
from skimage.morphology import remove_small_objects, ball, opening
from skimage.segmentation import watershed
from skimage.filters import sobel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Methods_ABE'))
from methods_a_b_e import (OTSU_LOWER, OTSU_UPPER, MIN_DIAMETER_UM,        # noqa: E402
                            VOXEL_SIZE_UM, sample_mask, normalize_volume)  # noqa: E402


def run_method_e_v2(volume):
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

    gx = sobel(volume, axis=2)
    gy = sobel(volume, axis=1)
    gz = sobel(volume, axis=0)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)

    labeled = watershed(grad, markers, mask=reconstructed)
    min_vox_e = int((150.0 / VOXEL_SIZE_UM) ** 3 * (np.pi / 6))
    return remove_small_objects(labeled, min_size=min_vox_e)
