"""
Method A v2 â intensity-seeded + gradient-relief variant (new, additive only)
=================================================================================
The original run_method_a (Methods_ABE/methods_a_b_e.py, untouched by this
file) has the same structure that made the original Method E tessellate
the whole matrix instead of tracking real particles: it seeds watershed
from local maxima of the binary mask's distance transform, then floods
using -dist as the relief surface. Both the seed placement and the relief
surface are purely geometric, so on fine/dense matrix regions the
resulting cell walls form at a roughly fixed spacing regardless of where
real particle edges are.

Same fix that worked for Method E (method_e_v3.py): seed from local
maxima of *smoothed intensity* within the binary mask (real particle
cores) instead of distance-transform peaks, and flood on *intensity
gradient magnitude* instead of -distance, so boundaries grow from real
particle centers and stop at real intensity edges.

Kept separate from methods_a_b_e.py so the original, verbatim-derived
Method A remains available for comparison -- nothing in it is modified by
this file.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label, maximum_filter, gaussian_filter
from skimage.morphology import remove_small_objects, ball, opening, closing
from skimage.segmentation import watershed
from skimage.filters import sobel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Methods_ABE'))
from methods_a_b_e import (OTSU_LOWER, OTSU_UPPER, MIN_VOXELS,            # noqa: E402
                            sample_mask, normalize_volume)                # noqa: E402


def run_method_a_v2(volume):
    binary = (volume > OTSU_LOWER) & (volume < OTSU_UPPER) & sample_mask(volume)
    binary = opening(binary, ball(1))
    binary = closing(binary, ball(1))

    smoothed = gaussian_filter(volume, sigma=2)
    smooth_max = maximum_filter(smoothed * binary, size=15)
    peak_mask = ((smoothed * binary == smooth_max) &
                 (smoothed > OTSU_LOWER + 0.05) & binary)
    markers, n_seeds = label(peak_mask)
    if n_seeds < 10:
        peak_mask = ((smoothed * binary == smooth_max) &
                     (smoothed > OTSU_LOWER) & binary)
        markers, _ = label(peak_mask)

    gx = sobel(volume, axis=2)
    gy = sobel(volume, axis=1)
    gz = sobel(volume, axis=0)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)

    labeled = watershed(grad, markers, mask=binary)
    return remove_small_objects(labeled, min_size=MIN_VOXELS)
