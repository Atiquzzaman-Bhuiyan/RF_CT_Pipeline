"""
Method D — Hybrid RF-Intensity Watershed.

Reuses Method C's trained RF model, probability volume, and diameter
computation, but replaces Method C's probability-gradient-only watershed with
a hybrid energy surface combining RF-probability gradient and raw-intensity
gradient, seeded from intensity local maxima instead of probability local
maxima.

This file does NOT modify pipeline_common.py or any other Method C script --
everything here is additive. It only imports existing, unmodified Method C
functions/constants for reuse: normalize_with_range, VOXEL_UM, MIN_VOXELS,
SEED_WINDOW_SIZE. predict_probability_volume, compute_diameters,
compute_diameters_central, resolve_ct_path etc. are called from the caller
scripts (zone_analysis_method_d.py / full_volume_analysis_method_d.py) exactly
as Method C already does -- unchanged.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label as nd_label
from skimage.filters import sobel, threshold_otsu
from skimage.morphology import opening, closing, ball, remove_small_objects
from skimage.segmentation import watershed

from pipeline_common import normalize_with_range, MIN_VOXELS, SEED_WINDOW_SIZE

# ---- Method D parameters (new, independent of Method C's constants) ----
OTSU_CLIP_MIN = 0.35
OTSU_CLIP_MAX = 0.65
SEED_INTENSITY_SIGMA = 2
SEED_INTENSITY_THRESHOLD = 0.45
MAX_SEEDS = 15000
ENERGY_SIGMA = 1
W_PROB = 0.7
W_INTENSITY = 0.3


def build_normalized_intensity_volume(volume: np.ndarray, intensity_range) -> np.ndarray:
    """Same normalization Method C's RF uses for feature extraction (train-only
    range) -- keeps intensity and probability on a comparable scale, which
    matters since the same fixed thresholds (e.g. 0.45) are applied to both."""
    lo, hi = intensity_range
    return np.stack([normalize_with_range(volume[i], lo, hi) for i in range(volume.shape[0])], axis=0)


def generate_binary_mask_otsu(prob_volume: np.ndarray, otsu_min: float = OTSU_CLIP_MIN,
                                otsu_max: float = OTSU_CLIP_MAX) -> np.ndarray:
    """Per-slice adaptive Otsu threshold on P(x), clipped to [otsu_min, otsu_max],
    followed by 3D morphological opening then closing with ball(1)."""
    binary = np.zeros(prob_volume.shape, dtype=bool)
    for i in range(prob_volume.shape[0]):
        slice_p = prob_volume[i]
        if slice_p.max() == slice_p.min():
            t = 0.5 * (otsu_min + otsu_max)  # degenerate/constant slice -- safe fallback
        else:
            t = float(np.clip(threshold_otsu(slice_p), otsu_min, otsu_max))
        binary[i] = slice_p > t
    struct = ball(1)
    binary = opening(binary, struct)
    binary = closing(binary, struct)
    return binary


def place_intensity_seeds(intensity_volume: np.ndarray, binary_mask: np.ndarray,
                            sigma: int = SEED_INTENSITY_SIGMA, threshold: float = SEED_INTENSITY_THRESHOLD,
                            max_seeds: int = MAX_SEEDS, window_size: int = SEED_WINDOW_SIZE):
    """Local maxima of Gaussian-smoothed intensity within the binary mask.
    If the fixed 0.45 threshold yields more than max_seeds peaks, adaptively
    raise the threshold to the value of the max_seeds-th highest peak
    (percentile fallback) instead of the fixed cutoff, to bound seed count."""
    smoothed = gaussian_filter(intensity_volume, sigma=sigma)
    masked = smoothed * binary_mask
    img_max = maximum_filter(masked, size=window_size)
    peak_mask = (masked == img_max) & (masked > threshold) & binary_mask

    n_seeds = int(peak_mask.sum())
    if n_seeds > max_seeds:
        peak_values = masked[peak_mask]
        adaptive_threshold = np.partition(peak_values, -max_seeds)[-max_seeds]
        peak_mask = (masked == img_max) & (masked >= adaptive_threshold) & binary_mask
        n_seeds = int(peak_mask.sum())

    markers, _ = nd_label(peak_mask)
    return markers, n_seeds


def compute_hybrid_energy(prob_volume: np.ndarray, intensity_volume: np.ndarray,
                            sigma: int = ENERGY_SIGMA, w_prob: float = W_PROB,
                            w_intensity: float = W_INTENSITY) -> np.ndarray:
    """E(x) = w_prob * |grad(P_sigma)| + w_intensity * |grad(I_sigma)|, both
    gradients computed as 3D Sobel magnitude after sigma=1 Gaussian smoothing."""
    def grad_mag_3d(vol):
        gx = sobel(vol, axis=2)
        gy = sobel(vol, axis=1)
        gz = sobel(vol, axis=0)
        return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    smoothed_prob = gaussian_filter(prob_volume, sigma=sigma)
    smoothed_intensity = gaussian_filter(intensity_volume, sigma=sigma)
    return w_prob * grad_mag_3d(smoothed_prob) + w_intensity * grad_mag_3d(smoothed_intensity)


def segment_method_d(prob_volume: np.ndarray, intensity_volume: np.ndarray):
    """Full Method D pipeline: Otsu binary mask -> intensity-maxima seeds ->
    hybrid energy surface -> watershed -> MIN_VOXELS size filter (same
    MIN_VOXELS/100um cutoff as Method C, reused unchanged).
    Returns (labeled_volume, n_seeds)."""
    binary_mask = generate_binary_mask_otsu(prob_volume)
    markers, n_seeds = place_intensity_seeds(intensity_volume, binary_mask)
    energy = compute_hybrid_energy(prob_volume, intensity_volume)
    labeled = watershed(energy, markers, mask=binary_mask)
    labeled = remove_small_objects(labeled, min_size=MIN_VOXELS)
    return labeled, n_seeds
