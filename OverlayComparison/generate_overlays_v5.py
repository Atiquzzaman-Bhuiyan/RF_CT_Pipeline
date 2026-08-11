"""
Overlay Comparison v5 - Methods C & D, three views per zone
===============================================================
Adapted from a draft script that trained its own throwaway RF on a single
slice with the old 8-feature layout (one channel a never-computed zero
placeholder) and pointed at paths that don't exist on this machine
(~/lhs1/...). Updated to:

  - use the real CT data location (via ZoneAnalysis/pipeline_common.py's
    resolve_ct_path) and the project's already-trained PRODUCTION model
    (ZoneAnalysis/Results/combined_model.joblib -- 11-feature RF trained on
    all 6 annotated slices) instead of retraining a weaker single-slice
    8-feature model. pipeline_common.py's own docstring flags the
    8-feature/single-slice approach as the thing that got replaced.
  - keep the three-overlay-per-method idea (boundary / lowthresh / probmap)
  - representative slices per zone: 150 (early), 350 (mid), 800 (late) --
    same three slices already used elsewhere in OverlayComparison/

Does not modify pipeline_common.py, pipeline_method_d.py, or any other
existing file -- only imports and calls them, same as generate_overlays.py.
The "lowthresh" variant for Method D is a genuine change (a lower Otsu
clip -> more permissive mask); the draft script's D-lowthresh re-ran the
exact same inputs and always produced an identical result to D-boundary.

Usage:
    python3 generate_overlays_v5.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter, label as nd_label
from skimage.filters import sobel
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import remove_small_objects

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'ZoneAnalysis'))

from pipeline_common import (resolve_ct_path, load_tiff, predict_probability_volume,
                              PROB_SMOOTH_SIGMA, SEED_WINDOW_SIZE, SEED_PROB_THRESHOLD,
                              MIN_VOXELS)                                               # noqa: E402
from pipeline_method_d import (segment_method_d, build_normalized_intensity_volume,
                                generate_binary_mask_otsu, place_intensity_seeds,
                                compute_hybrid_energy)                                  # noqa: E402

ZONE_RESULTS = PROJECT_ROOT / 'ZoneAnalysis' / 'Results'
OUT_DIR = Path(__file__).resolve().parent / 'Results' / 'PaperFigures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# name -> (start, end, rep_offset); rep_offset picks out slices 150/350/800
ZONES = {
    'early': (100, 200, 50),
    'mid':   (350, 450, 0),
    'late':  (750, 850, 50),
}

METHOD_COLORS = {
    'Method_C': [0.15, 0.85, 0.15],
    'Method_D': [1.0,  0.75, 0.0],
}


def normalize_per_slice(volume):
    """Per-slice min-max to 0-1, for display only."""
    out = np.empty_like(volume, dtype=np.float32)
    for i in range(volume.shape[0]):
        arr = volume[i].astype(np.float32)
        amin, amax = arr.min(), arr.max()
        out[i] = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr)
    return out


def load_zone_volume(start, end):
    return np.stack([load_tiff(resolve_ct_path(n)) for n in range(start, end)], axis=0)


def segment_method_c(prob_volume, threshold=0.5):
    """Same algorithm as pipeline_common.segment_watershed, with the binary
    mask threshold exposed so we can render a lower-threshold variant."""
    binary_rf = prob_volume > threshold
    smoothed_prob = gaussian_filter(prob_volume, sigma=PROB_SMOOTH_SIGMA)
    gx = sobel(smoothed_prob, axis=2)
    gy = sobel(smoothed_prob, axis=1)
    gz = sobel(smoothed_prob, axis=0)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)
    masked = smoothed_prob * binary_rf
    img_max = maximum_filter(masked, size=SEED_WINDOW_SIZE)
    peak_mask = (masked == img_max) & (masked > SEED_PROB_THRESHOLD)
    markers, _ = nd_label(peak_mask)
    labeled = watershed(grad, markers, mask=binary_rf)
    return remove_small_objects(labeled, min_size=MIN_VOXELS)


def segment_method_d_variant(prob_volume, intensity_volume, otsu_min):
    """Same as pipeline_method_d.segment_method_d, with the Otsu lower clip
    exposed for a more permissive 'lowthresh' mask variant."""
    binary_mask = generate_binary_mask_otsu(prob_volume, otsu_min=otsu_min)
    markers, _ = place_intensity_seeds(intensity_volume, binary_mask)
    energy = compute_hybrid_energy(prob_volume, intensity_volume)
    labeled = watershed(energy, markers, mask=binary_mask)
    return remove_small_objects(labeled, min_size=MIN_VOXELS)


def save_boundary_overlay(ct_slice, labeled_slice, zone_name, method_name, suffix):
    boundaries = find_boundaries(labeled_slice, mode='outer')
    color = METHOD_COLORS[method_name]
    rgb = np.stack([ct_slice] * 3, axis=-1)
    rgb[boundaries] = color
    rgb = np.clip(rgb, 0, 1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fname = f"{method_name.lower()}_{zone_name}_{suffix}.png"
    plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  saved {fname}")


def save_probmap_overlay(ct_slice, prob_slice, zone_name, method_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].axis('off')
    axes[1].imshow(ct_slice, cmap='gray')
    cmap = 'Greens' if method_name == 'Method_C' else 'YlOrBr'
    axes[1].imshow(prob_slice, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    axes[1].axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fname = f"{method_name.lower()}_{zone_name}_probmap.png"
    plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  saved {fname}")


def main():
    print('Loading RF model + intensity range (production combined model)...')
    rf = joblib.load(ZONE_RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(ZONE_RESULTS / 'combined_intensity_range.joblib')

    for zone_name, (start, end, rep_offset) in ZONES.items():
        rep_slice_num = start + rep_offset
        print(f"\n=== Zone {zone_name} (slices {start}-{end - 1}, representative slice {rep_slice_num}) ===")

        raw_volume = load_zone_volume(start, end)
        vol_display = normalize_per_slice(raw_volume)
        ct_slice = vol_display[rep_offset]

        print("  Predicting RF probability volume...")
        prob_volume = predict_probability_volume(raw_volume, rf, intensity_range)
        prob_slice = prob_volume[rep_offset]

        print("  Method C...")
        labeled_c = segment_method_c(prob_volume, threshold=0.5)
        save_boundary_overlay(ct_slice, labeled_c[rep_offset], zone_name, 'Method_C', 'boundary')

        labeled_c_low = segment_method_c(prob_volume, threshold=0.3)
        save_boundary_overlay(ct_slice, labeled_c_low[rep_offset], zone_name, 'Method_C', 'lowthresh')

        save_probmap_overlay(ct_slice, prob_slice, zone_name, 'Method_C')

        print("  Method D...")
        intensity_volume = build_normalized_intensity_volume(raw_volume, intensity_range)
        labeled_d, _ = segment_method_d(prob_volume, intensity_volume)
        save_boundary_overlay(ct_slice, labeled_d[rep_offset], zone_name, 'Method_D', 'boundary')

        labeled_d_low = segment_method_d_variant(prob_volume, intensity_volume, otsu_min=0.20)
        save_boundary_overlay(ct_slice, labeled_d_low[rep_offset], zone_name, 'Method_D', 'lowthresh')

        save_probmap_overlay(ct_slice, prob_slice, zone_name, 'Method_D')

        del raw_volume, vol_display, prob_volume, intensity_volume
        del labeled_c, labeled_c_low, labeled_d, labeled_d_low

    print(f"\nAll overlays saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
