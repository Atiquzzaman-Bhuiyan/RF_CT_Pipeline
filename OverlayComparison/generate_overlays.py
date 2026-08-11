"""
Overlay Comparison - Methods A-E
==================================
Builds particle-boundary overlay images for slices 50, 350, 800, one
color per method (color code as supplied):

    Method A  Otsu+Dist WS    #4878CF
    Method B  Intensity WS    #E8836E
    Method C  Random Forest   #2CA25F
    Method D  Hybrid RF+WS    #E8A838
    Method E  Morpho Recon    #9B72CF

For each target slice, a small 3D window centered on it is segmented
(watershed needs neighboring slices for context), then only the target
slice is extracted and its particle boundaries are drawn on top of the
raw CT image.

Two figure types are produced per method/slice:
  - one grid figure per slice (all 5 methods side by side, for spotting
    where methods disagree)
  - one 3-panel figure per method (that method's boundary across slices
    150/350/800 in one image), matching the reference layout

Does NOT modify any existing code - only imports and calls:
  - Methods A, B, E: Methods_ABE/methods_a_b_e.py (verbatim extraction
    of full_volume_run_v2.py)
  - Methods C, D:    ZoneAnalysis/pipeline_common.py + pipeline_method_d.py,
                      using the already-trained ZoneAnalysis/Results/
                      combined_model.joblib / combined_intensity_range.joblib

Boundaries are drawn with a 1px dilation (vs. the previous raw 1px
find_boundaries line) so they read cleanly against the grainy CT
texture instead of disappearing into it.

Usage:
    python3 generate_overlays.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from skimage.segmentation import find_boundaries

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'Methods_ABE'))
sys.path.insert(0, str(PROJECT_ROOT / 'ZoneAnalysis'))

from methods_a_b_e import run_method_a, run_method_b, run_method_e, normalize_volume  # noqa: E402
from pipeline_common import resolve_ct_path, load_tiff, predict_probability_volume, segment_watershed  # noqa: E402
from pipeline_method_d import segment_method_d, build_normalized_intensity_volume  # noqa: E402

RESULTS_DIR = ZONE_RESULTS = PROJECT_ROOT / 'ZoneAnalysis' / 'Results'
OUT_DIR = Path(__file__).resolve().parent / 'Results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SLICES = [110, 350, 800]
HALF_WINDOW = 25   # window = target +/- 25 slices, target sits in the middle
N_SLICES = 981     # slices 0..980 on disk

# Every output filename below includes this tag so re-running with a
# different TARGET_SLICES never overwrites a previous run's figures.
SLICE_TAG = '-'.join(str(s) for s in TARGET_SLICES)

METHODS = [
    ('Method A', 'Otsu+Dist WS',  'Method_A_Otsu',        '#4878CF'),
    ('Method B', 'Intensity WS',  'Method_B_IntWS',        '#E8836E'),
    ('Method C', 'Random Forest', 'Method_C_RF',           '#2CA25F'),
    ('Method D', 'Hybrid RF+WS',  'Method_D_Hybrid',       '#E8A838'),
    ('Method E', 'Morpho Recon',  'Method_E_MorphoRecon',  '#9B72CF'),
]


def load_window(start, end):
    """Raw (unnormalized) CT slices, one array per slice."""
    return np.stack([load_tiff(resolve_ct_path(n)) for n in range(start, end)], axis=0)


def boundary_mask(labeled_slice):
    """Raw 1px boundary, no dilation. Dilating looked fine for sparse
    segmentations (C, D) but turned into a thick, obscuring smear on dense
    ones (A, E) where boundaries are packed close together -- undilated
    lines track the actual particle silhouettes far more legibly there."""
    return find_boundaries(labeled_slice, mode='outer')


def segment_window(raw_window, rf, intensity_range):
    """Run all 5 methods on one 3D window, return dict of labeled volumes."""
    vol_abe = normalize_volume(raw_window)

    labeled = {}
    labeled['Method_A_Otsu'] = run_method_a(vol_abe)
    labeled['Method_B_IntWS'] = run_method_b(vol_abe)
    labeled['Method_E_MorphoRecon'] = run_method_e(vol_abe)

    prob_volume = predict_probability_volume(raw_window, rf, intensity_range)
    labeled['Method_C_RF'] = segment_watershed(prob_volume)

    intensity_volume = build_normalized_intensity_volume(raw_window, intensity_range)
    labeled['Method_D_Hybrid'], _ = segment_method_d(prob_volume, intensity_volume)

    return labeled, vol_abe


def make_grid_figure(slice_num, base_img, labels_by_key):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    axes[0].imshow(base_img, cmap='gray')
    axes[0].set_title(f'Original CT - slice {slice_num}')
    axes[0].axis('off')

    for ax, (label, desc, key, color) in zip(axes[1:], METHODS):
        overlay = np.stack([base_img, base_img, base_img], axis=-1)
        overlay[boundary_mask(labels_by_key[key])] = to_rgb(color)
        ax.imshow(overlay)
        ax.set_title(f'{label} - {desc}')
        ax.axis('off')

    fig.suptitle(f'Method comparison - slice {slice_num}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = OUT_DIR / f'slice_{slice_num:04d}_grid_{SLICE_TAG}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def make_method_panel_figure(label, desc, key, color, slice_nums, base_imgs, labels_by_slice):
    """One method, one panel per slice - same layout as the reference image.
    Thin boundary lines only - reads well for sparse segmentations (C, D)
    but gets illegible for heavily oversegmented ones (A, E); see
    make_filled_panel_figure for those."""
    fig, axes = plt.subplots(1, len(slice_nums), figsize=(6 * len(slice_nums), 6))
    if len(slice_nums) == 1:
        axes = [axes]

    for ax, slice_num in zip(axes, slice_nums):
        base_img = base_imgs[slice_num]
        overlay = np.stack([base_img, base_img, base_img], axis=-1)
        overlay[boundary_mask(labels_by_slice[slice_num][key])] = to_rgb(color)
        ax.imshow(overlay)
        ax.set_title(f'slice {slice_num}')
        ax.axis('off')

    fig.suptitle(f'{label} - {desc}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = OUT_DIR / f'{key}_3panel_{SLICE_TAG}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def make_filled_panel_figure(label, desc, key, slice_nums, base_imgs, labels_by_slice, alpha=0.55, seed=7):
    """One method, one panel per slice, particles filled with a random color
    per label (+ a thin dark outline). Much more legible than boundary-only
    lines when a method oversegments the whole matrix (Methods A/E), since
    the eye reads distinct color blobs far faster than overlapping outlines."""
    fig, axes = plt.subplots(1, len(slice_nums), figsize=(6 * len(slice_nums), 6))
    if len(slice_nums) == 1:
        axes = [axes]

    for ax, slice_num in zip(axes, slice_nums):
        base_img = base_imgs[slice_num]
        lbl = labels_by_slice[slice_num][key]
        base_rgb = np.stack([base_img, base_img, base_img], axis=-1)

        rng = np.random.RandomState(seed)
        n_labels = int(lbl.max()) + 1
        palette = rng.rand(max(n_labels, 1), 3)
        palette[0] = 0
        color_img = palette[lbl]

        overlay = base_rgb.copy()
        mask = lbl > 0
        overlay[mask] = (1 - alpha) * base_rgb[mask] + alpha * color_img[mask]
        overlay[boundary_mask(lbl)] = 0.0

        ax.imshow(overlay)
        ax.set_title(f'slice {slice_num}')
        ax.axis('off')

    fig.suptitle(f'{label} - {desc} (filled)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = OUT_DIR / f'{key}_3panel_filled_{SLICE_TAG}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def main():
    print('Loading RF model + intensity range (Methods C/D)...')
    rf = joblib.load(ZONE_RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(ZONE_RESULTS / 'combined_intensity_range.joblib')

    base_imgs = {}
    labels_by_slice = {}

    for slice_num in TARGET_SLICES:
        start = max(0, slice_num - HALF_WINDOW)
        end = min(N_SLICES, slice_num + HALF_WINDOW + 1)
        target_idx = slice_num - start
        print(f'\n=== Slice {slice_num} (window {start}-{end - 1}, target at offset {target_idx}) ===')

        raw_window = load_window(start, end)
        labeled, vol_abe = segment_window(raw_window, rf, intensity_range)

        base_imgs[slice_num] = vol_abe[target_idx]
        labels_by_slice[slice_num] = {key: labeled[key][target_idx] for _, _, key, _ in METHODS}

        make_grid_figure(slice_num, base_imgs[slice_num], labels_by_slice[slice_num])

    print('\nBuilding per-method 3-panel figures (boundary style)...')
    for label, desc, key, color in METHODS:
        make_method_panel_figure(label, desc, key, color, TARGET_SLICES, base_imgs, labels_by_slice)

    print('\nBuilding per-method 3-panel figures (filled style)...')
    for label, desc, key, color in METHODS:
        make_filled_panel_figure(label, desc, key, TARGET_SLICES, base_imgs, labels_by_slice)

    print('\nDone.')


if __name__ == '__main__':
    main()
