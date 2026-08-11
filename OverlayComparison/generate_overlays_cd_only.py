"""
Methods C & D only, custom slice set -- 3-panel boundary figures
=====================================================================
Same 3-panel-per-method boundary style as generate_overlays.py (that
script's make_method_panel_figure), but scoped to just Methods C and D so
a one-off slice swap doesn't re-run A/B/E unnecessarily. Every output
filename carries the slice-set tag, so re-running with a different slice
list never overwrites a previous run's figures -- nothing existing is
touched or deleted.

Does NOT modify pipeline_common.py, pipeline_method_d.py, or any other
existing file -- only imports and calls them.

Usage:
    python3 generate_overlays_cd_only.py
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
sys.path.insert(0, str(PROJECT_ROOT / 'ZoneAnalysis'))

from pipeline_common import resolve_ct_path, load_tiff, predict_probability_volume, segment_watershed  # noqa: E402
from pipeline_method_d import segment_method_d, build_normalized_intensity_volume  # noqa: E402

ZONE_RESULTS = PROJECT_ROOT / 'ZoneAnalysis' / 'Results'
OUT_DIR = Path(__file__).resolve().parent / 'Results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SLICES = [110, 363, 800]
HALF_WINDOW = 25
N_SLICES = 981
SLICE_TAG = '-'.join(str(s) for s in TARGET_SLICES)

METHODS = [
    ('Method C', 'Random Forest', 'Method_C_RF', '#2CA25F'),
    ('Method D', 'Hybrid RF+WS', 'Method_D_Hybrid', '#E8A838'),
]


def normalize_per_slice(volume):
    out = np.empty_like(volume, dtype=np.float32)
    for i in range(volume.shape[0]):
        arr = volume[i].astype(np.float32)
        amin, amax = arr.min(), arr.max()
        out[i] = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr)
    return out


def load_window(start, end):
    return np.stack([load_tiff(resolve_ct_path(n)) for n in range(start, end)], axis=0)


def boundary_mask(labeled_slice):
    return find_boundaries(labeled_slice, mode='outer')


def make_method_panel_figure(label, desc, key, color, slice_nums, base_imgs, labels_by_slice):
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


def save_individual_image(slice_num, key, color, base_img, labeled_slice):
    overlay = np.stack([base_img, base_img, base_img], axis=-1)
    overlay[boundary_mask(labeled_slice)] = to_rgb(color)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path = OUT_DIR / f'{key}_slice{slice_num:04d}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'  saved {out_path}')


def main():
    print('Loading RF model + intensity range...')
    rf = joblib.load(ZONE_RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(ZONE_RESULTS / 'combined_intensity_range.joblib')

    base_imgs = {}
    labels_by_slice = {}

    for slice_num in TARGET_SLICES:
        start = max(0, slice_num - HALF_WINDOW)
        end = min(N_SLICES, slice_num + HALF_WINDOW + 1)
        target_idx = slice_num - start
        print(f'\n=== Slice {slice_num} (window {start}-{end - 1}) ===')

        raw_window = load_window(start, end)
        base_imgs[slice_num] = normalize_per_slice(raw_window)[target_idx]

        prob_volume = predict_probability_volume(raw_window, rf, intensity_range)
        labeled_c = segment_watershed(prob_volume)

        intensity_volume = build_normalized_intensity_volume(raw_window, intensity_range)
        labeled_d, _ = segment_method_d(prob_volume, intensity_volume)

        labels_by_slice[slice_num] = {
            'Method_C_RF': labeled_c[target_idx],
            'Method_D_Hybrid': labeled_d[target_idx],
        }

    print('\nBuilding individual images...')
    for slice_num in TARGET_SLICES:
        for label, desc, key, color in METHODS:
            save_individual_image(slice_num, key, color, base_imgs[slice_num], labels_by_slice[slice_num][key])

    print('\nDone.')


if __name__ == '__main__':
    main()
