"""
Compares original Method E (distance-transform relief) against Method E v2
(gradient relief) on slices 50/350/800, side by side. Does not modify
either implementation -- read-only comparison.

Usage:
    python3 compare_e_v2.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from skimage.segmentation import find_boundaries

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'Methods_ABE'))
sys.path.insert(0, str(PROJECT_ROOT / 'ZoneAnalysis'))

from methods_a_b_e import run_method_e, normalize_volume         # noqa: E402
from pipeline_common import resolve_ct_path, load_tiff           # noqa: E402
from method_e_v2 import run_method_e_v2                          # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / 'Results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SLICES = [110, 350, 800]
HALF_WINDOW = 25
N_SLICES = 981
COLOR = '#9B72CF'
SLICE_TAG = '-'.join(str(s) for s in TARGET_SLICES)


def load_window(start, end):
    return np.stack([load_tiff(resolve_ct_path(n)) for n in range(start, end)], axis=0)


def overlay(base_img, labeled_slice):
    b = find_boundaries(labeled_slice, mode='outer')
    rgb = np.stack([base_img] * 3, axis=-1)
    rgb[b] = to_rgb(COLOR)
    return rgb


def main():
    fig, axes = plt.subplots(2, len(TARGET_SLICES), figsize=(6 * len(TARGET_SLICES), 12))

    for col, slice_num in enumerate(TARGET_SLICES):
        start = max(0, slice_num - HALF_WINDOW)
        end = min(N_SLICES, slice_num + HALF_WINDOW + 1)
        target_idx = slice_num - start
        print(f'Slice {slice_num} (window {start}-{end - 1})...')

        raw_window = load_window(start, end)
        vol = normalize_volume(raw_window)

        labeled_old = run_method_e(vol)
        labeled_new = run_method_e_v2(vol)

        base_img = vol[target_idx]
        axes[0, col].imshow(overlay(base_img, labeled_old[target_idx]))
        axes[0, col].set_title(f'slice {slice_num} - original (dist relief)')
        axes[0, col].axis('off')

        axes[1, col].imshow(overlay(base_img, labeled_new[target_idx]))
        axes[1, col].set_title(f'slice {slice_num} - v2 (gradient relief)')
        axes[1, col].axis('off')

    fig.suptitle('Method E: original vs. gradient-relief variant', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = OUT_DIR / f'method_e_v1_vs_v2_{SLICE_TAG}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
