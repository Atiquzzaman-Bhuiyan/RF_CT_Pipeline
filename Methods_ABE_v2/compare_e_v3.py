"""
Compares all three Method E variants -- original (distance relief),
v2 (gradient relief, same distance-transform seeds), and v3 (gradient
relief + intensity-based seeds) -- on slices 110/350/800, side by side.
Does not modify any of the three implementations -- read-only comparison.

Usage:
    python3 compare_e_v3.py
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
from method_e_v3 import run_method_e_v3                          # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / 'Results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SLICES = [110, 350, 800]
HALF_WINDOW = 25
N_SLICES = 981
COLOR = '#9B72CF'
SLICE_TAG = '-'.join(str(s) for s in TARGET_SLICES)

VARIANTS = [
    ('original (dist relief, dist seeds)', run_method_e),
    ('v2 (gradient relief, dist seeds)', run_method_e_v2),
    ('v3 (gradient relief, intensity seeds)', run_method_e_v3),
]


def load_window(start, end):
    return np.stack([load_tiff(resolve_ct_path(n)) for n in range(start, end)], axis=0)


def overlay(base_img, labeled_slice):
    b = find_boundaries(labeled_slice, mode='outer')
    rgb = np.stack([base_img] * 3, axis=-1)
    rgb[b] = to_rgb(COLOR)
    return rgb


def main():
    fig, axes = plt.subplots(len(VARIANTS), len(TARGET_SLICES),
                              figsize=(6 * len(TARGET_SLICES), 6 * len(VARIANTS)))

    for col, slice_num in enumerate(TARGET_SLICES):
        start = max(0, slice_num - HALF_WINDOW)
        end = min(N_SLICES, slice_num + HALF_WINDOW + 1)
        target_idx = slice_num - start
        print(f'Slice {slice_num} (window {start}-{end - 1})...')

        raw_window = load_window(start, end)
        vol = normalize_volume(raw_window)
        base_img = vol[target_idx]

        for row, (label, fn) in enumerate(VARIANTS):
            labeled = fn(vol)
            axes[row, col].imshow(overlay(base_img, labeled[target_idx]))
            axes[row, col].set_title(f'slice {slice_num} - {label}', fontsize=10)
            axes[row, col].axis('off')

    fig.suptitle('Method E: original vs. v2 vs. v3', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = OUT_DIR / f'method_e_v1_vs_v2_vs_v3_{SLICE_TAG}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
