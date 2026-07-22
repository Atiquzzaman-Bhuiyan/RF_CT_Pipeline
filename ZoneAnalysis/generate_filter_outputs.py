"""Visualize all 11 feature channels for a given slice."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pipeline_common import (resolve_ct_path, load_tiff, extract_features,
                              normalize_with_range, compute_training_intensity_range,
                              FEATURE_NAMES)

OUT_DIR = Path(__file__).parent / 'Results' / 'FilterOutputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLICES = [150, 363, 750]


def main():
    print('Computing training-derived intensity range (from the 6 annotated slices)...')
    lo, hi = compute_training_intensity_range()
    print(f'  range: [{lo:.4f}, {hi:.4f}]')

    for slice_num in SLICES:
        print(f'\nSlice {slice_num}...')
        path = resolve_ct_path(slice_num)
        raw = load_tiff(path)
        norm = normalize_with_range(raw, lo, hi)
        feats = extract_features(norm).reshape(*raw.shape, 11)

        fig, axes = plt.subplots(3, 4, figsize=(18, 13))
        axes = axes.ravel()
        axes[0].imshow(raw, cmap='gray')
        axes[0].set_title('Original CT (raw)')
        axes[0].axis('off')
        for i, name in enumerate(FEATURE_NAMES):
            ax = axes[i + 1]
            im = ax.imshow(feats[..., i], cmap='viridis')
            ax.set_title(name)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046)
        for ax in axes[len(FEATURE_NAMES) + 1:]:
            ax.axis('off')
        fig.suptitle(f'Slice {slice_num:06d} — 11 feature channels', fontsize=14)
        fig.tight_layout()
        out_path = OUT_DIR / f'filter_outputs_slice_{slice_num:06d}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved {out_path}')


if __name__ == '__main__':
    main()
