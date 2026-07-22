"""Generate histogram + overlay figures for Method D, matching Method C's
figure style. Re-runs the (fast, ~35-45s each) zone segmentation to get a
labeled volume for the overlay image; full-volume histogram is built directly
from the already-saved diameters CSV (no need to redo the ~28min chunked run).
Does not touch any Method C file or output."""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.segmentation import find_boundaries
from pipeline_common import resolve_ct_path, load_tiff, predict_probability_volume
from pipeline_method_d import segment_method_d, build_normalized_intensity_volume

BASE = Path(__file__).parent
RESULTS = BASE / 'Results'
OUT_DIR = RESULTS / 'MethodD'

ZONES = [('early', 100, 200), ('mid', 350, 450), ('late', 750, 850)]


def load_zone_volume(start, end):
    return np.stack([load_tiff(resolve_ct_path(n)) for n in range(start, end)], axis=0)


def save_histogram(diams, name, out_dir, label='Method D'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diams, bins=40, color='indianred', edgecolor='white')
    ax.axvline(diams.mean(), color='navy', linestyle='--', label=f'mean={diams.mean():.1f}µm')
    ax.set_xlabel('Diameter (µm)')
    ax.set_ylabel('Particle count')
    ax.set_title(f'{name} — {label} particle size distribution (n={len(diams)})')
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f'{name}_histogram_methodD.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def save_overlay_figure(volume, labeled, prob_volume, zone_name, out_dir, rep_offset=50):
    rep = min(rep_offset, volume.shape[0] - 1)
    image = volume[rep]
    lbl_slice = labeled[rep]
    prob_slice = prob_volume[rep]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original CT')
    axes[0].axis('off')

    axes[1].imshow(prob_slice, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('RF Probability Map (reused from Method C)')
    axes[1].axis('off')

    rng = np.random.RandomState(0)
    n_labels = int(lbl_slice.max()) + 1
    cmap_colors = rng.rand(max(n_labels, 1), 3)
    cmap_colors[0] = 0
    colored = cmap_colors[lbl_slice]
    axes[2].imshow(colored)
    axes[2].set_title(f'Method D Segmented Particles ({len(np.unique(lbl_slice)) - 1} in this slice)')
    axes[2].axis('off')

    boundaries = find_boundaries(lbl_slice, mode='outer')
    overlay_img = np.stack([image, image, image], axis=-1).astype(np.float32)
    if overlay_img.max() > 1:
        overlay_img = overlay_img / overlay_img.max()
    overlay_img[boundaries] = [1, 0, 0]
    axes[3].imshow(overlay_img)
    axes[3].set_title('Overlay (red = particle boundary)')
    axes[3].axis('off')

    fig.suptitle(f'{zone_name} zone — Method D — representative slice (offset {rep} within zone)', fontsize=14)
    fig.tight_layout()
    out_path = out_dir / f'{zone_name}_overlay_methodD.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def main():
    rf = joblib.load(RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(RESULTS / 'combined_intensity_range.joblib')

    for name, start, end in ZONES:
        print(f'\n=== {name} zone figures ===')
        volume = load_zone_volume(start, end)
        prob_volume = predict_probability_volume(volume, rf, intensity_range)
        intensity_volume = build_normalized_intensity_volume(volume, intensity_range)
        labeled, n_seeds = segment_method_d(prob_volume, intensity_volume)

        diams = pd.read_csv(OUT_DIR / f'{name}_diameters_methodD.csv')['diameter_um'].values
        save_histogram(diams, name, OUT_DIR)
        save_overlay_figure(volume, labeled, prob_volume, name, OUT_DIR)

    print('\n=== full volume histogram (from existing diameters CSV, no recompute) ===')
    full_diams = pd.read_csv(OUT_DIR / 'full_volume_diameters_methodD.csv')['diameter_um'].values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(full_diams, bins=50, color='indianred', edgecolor='white')
    ax.axvline(full_diams.mean(), color='navy', linestyle='--', label=f'mean={full_diams.mean():.1f}µm')
    ax.set_xlabel('Diameter (µm)')
    ax.set_ylabel('Particle count')
    ax.set_title(f'Full volume — Method D particle size distribution (n={len(full_diams)})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'full_volume_histogram_methodD.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {OUT_DIR / "full_volume_histogram_methodD.png"}')


if __name__ == '__main__':
    main()
