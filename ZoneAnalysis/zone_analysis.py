"""Zone-level particle segmentation: early (100-200), mid (350-450), late (750-850).
Each zone is processed as a standalone 100-slice volume (no chunking needed --
fits comfortably in memory), using the single combined RF model."""
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from pipeline_common import (resolve_ct_path, load_tiff, predict_probability_volume,
                              segment_watershed, compute_diameters)

BASE = Path(__file__).parent
RESULTS = BASE / 'Results'

ZONES = [
    ('early', 100, 200, RESULTS / 'ZoneEarly'),
    ('mid', 350, 450, RESULTS / 'ZoneMid'),
    ('late', 750, 850, RESULTS / 'ZoneLate'),
]


def load_zone_volume(start, end):
    slices = []
    for n in range(start, end):
        try:
            slices.append(load_tiff(resolve_ct_path(n)))
        except FileNotFoundError:
            print(f'  WARNING: slice {n} missing, skipping')
    return np.stack(slices, axis=0)


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
    axes[1].set_title('RF Probability Map')
    axes[1].axis('off')

    rng = np.random.RandomState(0)
    n_labels = int(lbl_slice.max()) + 1
    cmap_colors = rng.rand(max(n_labels, 1), 3)
    cmap_colors[0] = 0  # background stays black
    colored = cmap_colors[lbl_slice]
    axes[2].imshow(colored)
    axes[2].set_title(f'Segmented Particles ({len(np.unique(lbl_slice)) - 1} in this slice)')
    axes[2].axis('off')

    axes[3].imshow(image, cmap='gray')
    axes[3].imshow(np.ma.masked_where(lbl_slice == 0, colored.sum(axis=-1) * 0 + 1),
                    cmap='autumn', alpha=0.45)
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(lbl_slice, mode='outer')
    overlay_img = np.stack([image, image, image], axis=-1) if image.ndim == 2 else image
    overlay_img = overlay_img.astype(np.float32)
    if overlay_img.max() > 1:
        overlay_img = overlay_img / overlay_img.max()
    overlay_img[boundaries] = [1, 0, 0]
    axes[3].clear()
    axes[3].imshow(overlay_img)
    axes[3].set_title('Overlay (red = particle boundary)')
    axes[3].axis('off')

    fig.suptitle(f'{zone_name} zone — representative slice (offset {rep} within zone)', fontsize=14)
    fig.tight_layout()
    out_path = out_dir / f'{zone_name}_overlay.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def save_histogram(diams, zone_name, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diams, bins=40, color='steelblue', edgecolor='white')
    ax.axvline(diams.mean(), color='red', linestyle='--', label=f'mean={diams.mean():.1f}µm')
    ax.set_xlabel('Diameter (µm)')
    ax.set_ylabel('Particle count')
    ax.set_title(f'{zone_name} zone — particle size distribution (n={len(diams)})')
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f'{zone_name}_histogram.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def process_zone(name, start, end, out_dir, rf, intensity_range):
    print(f'\n=== Zone: {name} (slices {start}-{end - 1}) ===')
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading volume...')
    volume = load_zone_volume(start, end)
    print(f'  volume shape: {volume.shape}')

    print('Predicting RF probability volume...')
    prob_volume = predict_probability_volume(volume, rf, intensity_range)

    print('Running watershed segmentation...')
    labeled = segment_watershed(prob_volume)

    print('Computing diameters...')
    diams = compute_diameters(labeled)

    pd.DataFrame({'diameter_um': diams}).to_csv(out_dir / f'{name}_diameters.csv', index=False)

    save_histogram(diams, name, out_dir)
    save_overlay_figure(volume, labeled, prob_volume, name, out_dir)

    elapsed = time.time() - t0
    stats = {
        'zone': name, 'slice_range': f'{start}-{end - 1}', 'n_particles': len(diams),
        'mean_diameter_um': float(diams.mean()) if len(diams) else float('nan'),
        'std_diameter_um': float(diams.std()) if len(diams) else float('nan'),
        'median_diameter_um': float(np.median(diams)) if len(diams) else float('nan'),
        'min_diameter_um': float(diams.min()) if len(diams) else float('nan'),
        'max_diameter_um': float(diams.max()) if len(diams) else float('nan'),
        'elapsed_sec': round(elapsed, 1),
    }
    print(f'  n={stats["n_particles"]}, mean={stats["mean_diameter_um"]:.1f}um, '
          f'std={stats["std_diameter_um"]:.1f}um, elapsed={elapsed:.1f}s')
    return stats


def main():
    rf = joblib.load(RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(RESULTS / 'combined_intensity_range.joblib')

    all_stats = []
    for name, start, end, out_dir in ZONES:
        stats = process_zone(name, start, end, out_dir, rf, intensity_range)
        all_stats.append(stats)

    summary_df = pd.DataFrame(all_stats)
    summary_df.to_csv(RESULTS / 'zone_summary.csv', index=False)
    print('\n=== Zone summary ===')
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
