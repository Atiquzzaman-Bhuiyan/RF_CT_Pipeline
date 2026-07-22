"""Resume the full-volume run: chunks 1-9 (real slices 0-899) already completed
correctly before the crash on the corrupted chunk-10 slice range. This just
runs the corrected final chunk (real central slices 900-975) and merges with
the existing partial results, instead of redoing the whole ~35 minutes."""
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from pipeline_common import (resolve_ct_path, load_tiff, predict_probability_volume,
                              segment_watershed, compute_diameters_central, BATCH_SIZE, OVERLAP)
from full_volume_analysis import discover_slice_numbers

RESULTS = Path(__file__).parent / 'Results'
OUT_DIR = RESULTS / 'FullVolume'


def main():
    t_start = time.time()
    rf = joblib.load(RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(RESULTS / 'combined_intensity_range.joblib')

    slice_numbers = discover_slice_numbers()
    n_slices = len(slice_numbers)
    print(f'{n_slices} unique slices (range {slice_numbers[0]}-{slice_numbers[-1]})')

    existing = pd.read_csv(OUT_DIR / 'full_volume_diameters_PARTIAL.csv')['diameter_um'].tolist()
    print(f'Loaded {len(existing)} particles from chunks 1-9 (real central slices 0-899)')

    # Final chunk: central 900 through the true end (975), not the corrupted 6911.
    n_chunks = (n_slices + BATCH_SIZE - 1) // BATCH_SIZE
    c = n_chunks - 1
    central_start = c * BATCH_SIZE
    central_end = n_slices
    load_start = max(0, central_start - OVERLAP)
    load_end = n_slices

    chunk_slice_nums = slice_numbers[load_start:load_end]
    print(f'Final chunk: slices {chunk_slice_nums[0]}-{chunk_slice_nums[-1]} '
          f'(central {slice_numbers[central_start]}-{slice_numbers[-1]})')

    t_chunk = time.time()
    volume = np.stack([load_tiff(resolve_ct_path(n)) for n in chunk_slice_nums], axis=0)
    prob_volume = predict_probability_volume(volume, rf, intensity_range)
    labeled = segment_watershed(prob_volume)

    local_central_start = central_start - load_start
    local_central_end = central_end - load_start
    diams = compute_diameters_central(labeled, local_central_start, local_central_end)
    print(f'  final chunk particles: {len(diams)}, mean={diams.mean():.1f}um | {time.time() - t_chunk:.1f}s')

    all_diams = np.array(existing + diams.tolist())
    t_total = time.time() - t_start

    print(f'\n=== Full volume results (976 slices) ===')
    print(f'Total particles: {len(all_diams):,}')
    print(f'Mean: {all_diams.mean():.1f}um, Std: {all_diams.std():.1f}um, '
          f'Median: {np.median(all_diams):.1f}um, Min: {all_diams.min():.1f}um, Max: {all_diams.max():.1f}um')
    print(f'Resume runtime: {t_total / 60:.1f} min')

    pd.DataFrame({'diameter_um': all_diams}).to_csv(OUT_DIR / 'full_volume_diameters.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_diams, bins=50, color='steelblue', edgecolor='white')
    ax.axvline(all_diams.mean(), color='red', linestyle='--', label=f'mean={all_diams.mean():.1f}µm')
    ax.set_xlabel('Diameter (µm)')
    ax.set_ylabel('Particle count')
    ax.set_title(f'Full volume (976 slices) — particle size distribution (n={len(all_diams)})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'full_volume_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'zone': 'full_volume', 'slice_range': f'{slice_numbers[0]}-{slice_numbers[-1]}',
        'n_particles': len(all_diams),
        'mean_diameter_um': float(all_diams.mean()), 'std_diameter_um': float(all_diams.std()),
        'median_diameter_um': float(np.median(all_diams)),
        'min_diameter_um': float(all_diams.min()), 'max_diameter_um': float(all_diams.max()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / 'full_volume_summary.csv', index=False)
    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
