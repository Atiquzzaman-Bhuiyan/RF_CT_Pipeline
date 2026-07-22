"""Full-volume particle segmentation, chunked (BATCH_SIZE + 2*OVERLAP per chunk,
only central BATCH_SIZE slices' particles kept per chunk) to bound memory and
avoid double-counting particles at chunk seams -- same approach as
method_c_full_volume_final.py. Saves partial results after every chunk so a
crash/interrupt doesn't lose progress."""
import glob
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from pipeline_common import (CT_DIR, resolve_ct_path, load_tiff, predict_probability_volume,
                              segment_watershed, compute_diameters_central, BATCH_SIZE, OVERLAP)

RESULTS = Path(__file__).parent / 'Results'
OUT_DIR = RESULTS / 'FullVolume'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def discover_slice_numbers():
    import re
    files = sorted(glob.glob(str(CT_DIR / 'Copy of slice_*.tif')))
    nums = set()
    for f in files:
        name = Path(f).stem
        m = re.search(r'slice_(\d{6})', name)
        if m is None:
            print(f'  WARNING: could not parse slice number from {name!r}, skipping')
            continue
        nums.add(int(m.group(1)))
    return sorted(nums)


def main():
    t_start = time.time()
    rf = joblib.load(RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(RESULTS / 'combined_intensity_range.joblib')

    slice_numbers = discover_slice_numbers()
    n_slices = len(slice_numbers)
    print(f'Found {n_slices} slices (range {slice_numbers[0]}-{slice_numbers[-1]})')

    all_diams = []
    n_chunks = (n_slices + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'Processing {n_chunks} chunks of {BATCH_SIZE} slices ({OVERLAP}-slice overlap each side)...')

    for c in range(n_chunks):
        central_start = c * BATCH_SIZE
        central_end = min(central_start + BATCH_SIZE, n_slices)
        load_start = max(0, central_start - OVERLAP)
        load_end = min(n_slices, central_end + OVERLAP)

        chunk_slice_nums = slice_numbers[load_start:load_end]
        print(f'\nChunk {c + 1}/{n_chunks}: slices {chunk_slice_nums[0]}-{chunk_slice_nums[-1]} '
              f'(central {slice_numbers[central_start]}-{slice_numbers[central_end - 1]})')

        t_chunk = time.time()
        volume = np.stack([load_tiff(resolve_ct_path(n)) for n in chunk_slice_nums], axis=0)

        prob_volume = predict_probability_volume(volume, rf, intensity_range)
        labeled = segment_watershed(prob_volume)

        local_central_start = central_start - load_start
        local_central_end = central_end - load_start
        diams = compute_diameters_central(labeled, local_central_start, local_central_end)
        all_diams.extend(diams.tolist())

        elapsed = time.time() - t_chunk
        if len(diams) > 0:
            print(f'  chunk particles (central only): {len(diams)}, mean={diams.mean():.1f}um | {elapsed:.1f}s')
        else:
            print(f'  no particles in central region | {elapsed:.1f}s')

        pd.DataFrame({'diameter_um': all_diams}).to_csv(OUT_DIR / 'full_volume_diameters_PARTIAL.csv', index=False)
        with open(OUT_DIR / 'progress.txt', 'w') as f:
            f.write(f'Chunks completed: {c + 1}/{n_chunks}\n')
            f.write(f'Total particles so far: {len(all_diams)}\n')
            if all_diams:
                f.write(f'Running mean diameter: {np.mean(all_diams):.1f} um\n')
            f.write(f'Elapsed: {(time.time() - t_start) / 60:.1f} min\n')

        del volume, prob_volume, labeled

    all_diams = np.array(all_diams)
    t_total = time.time() - t_start

    print(f'\n=== Full volume results ===')
    print(f'Total particles: {len(all_diams):,}')
    if len(all_diams):
        print(f'Mean: {all_diams.mean():.1f}um, Std: {all_diams.std():.1f}um, '
              f'Median: {np.median(all_diams):.1f}um, Min: {all_diams.min():.1f}um, Max: {all_diams.max():.1f}um')
    print(f'Total runtime: {t_total / 60:.1f} min')

    pd.DataFrame({'diameter_um': all_diams}).to_csv(OUT_DIR / 'full_volume_diameters.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_diams, bins=50, color='steelblue', edgecolor='white')
    ax.axvline(all_diams.mean(), color='red', linestyle='--', label=f'mean={all_diams.mean():.1f}µm')
    ax.set_xlabel('Diameter (µm)')
    ax.set_ylabel('Particle count')
    ax.set_title(f'Full volume — particle size distribution (n={len(all_diams)})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'full_volume_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'zone': 'full_volume', 'slice_range': f'{slice_numbers[0]}-{slice_numbers[-1]}',
        'n_particles': len(all_diams),
        'mean_diameter_um': float(all_diams.mean()) if len(all_diams) else float('nan'),
        'std_diameter_um': float(all_diams.std()) if len(all_diams) else float('nan'),
        'median_diameter_um': float(np.median(all_diams)) if len(all_diams) else float('nan'),
        'min_diameter_um': float(all_diams.min()) if len(all_diams) else float('nan'),
        'max_diameter_um': float(all_diams.max()) if len(all_diams) else float('nan'),
        'elapsed_sec': round(t_total, 1),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / 'full_volume_summary.csv', index=False)
    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
