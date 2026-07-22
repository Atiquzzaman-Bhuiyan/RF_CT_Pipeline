"""Method D full-volume segmentation, chunked identically to Method C's
full_volume_analysis.py (same BATCH_SIZE/OVERLAP, same central-region
diameter counting to avoid double-counting at chunk seams), but using the
hybrid RF-intensity watershed instead. Imports (does not modify)
discover_slice_numbers from full_volume_analysis.py to reuse the corrected
duplicate-file-safe slice discovery. Writes to Results/MethodD/ only."""
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from pipeline_common import resolve_ct_path, load_tiff, predict_probability_volume, \
    compute_diameters_central, BATCH_SIZE, OVERLAP
from pipeline_method_d import segment_method_d, build_normalized_intensity_volume
from full_volume_analysis import discover_slice_numbers

RESULTS = Path(__file__).parent / 'Results'
OUT_DIR = RESULTS / 'MethodD'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_start = time.time()
    rf = joblib.load(RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(RESULTS / 'combined_intensity_range.joblib')

    slice_numbers = discover_slice_numbers()
    n_slices = len(slice_numbers)
    print(f'Method D full volume: {n_slices} unique slices (range {slice_numbers[0]}-{slice_numbers[-1]})')

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
        intensity_volume = build_normalized_intensity_volume(volume, intensity_range)

        labeled, n_seeds = segment_method_d(prob_volume, intensity_volume)

        local_central_start = central_start - load_start
        local_central_end = central_end - load_start
        diams = compute_diameters_central(labeled, local_central_start, local_central_end)
        all_diams.extend(diams.tolist())

        elapsed = time.time() - t_chunk
        if len(diams) > 0:
            print(f'  chunk particles (central only): {len(diams)}, n_seeds={n_seeds}, '
                  f'mean={diams.mean():.1f}um | {elapsed:.1f}s')
        else:
            print(f'  no particles in central region, n_seeds={n_seeds} | {elapsed:.1f}s')

        pd.DataFrame({'diameter_um': all_diams}).to_csv(
            OUT_DIR / 'full_volume_diameters_methodD_PARTIAL.csv', index=False)
        with open(OUT_DIR / 'progress_methodD.txt', 'w') as f:
            f.write(f'Chunks completed: {c + 1}/{n_chunks}\n')
            f.write(f'Total particles so far: {len(all_diams)}\n')
            if all_diams:
                f.write(f'Running mean diameter: {np.mean(all_diams):.1f} um\n')
            f.write(f'Elapsed: {(time.time() - t_start) / 60:.1f} min\n')

        del volume, prob_volume, intensity_volume, labeled

    all_diams = np.array(all_diams)
    t_total = time.time() - t_start

    print(f'\n=== Method D full volume results ===')
    print(f'Total particles: {len(all_diams):,}')
    if len(all_diams):
        print(f'Mean: {all_diams.mean():.1f}um, Std: {all_diams.std():.1f}um, '
              f'Median: {np.median(all_diams):.1f}um, Min: {all_diams.min():.1f}um, Max: {all_diams.max():.1f}um')
    print(f'Total runtime: {t_total / 60:.1f} min')

    pd.DataFrame({'diameter_um': all_diams}).to_csv(OUT_DIR / 'full_volume_diameters_methodD.csv', index=False)

    summary = {
        'method': 'D', 'zone': 'full_volume', 'slice_range': f'{slice_numbers[0]}-{slice_numbers[-1]}',
        'n_particles': len(all_diams),
        'mean_diameter_um': float(all_diams.mean()) if len(all_diams) else float('nan'),
        'median_diameter_um': float(np.median(all_diams)) if len(all_diams) else float('nan'),
        'std_diameter_um': float(all_diams.std()) if len(all_diams) else float('nan'),
        'min_diameter_um': float(all_diams.min()) if len(all_diams) else float('nan'),
        'max_diameter_um': float(all_diams.max()) if len(all_diams) else float('nan'),
        'elapsed_sec': round(t_total, 1),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / 'full_volume_summary_methodD.csv', index=False)
    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
