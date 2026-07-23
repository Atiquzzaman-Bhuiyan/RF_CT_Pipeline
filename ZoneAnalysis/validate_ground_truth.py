"""
Ground-truth validation for Methods C and D against manually-traced particles
(slices 300-399). Read-only reuse of the already-trained model and the
existing segmentation functions via import -- no retraining, no modification
to pipeline_common.py, pipeline_method_d.py, zone_analysis.py,
zone_analysis_method_d.py, full_volume_analysis*.py, or any existing output
file. Writes exclusively to Results/Validation/, a new subfolder that does
not overlap with ZoneEarly/ZoneMid/ZoneLate/FullVolume/MethodD.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline_common import (resolve_ct_path, load_tiff, predict_probability_volume,
                              compute_diameters, segment_watershed, VOXEL_UM)
from pipeline_method_d import segment_method_d, build_normalized_intensity_volume

BASE = Path(__file__).parent
RESULTS = BASE / 'Results'
OUT_DIR = RESULTS / 'Validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLICE_START, SLICE_END = 300, 400  # [300, 400) -> slices 300..399 inclusive

# Ground truth reference (manual tracing, ORS Dragonfly, slices 300-399)
# Source: manuscript Section 2.2.6 -- NOT computed by this script.
# If a per-particle ground truth CSV becomes available, replace these
# constants with a direct load + computed stats for full traceability.
GT_N = 1815
GT_MEAN_UM = 185.3
GT_MEDIAN_UM = 147.8  # reference only -- not used in the two requested error formulas
GT_STD_UM = 119.1     # reference only -- not used in the two requested error formulas
GT_Z_RANGE_UM = (5940, 7900)  # manuscript-stated z-range for this ground truth


def load_volume(start, end):
    slices, filenames = [], []
    for n in range(start, end):
        path = resolve_ct_path(n)
        slices.append(load_tiff(path))
        filenames.append(path.name)
    return np.stack(slices, axis=0), filenames


def sanity_check_slice_indexing(filenames, start, end):
    """Confirm slice indexing before trusting anything downstream: print the
    first/last resolved filenames and the z-range implied by
    slice_index * VOXEL_UM, so a 0-indexing vs 1-indexing or origin mismatch
    against the manuscript's stated z-range would be caught here rather than
    silently baked into the validation numbers."""
    z_start = start * VOXEL_UM
    z_end = (end - 1) * VOXEL_UM
    gt_start, gt_end = GT_Z_RANGE_UM
    matches = abs(z_start - gt_start) < 1 and abs(z_end - gt_end) < 1

    print('=== Slice-indexing sanity check ===')
    print(f'  Requested slice range: {start}-{end - 1} (inclusive)')
    print(f'  First file resolved: {filenames[0]}')
    print(f'  Last file resolved:  {filenames[-1]}')
    print(f'  Implied z-range (slice_index * VOXEL_UM={VOXEL_UM}um): '
          f'{z_start:.1f}um - {z_end:.1f}um')
    print(f'  Manuscript-stated z-range for this ground truth: '
          f'{gt_start}um - {gt_end}um')
    print(f'  Match: {"YES (within rounding)" if matches else "MISMATCH -- STOP, DO NOT TRUST RESULTS BELOW"}')
    print()
    if not matches:
        raise RuntimeError(
            'Slice-indexing sanity check failed: implied z-range does not match '
            'the manuscript-stated ground-truth z-range. Refusing to proceed -- '
            'check whether slice numbering is 0-indexed vs 1-indexed, or whether '
            'the acquisition origin differs from what this script assumes.'
        )


def run_method_c(volume, rf, intensity_range):
    prob_volume = predict_probability_volume(volume, rf, intensity_range)
    labeled = segment_watershed(prob_volume)
    return compute_diameters(labeled)


def run_method_d(volume, rf, intensity_range):
    prob_volume = predict_probability_volume(volume, rf, intensity_range)
    intensity_volume = build_normalized_intensity_volume(volume, intensity_range)
    labeled, n_seeds = segment_method_d(prob_volume, intensity_volume)
    return compute_diameters(labeled)


def stats_row(name, diams):
    return {
        'method': name, 'n_particles': len(diams),
        'mean_diameter_um': float(diams.mean()) if len(diams) else float('nan'),
        'median_diameter_um': float(np.median(diams)) if len(diams) else float('nan'),
        'std_diameter_um': float(diams.std()) if len(diams) else float('nan'),
        'min_diameter_um': float(diams.min()) if len(diams) else float('nan'),
        'max_diameter_um': float(diams.max()) if len(diams) else float('nan'),
    }


def add_error_metrics(row):
    row['diameter_error_pct'] = abs(row['mean_diameter_um'] - GT_MEAN_UM) / GT_MEAN_UM * 100
    row['count_error_pct'] = (row['n_particles'] - GT_N) / GT_N * 100
    return row


def main():
    print(f'Loading slices {SLICE_START}-{SLICE_END - 1}...')
    volume, filenames = load_volume(SLICE_START, SLICE_END)
    print(f'  volume shape: {volume.shape}')
    sanity_check_slice_indexing(filenames, SLICE_START, SLICE_END)

    rf = joblib.load(RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(RESULTS / 'combined_intensity_range.joblib')

    print('Running Method C on slices 300-399...')
    diams_c = run_method_c(volume, rf, intensity_range)
    pd.DataFrame({'diameter_um': diams_c}).to_csv(OUT_DIR / 'validation_300_399_methodC.csv', index=False)
    print(f'  n={len(diams_c)}, mean={diams_c.mean():.1f}um')

    print('Running Method D on slices 300-399...')
    diams_d = run_method_d(volume, rf, intensity_range)
    pd.DataFrame({'diameter_um': diams_d}).to_csv(OUT_DIR / 'validation_300_399_methodD.csv', index=False)
    print(f'  n={len(diams_d)}, mean={diams_d.mean():.1f}um')

    row_gt = {
        'method': 'ground_truth (manual, Dragonfly)', 'n_particles': GT_N,
        'mean_diameter_um': GT_MEAN_UM, 'median_diameter_um': GT_MEDIAN_UM,
        'std_diameter_um': GT_STD_UM, 'min_diameter_um': float('nan'), 'max_diameter_um': float('nan'),
        'diameter_error_pct': 0.0, 'count_error_pct': 0.0,
    }
    row_c = add_error_metrics(stats_row('C', diams_c))
    row_d = add_error_metrics(stats_row('D', diams_d))
    summary_df = pd.DataFrame([row_gt, row_c, row_d])
    summary_df.to_csv(OUT_DIR / 'validation_summary.csv', index=False)

    print('\n=== Validation summary (slices 300-399 vs manual ground truth) ===')
    print(summary_df.to_string(index=False))

    # Before/after context: existing full-run zone stats, read-only, unmodified.
    print('\n=== For reference only -- existing full-run zone stats (unchanged, read-only) ===')
    try:
        zone_c = pd.read_csv(RESULTS / 'zone_summary.csv')
        print('\nMethod C zones:')
        print(zone_c[['zone', 'slice_range', 'n_particles', 'mean_diameter_um', 'std_diameter_um']].to_string(index=False))
    except FileNotFoundError:
        print('  (Method C zone_summary.csv not found)')
    try:
        zone_d = pd.read_csv(RESULTS / 'MethodD' / 'zone_summary_methodD.csv')
        print('\nMethod D zones:')
        print(zone_d[['zone', 'slice_range', 'n_particles', 'mean_diameter_um', 'std_diameter_um']].to_string(index=False))
    except FileNotFoundError:
        print('  (Method D zone_summary_methodD.csv not found)')

    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
