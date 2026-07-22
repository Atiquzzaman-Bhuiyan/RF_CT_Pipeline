"""Method D zone-level segmentation. Reuses Method C's already-trained model
(combined_model.joblib) and the unmodified predict_probability_volume /
compute_diameters functions from pipeline_common.py, but segments with the
hybrid RF-intensity watershed from pipeline_method_d.py instead of Method C's
probability-only watershed. Writes to Results/MethodD/ -- never touches
Method C's ZoneEarly/ZoneMid/ZoneLate outputs."""
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from pipeline_common import resolve_ct_path, load_tiff, predict_probability_volume, compute_diameters
from pipeline_method_d import segment_method_d, build_normalized_intensity_volume

BASE = Path(__file__).parent
RESULTS = BASE / 'Results'
OUT_DIR = RESULTS / 'MethodD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZONES = [
    ('early', 100, 200),
    ('mid', 350, 450),
    ('late', 750, 850),
]


def load_zone_volume(start, end):
    slices = []
    for n in range(start, end):
        try:
            slices.append(load_tiff(resolve_ct_path(n)))
        except FileNotFoundError:
            print(f'  WARNING: slice {n} missing, skipping')
    return np.stack(slices, axis=0)


def process_zone(name, start, end, rf, intensity_range):
    print(f'\n=== Method D — Zone: {name} (slices {start}-{end - 1}) ===')
    t0 = time.time()

    volume = load_zone_volume(start, end)
    print(f'  volume shape: {volume.shape}')

    prob_volume = predict_probability_volume(volume, rf, intensity_range)
    intensity_volume = build_normalized_intensity_volume(volume, intensity_range)

    labeled, n_seeds = segment_method_d(prob_volume, intensity_volume)
    diams = compute_diameters(labeled)

    pd.DataFrame({'diameter_um': diams}).to_csv(OUT_DIR / f'{name}_diameters_methodD.csv', index=False)

    elapsed = time.time() - t0
    stats = {
        'method': 'D', 'zone': name, 'slice_range': f'{start}-{end - 1}',
        'n_particles': len(diams), 'n_seeds': n_seeds,
        'mean_diameter_um': float(diams.mean()) if len(diams) else float('nan'),
        'median_diameter_um': float(np.median(diams)) if len(diams) else float('nan'),
        'std_diameter_um': float(diams.std()) if len(diams) else float('nan'),
        'min_diameter_um': float(diams.min()) if len(diams) else float('nan'),
        'max_diameter_um': float(diams.max()) if len(diams) else float('nan'),
        'elapsed_sec': round(elapsed, 1),
    }
    print(f'  n={stats["n_particles"]}, n_seeds={n_seeds}, mean={stats["mean_diameter_um"]:.1f}um, '
          f'std={stats["std_diameter_um"]:.1f}um, elapsed={elapsed:.1f}s')
    return stats


def main():
    rf = joblib.load(RESULTS / 'combined_model.joblib')
    intensity_range = joblib.load(RESULTS / 'combined_intensity_range.joblib')

    all_stats = []
    for name, start, end in ZONES:
        stats = process_zone(name, start, end, rf, intensity_range)
        all_stats.append(stats)

    summary_df = pd.DataFrame(all_stats)
    summary_df.to_csv(OUT_DIR / 'zone_summary_methodD.csv', index=False)
    print('\n=== Method D zone summary ===')
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
