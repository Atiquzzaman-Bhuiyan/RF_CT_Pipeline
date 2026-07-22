"""Method C vs Method D size-distribution comparison, per zone and full volume.
Reads only the already-saved diameter CSVs from both methods -- no recompute."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

RESULTS = Path(__file__).parent / 'Results'

PANELS = [
    ('early', 'Early (100-199)', RESULTS / 'ZoneEarly' / 'early_diameters.csv',
     RESULTS / 'MethodD' / 'early_diameters_methodD.csv'),
    ('mid', 'Mid (350-449)', RESULTS / 'ZoneMid' / 'mid_diameters.csv',
     RESULTS / 'MethodD' / 'mid_diameters_methodD.csv'),
    ('late', 'Late (750-849)', RESULTS / 'ZoneLate' / 'late_diameters.csv',
     RESULTS / 'MethodD' / 'late_diameters_methodD.csv'),
    ('full_volume', 'Full volume (0-975)', RESULTS / 'FullVolume' / 'full_volume_diameters.csv',
     RESULTS / 'MethodD' / 'full_volume_diameters_methodD.csv'),
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, (key, title, c_path, d_path) in zip(axes, PANELS):
        c_diams = pd.read_csv(c_path)['diameter_um'].values
        d_diams = pd.read_csv(d_path)['diameter_um'].values

        kde_c = gaussian_kde(c_diams)
        kde_d = gaussian_kde(d_diams)
        xmax = max(c_diams.max(), d_diams.max())
        xs = np.linspace(90, xmax, 400)

        ax.plot(xs, kde_c(xs), color='steelblue', lw=2,
                label=f'Method C (n={len(c_diams)}, mean={c_diams.mean():.0f}µm)')
        ax.plot(xs, kde_d(xs), color='indianred', lw=2,
                label=f'Method D (n={len(d_diams)}, mean={d_diams.mean():.0f}µm)')
        ax.fill_between(xs, kde_c(xs), alpha=0.15, color='steelblue')
        ax.fill_between(xs, kde_d(xs), alpha=0.15, color='indianred')
        ax.set_xlabel('Diameter (µm)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle('Method C (RF probability-gradient watershed) vs Method D (hybrid RF-intensity watershed)',
                 fontsize=13)
    fig.tight_layout()
    out_path = RESULTS / 'methodC_vs_methodD_comparison.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
