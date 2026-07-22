"""Overlay all 3 zones' particle-size distributions on one chart for direct comparison."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).parent / 'Results'

ZONES = [
    ('early (100-199)', RESULTS / 'ZoneEarly' / 'early_diameters.csv', 'tab:blue'),
    ('mid (350-449)', RESULTS / 'ZoneMid' / 'mid_diameters.csv', 'tab:orange'),
    ('late (750-849)', RESULTS / 'ZoneLate' / 'late_diameters.csv', 'tab:green'),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, path, color in ZONES:
        diams = pd.read_csv(path)['diameter_um'].values
        axes[0].hist(diams, bins=40, alpha=0.45, color=color, label=f'{name} (n={len(diams)})', density=True)

        from scipy.stats import gaussian_kde
        import numpy as np
        kde = gaussian_kde(diams)
        xs = np.linspace(diams.min(), diams.max(), 300)
        axes[1].plot(xs, kde(xs), color=color, lw=2, label=f'{name} (mean={diams.mean():.0f}µm)')

    axes[0].set_xlabel('Diameter (µm)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Particle size distribution — histogram overlay')
    axes[0].legend()

    axes[1].set_xlabel('Diameter (µm)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Particle size distribution — KDE overlay')
    axes[1].legend()

    fig.tight_layout()
    out_path = RESULTS / 'zone_distribution_comparison.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
