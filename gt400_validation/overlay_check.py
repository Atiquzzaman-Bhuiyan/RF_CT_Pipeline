#!/usr/bin/env python3
"""
Visual check of GT vs one method on chosen slices.  Left: CT + GT outlines (green; blue = GT grain the
method missed entirely).  Right: CT + method outlines (red; yellow = method label with NO overlap with
any GT grain -> either a false positive or a real grain the GT never traced).

  python3 overlay_check.py --gt "/Volumes/My Passport/roi5" --method ~/gt400_out/method_C_400_489.tif \
      --ct-dir "/Volumes/My Passport/ct_slices" --slices 10 45 80 --out ~/gt400_overlays
Slice numbers are indices inside the core (0 -> CT slice 400).
"""
import argparse, glob
from pathlib import Path
import numpy as np, tifffile
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

ap = argparse.ArgumentParser()
ap.add_argument('--gt', required=True); ap.add_argument('--method', required=True)
ap.add_argument('--ct-dir', required=True); ap.add_argument('--slices', type=int, nargs='+', default=[10, 45, 80])
ap.add_argument('--start', type=int, default=400); ap.add_argument('--out', default='gt400_overlays')
a = ap.parse_args()
out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(str(Path(a.gt).expanduser() / '*.tif*')))
m = tifffile.imread(Path(a.method).expanduser())
name = Path(a.method).stem
for k in a.slices:
    g = tifffile.imread(files[k]); ms = m[k]
    cands = [Path(a.ct_dir) / f'Copy of slice_{a.start + k:06d}.tif', Path(a.ct_dir) / f'slice_{a.start + k:06d}.tif']
    ct = next((tifffile.imread(c).astype(np.float32) for c in cands if c.exists()), None)
    if ct is None: raise SystemExit(f'CT slice {a.start + k} not found in {a.ct_dir}')
    lo, hi = np.percentile(ct, [0.5, 99.5]); img = np.clip((ct - lo) / (hi - lo), 0, 1)
    g_ids, m_ids = np.unique(g[g > 0]), np.unique(ms[ms > 0])
    g_in_m = np.unique(g[(g > 0) & (ms > 0)]); m_in_g = np.unique(ms[(g > 0) & (ms > 0)])
    g_missed = np.isin(g, np.setdiff1d(g_ids, g_in_m)) & (g > 0)
    m_only = np.isin(ms, np.setdiff1d(m_ids, m_in_g)) & (ms > 0)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for axis, title in zip(ax, [f'GT  (slice {a.start + k})', f'{name}  (slice {a.start + k})']):
        axis.imshow(img, cmap='gray'); axis.set_title(title); axis.axis('off')
    rgb = np.zeros(g.shape + (4,)); rgb[find_boundaries(g > 0, mode='inner') | find_boundaries(g, mode='inner')] = (0, 1, 0, 1)
    rgb[g_missed & find_boundaries(g, mode='inner')] = (0.2, 0.5, 1, 1); ax[0].imshow(rgb)
    rgb = np.zeros(ms.shape + (4,)); rgb[find_boundaries(ms, mode='inner')] = (1, 0, 0, 1)
    rgb[m_only & find_boundaries(ms, mode='inner')] = (1, 1, 0, 1); ax[1].imshow(rgb)
    fig.tight_layout(); fig.savefig(out / f'{name}_slice{a.start + k}.png', dpi=110); plt.close(fig)
    print('saved', out / f'{name}_slice{a.start + k}.png', f'| GT grains {len(g_ids)}, missed {int(len(g_ids) - len(g_in_m))}; '
          f'method labels {len(m_ids)}, no-GT-overlap {int(len(m_ids) - len(m_in_g))}')
