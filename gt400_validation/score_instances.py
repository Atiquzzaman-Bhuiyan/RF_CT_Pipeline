#!/usr/bin/env python3
"""
Instance-level validation of segmentation methods against the held-out manual reference
(LHS-1, slices 400-489).  Run on the Mac (GT on the external drive; copy the method .tif
files from Jubail first).

    python3 score_instances.py --gt "/Volumes/My Passport/roi5" \
        --methods-dir ~/gt400_out --gt-slices 90 --out ~/gt400_scores --check-offset

Rules (declare these in the paper):
  * GT = roi5 cropped to its first --gt-slices slices (slice index 0 == CT slice 400; verified by
    --check-offset).  Labels are compacted; IDs are not otherwise changed.
  * Size floors: a label counts only if it has >= floor voxels.  Floors 67 (100 um) and 227
    (150 um, Method E's native floor) are both reported, applied identically to GT and methods.
  * Matching uses ALL grains, edge grains included (every method sees the same cut).
  * Edge grains = labels whose cross-section on the first/last slice exceeds --edge-ratio
    (default 0.2) of their maximum cross-section, or that touch the x/y frame border.  They are
    excluded from SIZE statistics only.
  * Non-compact GT grains (V/sphere-of-max-inscribed-radius > 12 or >= 3 EDT peaks at 70%) are
    NOT excluded; they are reported as a sensitivity subset ("_nc" columns exclude them).
  * GT grain classes (IoU threshold --iou, default 0.5):
      detected : some method label has IoU >= thr
      merged   : no IoU >= thr, but a method label covers >= 50% of the GT grain (it also covers
                 other material, i.e. under-segmentation)
      split    : >= 2 method labels each cover >= 10% of the GT grain and together >= 50%
      missed   : total coverage < 10%
      other    : anything else (poor overlap)
  * Merged grains are further split into merged_true (the covering label also covers >= 1 other GT
    grain: real under-segmentation) and merged_leak (the label covers this GT grain plus non-GT
    material: a foreground-definition effect, not a failure to separate grains).
  * Recall is robust to an incomplete GT.  Precision is only a LOWER BOUND when the GT is
    incomplete (real grains the annotator did not trace count as false positives).
"""
import argparse, glob, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.measure import label as cc_label

VOXEL_UM = 19.8


def load_stack(path, n=None):
    p = Path(path).expanduser()
    if p.is_dir():
        files = sorted(glob.glob(str(p / '*.tif*')))
        if not files:
            sys.exit(f'no tif files in {p}')
        files = files[:n] if n else files
        return np.stack([tifffile.imread(f) for f in files])
    vol = tifffile.imread(p)
    return vol[:n] if n else vol


def compact(lab):
    ids, inv = np.unique(lab, return_inverse=True)
    inv = np.asarray(inv).reshape(lab.shape)
    if ids[0] != 0:                      # no background present
        inv = inv + 1
        ids = np.concatenate([[0], ids])
    return inv.astype(np.int32), ids


def diam_um(vox):
    return (6.0 * np.asarray(vox, dtype=float) / np.pi) ** (1 / 3) * VOXEL_UM


def label_stats(lab, n, edge_ratio):
    Z = lab.shape[0]
    size = np.bincount(lab.ravel(), minlength=n)
    areas = np.stack([np.bincount(lab[z].ravel(), minlength=n) for z in range(Z)])
    maxarea = np.maximum(areas.max(axis=0), 1)
    r_first, r_last = areas[0] / maxarea, areas[-1] / maxarea
    xy = np.zeros(n, bool)
    for face in (lab[:, 0, :], lab[:, -1, :], lab[:, :, 0], lab[:, :, -1]):
        xy[np.unique(face)] = True
    touches_z = (areas[0] > 0) | (areas[-1] > 0)
    edge = (np.maximum(r_first, r_last) > edge_ratio) | xy
    edge[0] = False; touches_z[0] = False; xy[0] = False
    return size, edge, touches_z, xy


def noncompact(lab, size, min_vox=3000, ratio_thr=12.0, peaks_thr=3):
    flag = np.zeros(len(size), bool)
    objs = ndi.find_objects(lab)
    for i in np.where(size > min_vox)[0]:
        if i == 0 or objs[i - 1] is None:
            continue
        m = np.pad(lab[objs[i - 1]] == i, 1)
        edt = ndi.distance_transform_edt(m)
        r = edt.max()
        ratio = size[i] / (4 / 3 * np.pi * r ** 3)
        peaks = cc_label(edt > 0.7 * r).max()
        flag[i] = (ratio > ratio_thr) or (peaks >= peaks_thr)
    return flag


def overlaps(gt, m, nm):
    sel = (gt > 0) & (m > 0)
    key = gt[sel].astype(np.int64) * nm + m[sel]
    u, c = np.unique(key, return_counts=True)
    return (u // nm).astype(np.int64), (u % nm).astype(np.int64), c.astype(np.int64)


def evaluate(floor, gsize, gedge, gnc, msize, medge, ov, iou_thr, alt_thr):
    gi, mi, inter = ov
    ok = (gsize[gi] >= floor) & (msize[mi] >= floor)
    df = pd.DataFrame({'gi': gi[ok], 'mi': mi[ok], 'I': inter[ok]})
    df['Vg'] = gsize[df.gi]; df['Vm'] = msize[df.mi]
    df['iou'] = df.I / (df.Vg + df.Vm - df.I)
    df['cov'] = df.I / df.Vg
    groups = {g: grp for g, grp in df.groupby('gi')}
    cover_cnt = df[df['cov'] >= 0.5].groupby('mi').size().to_dict() if len(df) else {}
    gids = np.array([g for g in np.where(gsize >= floor)[0] if g > 0])
    rows = []
    for g in gids:
        grp = groups.get(g)
        if grp is None:
            rows.append((g, 'missed', 0.0, -1, 0, 0.0, '')); continue
        iou, cov, mids = grp.iou.values, grp['cov'].values, grp.mi.values
        k = iou.argmax(); best, bm = iou[k], mids[k]
        nfrag, covsum = int((cov >= 0.1).sum()), float(cov.sum())
        kind = ''
        if best >= iou_thr: cls = 'detected'
        elif (cov >= 0.5).any():
            cls = 'merged'
            mstar = mids[np.where(cov >= 0.5)[0][cov[cov >= 0.5].argmax()]]
            kind = 'true' if cover_cnt.get(mstar, 0) >= 2 else 'leak'
        elif nfrag >= 2 and cov[cov >= 0.1].sum() >= 0.5: cls = 'split'
        elif covsum < 0.1: cls = 'missed'
        else: cls = 'other'
        rows.append((g, cls, float(best), int(bm), nfrag, covsum, kind))
    gdf = pd.DataFrame(rows, columns=['gt_id', 'cls', 'best_iou', 'best_m', 'n_frag', 'cov_sum', 'merge_kind'])
    gdf['gt_vox'] = gsize[gdf.gt_id]; gdf['gt_D_um'] = diam_um(gdf.gt_vox)
    gdf['gt_edge'] = gedge[gdf.gt_id]; gdf['gt_noncompact'] = gnc[gdf.gt_id]
    gdf['m_vox'] = np.where(gdf.best_m >= 0, msize[np.maximum(gdf.best_m, 0)], 0)
    gdf['m_edge'] = np.where(gdf.best_m >= 0, medge[np.maximum(gdf.best_m, 0)], False)

    # method side
    mids_all = np.array([m for m in np.where(msize >= floor)[0] if m > 0])
    best_m = df.groupby('mi').iou.max() if len(df) else pd.Series(dtype=float)
    matched = int((best_m.reindex(mids_all).fillna(0) >= iou_thr).sum())
    merge_events = int((df[df['cov'] >= 0.5].groupby('mi').size() >= 2).sum()) if len(df) else 0

    def summarise(sub, tag):
        n = len(sub); out = {}
        for c in ['detected', 'merged', 'split', 'missed', 'other']:
            out[f'{c}{tag}'] = int((sub.cls == c).sum())
        out[f'recall@{iou_thr}{tag}'] = round((sub.cls == 'detected').mean(), 3) if n else np.nan
        out[f'recall@{alt_thr}{tag}'] = round((sub.best_iou >= alt_thr).mean(), 3) if n else np.nan
        out[f'merged_true{tag}'] = int(((sub.cls == 'merged') & (sub.merge_kind == 'true')).sum())
        out[f'merged_leak{tag}'] = int(((sub.cls == 'merged') & (sub.merge_kind == 'leak')).sum())
        return n, out

    n_all, s_all = summarise(gdf, '')
    n_nc, s_nc = summarise(gdf[~gdf.gt_noncompact], '_nc')
    det = gdf[(gdf.cls == 'detected') & ~gdf.gt_edge & ~gdf.m_edge]
    det_nc = det[~det.gt_noncompact]
    ratio = diam_um(det.m_vox) / det.gt_D_um if len(det) else pd.Series(dtype=float)
    ratio_nc = diam_um(det_nc.m_vox) / det_nc.gt_D_um if len(det_nc) else pd.Series(dtype=float)
    g_ne = gdf[~gdf.gt_edge]
    m_ne = mids_all[~medge[mids_all]]
    summ = {'floor_vox': floor, 'n_gt': n_all, **s_all,
            'recall_nonedge': round((g_ne.cls == 'detected').mean(), 3) if len(g_ne) else np.nan,
            'n_gt_nc': n_nc, **s_nc,
            'n_method': len(mids_all), 'precision_lb': round(matched / max(len(mids_all), 1), 3),
            'merge_events': merge_events,
            'mean_iou_det': round(gdf[gdf.cls == 'detected'].best_iou.mean(), 3) if (gdf.cls == 'detected').any() else np.nan,
            'n_size_pairs': len(det),
            'median_D_ratio': round(float(ratio.median()), 3) if len(ratio) else np.nan,
            'MAPE_D_pct': round(float((ratio - 1).abs().mean() * 100), 1) if len(ratio) else np.nan,
            'median_D_ratio_nc': round(float(ratio_nc.median()), 3) if len(ratio_nc) else np.nan,
            'n_gt_nonedge': len(g_ne), 'gt_median_D': round(float(g_ne.gt_D_um.median()), 1) if len(g_ne) else np.nan,
            'gt_mean_D': round(float(g_ne.gt_D_um.mean()), 1) if len(g_ne) else np.nan,
            'n_m_nonedge': len(m_ne),
            'm_median_D': round(float(np.median(diam_um(msize[m_ne]))), 1) if len(m_ne) else np.nan,
            'm_mean_D': round(float(np.mean(diam_um(msize[m_ne]))), 1) if len(m_ne) else np.nan,
            'gt_std_D': round(float(g_ne.gt_D_um.std()), 1) if len(g_ne) > 1 else np.nan,
            'm_std_D': round(float(np.std(diam_um(msize[m_ne]), ddof=1)), 1) if len(m_ne) > 1 else np.nan}
    if len(g_ne) > 1 and len(m_ne) > 1:
        from scipy.stats import ks_2samp, wasserstein_distance
        _g = g_ne.gt_D_um.values.astype(float); _m = diam_um(msize[m_ne]).astype(float)
        summ['KS_stat'] = round(float(ks_2samp(_g, _m).statistic), 3)
        summ['Wasserstein_um'] = round(float(wasserstein_distance(_g, _m)), 1)
    else:
        summ['KS_stat'] = summ['Wasserstein_um'] = np.nan
    return summ, gdf


def disc_geometry(fg3d):
    u = fg3d.any(axis=0)
    cy, cx = ndi.center_of_mass(u)
    return cy, cx, float(np.sqrt(u.sum() / np.pi))


def auto_roi_radius(gtfg, cy, cx, R, step=0.05):
    """Outer radius of the region where the GT was actually traced, from the GT alone (never from method
    scores): walk outward from 0.6 R and stop at the first ring whose GT foreground fraction drops below
    50% of the interior median (rings 0.2-0.6 R)."""
    Y, X = gtfg.shape[1:]
    yy, xx = np.mgrid[:Y, :X]
    r = np.hypot(yy - cy, xx - cx) / R
    cnt = gtfg.sum(axis=0)
    edges = np.arange(0, 1.0 + step, step)
    prof = np.array([cnt[(r >= lo) & (r < hi)].sum() / max(((r >= lo) & (r < hi)).sum() * gtfg.shape[0], 1)
                     for lo, hi in zip(edges[:-1], edges[1:])])
    interior = np.median(prof[(edges[:-1] >= 0.2) & (edges[:-1] < 0.6)])
    roi = 1.0
    for lo, p in zip(edges[:-1], prof):
        if lo >= 0.6 and p < 0.5 * interior:
            roi = float(lo); break
    return roi, edges, prof, interior


def roi_keep(lab, cy, cx, r_roi):
    keep = np.zeros(int(lab.max()) + 1, bool)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        y, x = (sl[1].start + sl[1].stop - 1) / 2, (sl[2].start + sl[2].stop - 1) / 2
        keep[i] = np.hypot(y - cy, x - cx) <= r_roi
    return keep


def offset_check(gt_fg, m_fg):
    print('\nZ-offset check (Dice of GT foreground vs method foreground; shift s means GT slice z+s <-> method slice z):')
    best = None
    for s in range(-3, 4):
        a, b = (gt_fg[s:], m_fg[:m_fg.shape[0] - s]) if s >= 0 else (gt_fg[:s], m_fg[-s:])
        d = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
        print(f'  shift {s:+d}: Dice {d:.4f}')
        if best is None or d > best[1]:
            best = (s, d)
    print(f'  best shift = {best[0]:+d}', '(OK, aligned)' if best[0] == 0 else '  <-- NOT ZERO: GT/CT slice indexing is off, fix before trusting any number')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True); ap.add_argument('--methods-dir', required=True)
    ap.add_argument('--gt-slices', type=int, default=90); ap.add_argument('--out', default='gt400_scores')
    ap.add_argument('--floors', type=int, nargs='+', default=[68, 227])
    ap.add_argument('--edge-ratio', type=float, default=0.2); ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--alt-iou', type=float, default=0.3); ap.add_argument('--no-shape', action='store_true')
    ap.add_argument('--oracle-fg', action='store_true', help='DIAGNOSTIC ONLY: zero method labels outside the GT foreground, to separate foreground-definition mismatch from splitting quality. Never a headline result.')
    ap.add_argument('--roi', choices=['none', 'auto'], default='none',
                    help="auto: score only inside the disc region where the GT was traced (radius from the GT's own radial density, never from method scores); labels are kept if their bbox centre lies inside. Applied identically to GT and methods.")
    ap.add_argument('--roi-frac', type=float, default=None, help='manual ROI radius as a fraction of the specimen radius (overrides auto)')
    ap.add_argument('--check-offset', action='store_true'); ap.add_argument('--pattern', default='method_*.tif')
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(a.methods_dir).expanduser() / a.pattern)))
    if not files: sys.exit('no method files found')
    gt_raw = load_stack(a.gt, a.gt_slices)
    roi_on, cy, cx, R, r_roi = a.roi != 'none' or a.roi_frac is not None, None, None, None, None
    if roi_on:
        dfile = next((f for f in files if Path(f).name.startswith('method_A_400') or Path(f).name.startswith('method_A_4')), files[0])
        cy, cx, R = disc_geometry(tifffile.imread(dfile)[:gt_raw.shape[0]] > 0)
        frac, edges, prof, interior = auto_roi_radius(gt_raw > 0, cy, cx, R)
        if a.roi_frac is not None: frac = a.roi_frac
        r_roi = frac * R
        print(f'Specimen disc from {Path(dfile).name}: centre (y={cy:.0f}, x={cx:.0f}), R={R:.0f} px')
        print('GT foreground fraction by ring (fraction of R -> % of ring voxels labelled):')
        print('  ' + '  '.join(f'{lo:.2f}:{100 * p:.1f}' for lo, p in zip(edges[:-1], prof)))
        print(f'ROI radius = {frac:.2f} R = {r_roi:.0f} px ({r_roi * VOXEL_UM / 1000:.2f} mm); interior GT fraction {100 * interior:.1f}%')
    gt, _ = compact(gt_raw); ng = int(gt.max()) + 1
    if roi_on:
        kg = roi_keep(gt, cy, cx, r_roi); gt = np.where(kg[gt], gt, 0).astype(np.int32)
    gsize, gedge, gtz, gxy = label_stats(gt, ng, a.edge_ratio)
    gnc = np.zeros(ng, bool) if a.no_shape else noncompact(gt, gsize)
    print(f'GT {gt.shape}: {ng - 1} labels; >=68 vox: {int((gsize[1:] >= 68).sum())}; '
          f'edge (ratio>{a.edge_ratio} or xy): {int(gedge[gsize >= 68].sum())}; '
          f'touch z-face at all: {int(gtz[gsize >= 68].sum())}; touch xy border: {int(gxy[gsize >= 68].sum())}; '
          f'non-compact (>=3000 vox): {int(gnc.sum())}')

    summaries, grain_tables, checked, fg_rows = [], [], False, []
    for f in files:
        name = re.match(r'method_(.+?)_\d+_\d+\.tif', Path(f).name)
        name = name.group(1) if name else Path(f).stem
        m_raw = tifffile.imread(f)[:gt.shape[0]]
        if m_raw.shape != gt.shape:
            print(f'SKIP {name}: shape {m_raw.shape} != GT {gt.shape}'); continue
        if a.check_offset and not checked and (name in ('C', 'D', 'A') or f == files[-1]):
            offset_check(gt_raw > 0, m_raw > 0); checked = True
        gfg, mfg = gt_raw > 0, m_raw > 0
        if roi_on:
            yy, xx = np.mgrid[:gt.shape[1], :gt.shape[2]]
            disc = (np.hypot(yy - cy, xx - cx) <= r_roi)[None]
            gfg, mfg = gfg & disc, mfg & disc
        fg_rows.append({'method': name, 'gt_fg_pct': round(100 * gfg.mean(), 1), 'method_fg_pct': round(100 * mfg.mean(), 1),
                        'gt_voxels_covered': round((gfg & mfg).sum() / gfg.sum(), 3),
                        'method_voxels_in_gt_lb': round((gfg & mfg).sum() / max(mfg.sum(), 1), 3),
                        'fg_dice': round(2 * (gfg & mfg).sum() / (gfg.sum() + mfg.sum()), 3)})
        if a.oracle_fg:
            m_raw = np.where(gfg, m_raw, 0)
        mlab, _ = compact(m_raw)
        if roi_on:
            km = roi_keep(mlab, cy, cx, r_roi); mlab = np.where(km[mlab], mlab, 0).astype(np.int32)
        nm = int(mlab.max()) + 1
        msize, medge, _, _ = label_stats(mlab, nm, a.edge_ratio)
        ov = overlaps(gt, mlab, nm)
        for fl in a.floors:
            s, g = evaluate(fl, gsize, gedge, gnc, msize, medge, ov, a.iou, a.alt_iou)
            s['method'] = name; summaries.append(s)
            g.insert(0, 'floor_vox', fl); g.insert(0, 'method', name); grain_tables.append(g)
        print(f'scored {name}', flush=True)

    FG = pd.DataFrame(fg_rows); FG.to_csv(out / 'foreground_agreement.csv', index=False)
    print('\nForeground agreement at voxel level' + (' (inside ROI)' if roi_on else '') + ' (before any size floor; method_voxels_in_gt is a LOWER bound):')
    print(FG.to_string(index=False))
    if a.oracle_fg:
        print('\n*** ORACLE-FG DIAGNOSTIC: method labels were masked to the GT foreground. Not a valid result. ***')
    S = pd.DataFrame(summaries)
    S.insert(0, 'method', S.pop('method'))
    S.to_csv(out / 'summary.csv', index=False)
    pd.concat(grain_tables).to_csv(out / 'per_grain.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    core = ['method', 'n_gt', 'detected', 'merged', 'merged_true', 'merged_leak', 'split', 'missed', 'other', f'recall@{a.iou}',
            f'recall@{a.alt_iou}', 'n_method', 'precision_lb', 'merge_events']
    size_cols = ['method', 'n_size_pairs', 'median_D_ratio', 'MAPE_D_pct', 'gt_median_D', 'm_median_D', 'gt_mean_D', 'm_mean_D', 'gt_std_D', 'm_std_D', 'KS_stat', 'Wasserstein_um', 'mean_iou_det', 'n_gt_nonedge', 'n_m_nonedge']
    nc_cols = ['method', 'n_gt_nc', f'recall@{a.iou}_nc', 'median_D_ratio_nc']
    for fl in a.floors:
        sub = S[S.floor_vox == fl]
        print(f'\n===== size floor {fl} voxels ({diam_um(fl):.0f} um) =====')
        print('Instance matching (all grains):'); print(sub[core].to_string(index=False))
        print('Size agreement (non-edge, detected pairs):'); print(sub[size_cols].to_string(index=False))
        print('Sensitivity, non-compact GT grains removed:'); print(sub[nc_cols].to_string(index=False))
    print(f'\nSaved: {out / "summary.csv"} and {out / "per_grain.csv"}')


if __name__ == '__main__':
    main()
