"""
Pixel-level segmentation accuracy metrics -- all five methods (A-E), at the
6 manually-annotated slices (151, 350, 500, 700, 800, 900), evaluated
against the Dragonfly ground truth.

Ground truth: pipeline_common.load_particle_labels() -- closed-world
labeling (particle=1, everything else=0), the exact same labeling rule
that produced Table 2's published numbers (verified: matches
Results/Tables/metrics_per_fold_bgFalse.csv's n_pixels=260,925 per slice,
i.e. the full 525x497 frame, no cropping/masking).

Metric implementation: reproduces RandomForest_Validation.ipynb Cell 11's
compute_metrics() exactly -- sklearn precision/recall/F1 with
zero_division=0, matthews_corrcoef for MCC, roc_auc_score /
average_precision_score for ROC-AUC / PR-AUC where a continuous score is
available.

IMPORTANT METHODOLOGICAL NOTE: this evaluates each method's FINAL
SEGMENTATION OUTPUT (post-watershed, post MIN_VOXELS size-filter, binarized
foreground mask), not the raw RF pixel classifier that Table 2 measures.
For C/D this is a meaningfully different y_pred than Table 2's (watershed
+ size-filtering changes the boundary and can drop sub-100um regions), so
C/D-LOSO numbers here will NOT exactly match Table 2's published figures
even though they use the identical underlying fold models. ROC-AUC/PR-AUC
for C/D use the RF's raw per-pixel probability map (the only continuous
score the pipeline produces) -- pre-watershed -- so those two metrics
specifically DO measure the same thing as Table 2.

Methods A, B, E: unmodified algorithms from full_volume_run_v2.py (same
code used for the timing benchmark and verified to match the paper's
actual Table 1/3 published numbers) -- NOT this session's experimental
masking/gradient-relief variants in Methods_ABE/, which were never used to
produce any published result.

Methods C, D: imported directly, unmodified, from
ZoneAnalysis/pipeline_common.py and pipeline_method_d.py -- the exact code
that produced Table 1/3's real C/D numbers.

  - LOSO variant: for each slice, uses the fold model that held that exact
    slice out (Results/Models/rf_fold_0{1-6}.joblib), with a fold-specific
    intensity range recomputed via fold_intensity_range() (raw min/max
    over that fold's 5 training slices only -- not saved anywhere, so
    reproduced here from RandomForest_Validation.ipynb Cell 4's function
    of the same name). This is the generalization/held-out number.
  - combined variant: the single production model
    (ZoneAnalysis/Results/combined_model.joblib) used for the actual
    Table 1/3 results, evaluated on the same 6 slices it was trained on.
    This is the training-set number, expected to be optimistic.

Does not modify pipeline_common.py, pipeline_method_d.py, the notebook, or
any existing file -- only imports/reads.

Usage:
    python3 compute_pixel_metrics.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import (distance_transform_edt, label, maximum_filter,
                            gaussian_filter, binary_dilation, uniform_filter)
from skimage.morphology import remove_small_objects, ball, opening, closing
from skimage.segmentation import watershed
from skimage.filters import sobel
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                              precision_score, recall_score, roc_auc_score,
                              average_precision_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'ZoneAnalysis'))

from pipeline_common import (resolve_ct_path, load_tiff, load_particle_labels,  # noqa: E402
                              predict_probability_volume, segment_watershed)
from pipeline_method_d import segment_method_d, build_normalized_intensity_volume  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / 'Results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_ROOT / 'Results' / 'Models'
COMBINED_MODEL_PATH = PROJECT_ROOT / 'ZoneAnalysis' / 'Results' / 'combined_model.joblib'
COMBINED_RANGE_PATH = PROJECT_ROOT / 'ZoneAnalysis' / 'Results' / 'combined_intensity_range.joblib'

TARGET_SLICES = [151, 350, 500, 700, 800, 900]
HALF_WINDOW = 25
N_SLICES = 981

FOLD_FOR_SLICE = {151: 1, 350: 2, 500: 3, 700: 4, 800: 5, 900: 6}
FOLD_TRAIN_SLICES = {
    1: [350, 500, 700, 800, 900],
    2: [151, 500, 700, 800, 900],
    3: [151, 350, 700, 800, 900],
    4: [151, 350, 500, 800, 900],
    5: [151, 350, 500, 700, 900],
    6: [151, 350, 500, 700, 800],
}

VOXEL_UM = 19.8
MIN_DIAM_UM = 100.0
MIN_VOXELS = int((MIN_DIAM_UM / VOXEL_UM) ** 3 * (np.pi / 6))


# ============================================================
# Methods A, B, E -- unmodified, from full_volume_run_v2.py
# (same source used for the timing benchmark)
# ============================================================
OTSU_LOWER = 0.40
OTSU_UPPER = 0.85


def normalize_per_slice_abe(volume):
    out = np.empty(volume.shape, dtype=np.float32)
    for i in range(volume.shape[0]):
        arr = volume[i].astype(np.float32)
        amin, amax = arr.min(), arr.max()
        out[i] = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr)
    return out


def run_method_a(volume):
    binary = (volume > OTSU_LOWER) & (volume < OTSU_UPPER)
    binary = opening(binary, ball(1))
    binary = closing(binary, ball(1))
    dist = distance_transform_edt(binary)
    dist_max = maximum_filter(dist, size=15)
    peak_mask = (dist == dist_max) & (dist > 3) & binary
    markers, _ = label(peak_mask)
    labeled = watershed(-dist, markers, mask=binary)
    return remove_small_objects(labeled, min_size=MIN_VOXELS)


def run_method_b(volume):
    binary = (volume > OTSU_LOWER) & (volume < OTSU_UPPER)
    gx = sobel(volume, axis=2)
    gy = sobel(volume, axis=1)
    gz = sobel(volume, axis=0)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)
    dist = distance_transform_edt(binary)
    combined = 0.6 * grad + 0.4 * (1.0 / (dist + 1.0))
    smoothed = gaussian_filter(volume, sigma=2)
    smooth_max = maximum_filter(smoothed * binary, size=13)
    peak_mask = ((smoothed * binary == smooth_max) &
                 (smoothed > OTSU_LOWER + 0.08) & binary)
    if peak_mask.sum() < 100:
        peak_mask = ((smoothed * binary == smooth_max) &
                     (smoothed > OTSU_LOWER + 0.04) & binary)
    markers, _ = label(peak_mask)
    labeled = watershed(combined, markers, mask=binary)
    return remove_small_objects(labeled, min_size=MIN_VOXELS)


def run_method_e(volume):
    binary = (volume > OTSU_LOWER) & (volume < OTSU_UPPER)
    min_radius = max(3, int((MIN_DIAM_UM / VOXEL_UM) / 2))
    opened = opening(binary, ball(min_radius))
    reconstructed = opened.copy()
    for _ in range(50):
        expanded = binary_dilation(reconstructed, structure=ball(1))
        new = expanded & binary
        if np.array_equal(new, reconstructed):
            break
        reconstructed = new
    dist = distance_transform_edt(reconstructed)
    dist_max = maximum_filter(dist, size=13)
    peak_mask = (dist == dist_max) & (dist > 2) & reconstructed
    markers, _ = label(peak_mask)
    labeled = watershed(-dist, markers, mask=reconstructed)
    min_vox_e = int((150.0 / VOXEL_UM) ** 3 * (np.pi / 6))
    return remove_small_objects(labeled, min_size=min_vox_e)


# ============================================================
# Helpers
# ============================================================
def load_window(start, end):
    return np.stack([load_tiff(resolve_ct_path(n)) for n in range(start, end)], axis=0)


def fold_intensity_range(train_slice_numbers):
    """Reproduces RandomForest_Validation.ipynb Cell 4's fold_intensity_range():
    raw min/max over a fold's training slices only, never the held-out slice."""
    los, his = [], []
    for n in train_slice_numbers:
        arr = load_tiff(resolve_ct_path(n)).astype(np.float32)
        los.append(float(arr.min()))
        his.append(float(arr.max()))
    return min(los), max(his)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Reproduces RandomForest_Validation.ipynb Cell 11's compute_metrics()."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    if y_prob is not None:
        y_prob = y_prob.ravel()
        try:
            roc_val = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_val = np.nan
        try:
            pr_auc_val = average_precision_score(y_true, y_prob)
        except ValueError:
            pr_auc_val = np.nan
    else:
        roc_val = np.nan
        pr_auc_val = np.nan
    return {
        'roc_auc': roc_val,
        'pr_auc': pr_auc_val,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'n_pixels': int(len(y_true)),
    }


# ============================================================
# Main
# ============================================================
def main():
    print('Loading ground truth (closed-world, matches Table 2) for 6 annotated slices...')
    gt = {n: load_particle_labels(n) for n in TARGET_SLICES}
    for n, g in gt.items():
        print(f'  slice {n}: {g.shape}, {int((g == 1).sum())} particle px / {g.size} total')

    rows = []

    # ---- A, B, E ----
    for slice_num in TARGET_SLICES:
        start = max(0, slice_num - HALF_WINDOW)
        end = min(N_SLICES, slice_num + HALF_WINDOW + 1)
        idx = slice_num - start
        print(f'\n=== A/B/E, slice {slice_num} (window {start}-{end - 1}) ===')
        raw_window = load_window(start, end)
        vol_abe = normalize_per_slice_abe(raw_window)
        for name, fn in [('A', run_method_a), ('B', run_method_b), ('E', run_method_e)]:
            labeled = fn(vol_abe)
            pred = (labeled[idx] > 0).astype(np.int8)
            m = compute_metrics(gt[slice_num], pred, y_prob=None)
            rows.append({'method': name, 'variant': 'single', 'slice': slice_num, **m})
            print(f"  {name}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"f1={m['f1']:.3f} mcc={m['mcc']:.3f}")

    # ---- C, D: LOSO fold models ----
    for slice_num in TARGET_SLICES:
        fold_id = FOLD_FOR_SLICE[slice_num]
        print(f'\n=== C/D LOSO, slice {slice_num} (fold {fold_id} held out) ===')
        rf = joblib.load(MODELS_DIR / f'rf_fold_{fold_id:02d}.joblib')
        intensity_range = fold_intensity_range(FOLD_TRAIN_SLICES[fold_id])

        start = max(0, slice_num - HALF_WINDOW)
        end = min(N_SLICES, slice_num + HALF_WINDOW + 1)
        idx = slice_num - start
        raw_window = load_window(start, end)

        prob_volume = predict_probability_volume(raw_window, rf, intensity_range)
        prob_slice = prob_volume[idx]

        labeled_c = segment_watershed(prob_volume)
        pred_c = (labeled_c[idx] > 0).astype(np.int8)
        m_c = compute_metrics(gt[slice_num], pred_c, y_prob=prob_slice)
        rows.append({'method': 'C', 'variant': 'LOSO', 'slice': slice_num, **m_c})

        intensity_volume = build_normalized_intensity_volume(raw_window, intensity_range)
        labeled_d, _ = segment_method_d(prob_volume, intensity_volume)
        pred_d = (labeled_d[idx] > 0).astype(np.int8)
        m_d = compute_metrics(gt[slice_num], pred_d, y_prob=prob_slice)
        rows.append({'method': 'D', 'variant': 'LOSO', 'slice': slice_num, **m_d})

        print(f"  C: precision={m_c['precision']:.3f} recall={m_c['recall']:.3f} "
              f"f1={m_c['f1']:.3f} mcc={m_c['mcc']:.3f} roc_auc={m_c['roc_auc']:.3f}")
        print(f"  D: precision={m_d['precision']:.3f} recall={m_d['recall']:.3f} "
              f"f1={m_d['f1']:.3f} mcc={m_d['mcc']:.3f} roc_auc={m_d['roc_auc']:.3f}")

    # ---- C, D: combined production model ----
    print('\n=== C/D combined production model (Table 1/3 model) ===')
    rf_combined = joblib.load(COMBINED_MODEL_PATH)
    intensity_range_combined = joblib.load(COMBINED_RANGE_PATH)
    for slice_num in TARGET_SLICES:
        start = max(0, slice_num - HALF_WINDOW)
        end = min(N_SLICES, slice_num + HALF_WINDOW + 1)
        idx = slice_num - start
        raw_window = load_window(start, end)

        prob_volume = predict_probability_volume(raw_window, rf_combined, intensity_range_combined)
        prob_slice = prob_volume[idx]

        labeled_c = segment_watershed(prob_volume)
        pred_c = (labeled_c[idx] > 0).astype(np.int8)
        m_c = compute_metrics(gt[slice_num], pred_c, y_prob=prob_slice)
        rows.append({'method': 'C', 'variant': 'combined', 'slice': slice_num, **m_c})

        intensity_volume = build_normalized_intensity_volume(raw_window, intensity_range_combined)
        labeled_d, _ = segment_method_d(prob_volume, intensity_volume)
        pred_d = (labeled_d[idx] > 0).astype(np.int8)
        m_d = compute_metrics(gt[slice_num], pred_d, y_prob=prob_slice)
        rows.append({'method': 'D', 'variant': 'combined', 'slice': slice_num, **m_d})

        print(f"  slice {slice_num}  C: f1={m_c['f1']:.3f} mcc={m_c['mcc']:.3f}  |  "
              f"D: f1={m_d['f1']:.3f} mcc={m_d['mcc']:.3f}")

    # ---- Save per-slice ----
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'per_slice_metrics.csv', index=False)
    print(f'\nSaved per-slice metrics: {OUT_DIR / "per_slice_metrics.csv"}')

    # ---- Aggregate: median [min, max] across the 6 slices ----
    agg_rows = []
    for (method, variant), grp in df.groupby(['method', 'variant'], sort=False):
        row = {'method': method, 'variant': variant, 'n_slices': len(grp)}
        for metric in ['precision', 'recall', 'f1', 'mcc', 'roc_auc', 'pr_auc']:
            vals = grp[metric].dropna().values
            if len(vals) == 0:
                row[f'{metric}_median'] = np.nan
                row[f'{metric}_min'] = np.nan
                row[f'{metric}_max'] = np.nan
            else:
                row[f'{metric}_median'] = float(np.median(vals))
                row[f'{metric}_min'] = float(np.min(vals))
                row[f'{metric}_max'] = float(np.max(vals))
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / 'aggregated_metrics.csv', index=False)

    print('\n' + '=' * 70)
    print('AGGREGATED (median [min, max] across 6 slices)')
    print('=' * 70)
    print(agg_df.to_string(index=False))
    print(f'\nSaved: {OUT_DIR / "aggregated_metrics.csv"}')


if __name__ == '__main__':
    main()
