"""
Run Methods A-E on the held-out GT block (default slices 400-489) and save instance-label
volumes. Read-only reuse of the existing pipeline code -- nothing in the repo is modified.

Design choices (state these in the paper):
  * Scored core = slices 400..CORE_END-1 (default 400-489). The processing window is padded
    by 10 slices on each side but CAPPED AT SLICE 499, so slice 500 (an RF training slice)
    is never loaded. Outputs are cropped to the core AFTER segmentation.
  * C and D use the combined production model (the one behind the paper's Tables 1/3).
    None of its training slices (151, 350, 500, 700, 800, 900) lies in the window 390-499.
  * A, B, E (MAIN) use the corrected pipeline behind the paper's updated tables:
    Methods_ABE/methods_a_b_e.py with normalize_volume() (per-slice 0.01/99.99
    percentile) + sample_mask(), OTSU_LOWER=0.475 (env OTSU_LOWER), OTSU_UPPER=0.85,
    MIN_VOXELS=68; Method E native 150 um = 227 vox.
  * Method E is also run at the 100 um floor (68 vox) as E_100um.
  * Optional ABE_ALSO_ORIGINAL=1: superseded min-max A/B/E, saved as method_X_orig.
  * Optional RUN_FOLD3=1: C/D also with a leave-slice-500-out model, saved as C_fold3/D_fold3.

Env vars (set in the SLURM script):
  REPO_DIR, CT_DIR, OUT_DIR, OTSU_LOWER, CORE_END (default 490, exclusive)
  COMBINED_MODEL / COMBINED_RANGE  REQUIRED: combined_model.joblib, combined_intensity_range.joblib
  RUN_FOLD3, MODELS_DIR, ANN_DIR   only for the optional fold-3 run
"""
import os, sys, time
from pathlib import Path
import numpy as np
import joblib
import tifffile

REPO = Path(os.environ.get('REPO_DIR', Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO / 'ZoneAnalysis'))
sys.path.insert(0, str(REPO / 'PixelMetrics'))
sys.path.insert(0, str(REPO / 'Methods_ABE'))

import pipeline_common as pc
if os.environ.get('CT_DIR'):
    pc.CT_DIR = Path(os.environ['CT_DIR'])          # resolve_ct_path reads this global
from pipeline_common import predict_probability_volume, segment_watershed
from pipeline_method_d import segment_method_d, build_normalized_intensity_volume
import compute_pixel_metrics as cpm                 # helpers + original A/B/E, unmodified
import methods_a_b_e as abe                         # corrected A/B/E, unmodified
abe.OTSU_LOWER = float(os.environ.get('OTSU_LOWER', 0.475))   # module global read at call time

CORE_START = 400
CORE_END = int(os.environ.get('CORE_END', 490))     # exclusive -> core = 400..489
PAD = 10
TRAIN_SLICE_500 = 500
WIN_START = CORE_START - PAD
WIN_END = min(CORE_END + PAD, TRAIN_SLICE_500)      # never load slice 500 (RF training slice)
LO = CORE_START - WIN_START
HI = LO + (CORE_END - CORE_START)
assert HI <= WIN_END - WIN_START, 'core does not fit inside the window'
FOLD3_TRAIN = [151, 350, 700, 800, 900]
TAG = f'{CORE_START}_{CORE_END - 1}'

MODELS_DIR = Path(os.environ.get('MODELS_DIR', REPO / 'Results' / 'Models'))
OUT_DIR = Path(os.environ.get('OUT_DIR', 'gt400_out')); OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(name, labeled):
    core = labeled[LO:HI].astype(np.uint32)
    ids = np.unique(core); ids = ids[ids > 0]
    tifffile.imwrite(OUT_DIR / f'method_{name}_{TAG}.tif', core, compression='zlib')
    print(f'  saved method {name}: {len(ids)} labels in core, shape {core.shape}', flush=True)


def get_fold3_model():
    """Load rf_fold_03.joblib or retrain it with the pipeline's production recipe on the 5
    slices excluding 500 (optional robustness run only)."""
    from sklearn.ensemble import RandomForestClassifier
    path = MODELS_DIR / 'rf_fold_03.joblib'
    rng = cpm.fold_intensity_range(FOLD3_TRAIN)
    if path.exists():
        print('Using existing', path, flush=True)
        return joblib.load(path), rng
    print('rf_fold_03.joblib not found -> retraining on', FOLD3_TRAIN, flush=True)
    pc.ANN_DIR = Path(os.environ.get('ANN_DIR', REPO / 'annotations'))
    lo, hi = rng
    X, y = [], []
    for n in FOLD3_TRAIN:
        norm = pc.normalize_with_range(pc.load_tiff(pc.resolve_ct_path(n)), lo, hi)
        X.append(pc.extract_features(norm)); y.append(pc.load_particle_labels(n).ravel())
    X, y = np.vstack(X), np.concatenate(y)
    t = time.time(); rf = RandomForestClassifier(**pc.RF_PARAMS).fit(X, y)
    print(f'  trained on {len(y):,} px in {time.time() - t:.0f}s', flush=True)
    joblib.dump(rf, OUT_DIR / 'rf_fold_03_retrained.joblib')
    return rf, rng


def run_cd(raw, rf, rng, suffix):
    t = time.time(); print(f'RF probability volume ({suffix or "combined"}) ...', flush=True)
    prob = predict_probability_volume(raw, rf, rng); print(f'  {time.time() - t:.0f}s', flush=True)
    t = time.time(); print('Method C ...', flush=True)
    save('C' + suffix, segment_watershed(prob)); print(f'  {time.time() - t:.0f}s', flush=True)
    t = time.time(); print('Method D ...', flush=True)
    labeled_d, n_seeds = segment_method_d(prob, build_normalized_intensity_volume(raw, rng))
    save('D' + suffix, labeled_d); print(f'  seeds={n_seeds}, {time.time() - t:.0f}s', flush=True)


def main():
    t0 = time.time()
    print(f'Core slices {CORE_START}-{CORE_END - 1}; window {WIN_START}-{WIN_END - 1} '
          f'(slice 500 excluded: {WIN_END - 1 < TRAIN_SLICE_500})', flush=True)
    raw = cpm.load_window(WIN_START, WIN_END)
    print('  shape', raw.shape, flush=True)

    # ---- A, B, E: corrected pipeline (MAIN) ----
    print(f'A/B/E corrected: OTSU_LOWER={abe.OTSU_LOWER}, OTSU_UPPER={abe.OTSU_UPPER}', flush=True)
    vol_fix = abe.normalize_volume(raw)
    for name, fn in [('A', abe.run_method_a), ('B', abe.run_method_b), ('E', abe.run_method_e)]:
        t = time.time(); print(f'Method {name} ...', flush=True)
        save(name, fn(vol_fix)); print(f'  {time.time() - t:.0f}s', flush=True)

    t = time.time(); print('Method E (100 um floor) ...', flush=True)
    orig_rso = abe.remove_small_objects
    abe.remove_small_objects = lambda lab, min_size: orig_rso(lab, min_size=abe.MIN_VOXELS)
    try:
        save('E_100um', abe.run_method_e(vol_fix))
    finally:
        abe.remove_small_objects = orig_rso
    print(f'  {time.time() - t:.0f}s', flush=True)

    if os.environ.get('ABE_ALSO_ORIGINAL') == '1':
        vol_orig = cpm.normalize_per_slice_abe(raw)
        for name, fn in [('A', cpm.run_method_a), ('B', cpm.run_method_b), ('E', cpm.run_method_e)]:
            t = time.time(); print(f'Original Method {name} ...', flush=True)
            save(name + '_orig', fn(vol_orig)); print(f'  {time.time() - t:.0f}s', flush=True)

    # ---- C, D: combined production model (MAIN) ----
    cm, cr = os.environ['COMBINED_MODEL'], os.environ['COMBINED_RANGE']
    run_cd(raw, joblib.load(cm), joblib.load(cr), '')

    # ---- optional: leave-slice-500-out model ----
    if os.environ.get('RUN_FOLD3') == '1':
        rf3, rng3 = get_fold3_model()
        run_cd(raw, rf3, rng3, '_fold3')

    print(f'Done in {(time.time() - t0) / 60:.1f} min -> {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
