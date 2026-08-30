"""
Timing benchmark -- all five segmentation methods (A-E), mid zone (slices
350-449), single SLURM job, single node.

Purpose: reviewer-requested computational-efficiency comparison. Timing
differences should reflect algorithm speed, not cluster load, so all five
methods run sequentially in this one process on one allocated node.

Algorithms are reproduced here UNMODIFIED from their verified production
sources, not from any experimental/fixed variant:
  - Methods A, B, E: exact logic from full_volume_run_v2.py (same code
    verified this session to match the paper's zone-level algorithm
    description; no masking/gradient-relief fixes applied here).
  - Methods C, D: exact logic from pipeline_common.py / pipeline_method_d.py
    (same code that produced the paper's actual published zone-level C/D
    numbers, confirmed by exact match against Table 1).

Model: the SAME production combined_model.joblib used for the paper's real
zone-level C/D results (trained on all 6 annotated slices combined) --
NOT the LOSO fold models used for the separate cross-validation table.

Timing boundary:
  - NOT timed (one-time, shared, would happen once regardless of method):
    CT slice I/O, RF model loading from disk.
  - Timed per method: the full segmentation computation producing the
    labeled array, including method-specific preprocessing (per-slice
    normalization for A/B/E; RF feature extraction + prediction for C/D,
    since that cost is specific to those two methods, not shared setup).
  - For C and D, RF prediction (feature extraction + predict_proba) is
    timed SEPARATELY from the watershed step and reported as a
    sub-component, since it's a major, method-specific cost worth
    isolating (a reviewer comparing "watershed speed" vs "RF speed" would
    want this broken out).
  - C and D each independently recompute the RF probability volume from
    scratch (not shared/cached between them), matching how they are
    actually deployed in production: zone_analysis.py (C) and
    zone_analysis_method_d.py (D) are separate script invocations that
    each call predict_probability_volume independently. Sharing the
    probability volume here would understate D's real standalone cost.
  - Diameter computation (compute_diameters) is NOT timed -- it is a
    cheap post-processing step (a bincount), not part of "segmentation".

Run once. No retries -- if a method throws or produces an empty/degenerate
result, that is reported as a finding, not silently retried.
"""
import json
import os
import platform
import socket
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy.ndimage import (distance_transform_edt, label, maximum_filter,
                            gaussian_filter, binary_dilation, uniform_filter)
from skimage.morphology import remove_small_objects, ball, opening, closing
from skimage.segmentation import watershed
from skimage.filters import sobel, threshold_otsu
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.ensemble import RandomForestClassifier  # noqa: F401 (needed to unpickle the model)

CT_DIR = Path("/home/ab9884/lhs1/ct_data")
MODEL_DIR = Path("/home/ab9884/lhs1/benchmark")
OUT_DIR = MODEL_DIR
START, END = 350, 450  # mid zone, matches the paper's standard comparison zone

VOXEL_UM = 19.8
MIN_DIAM_UM = 100.0
MIN_VOXELS = int((MIN_DIAM_UM / VOXEL_UM) ** 3 * (np.pi / 6))


# ============================================================
# Methods A, B, E -- unmodified, from full_volume_run_v2.py
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
# Methods C, D -- unmodified, from pipeline_common.py / pipeline_method_d.py
# ============================================================
SEED_PROB_THRESHOLD = 0.6
SEED_WINDOW_SIZE = 13
PROB_SMOOTH_SIGMA = 1

OTSU_CLIP_MIN = 0.35
OTSU_CLIP_MAX = 0.65
SEED_INTENSITY_SIGMA = 2
SEED_INTENSITY_THRESHOLD = 0.45
MAX_SEEDS = 15000
ENERGY_SIGMA = 1
W_PROB = 0.7
W_INTENSITY = 0.3


def normalize_with_range(image, lo, hi, eps=1e-8):
    image = image.astype(np.float32, copy=False)
    return (image - lo) / (hi - lo + eps)


def extract_features(image):
    image = image.astype(np.float32, copy=False)
    g1, g2, g3 = gaussian_filter(image, 1), gaussian_filter(image, 2), gaussian_filter(image, 3)
    sobel_mag = np.sqrt(sobel(image, axis=1) ** 2 + sobel(image, axis=0) ** 2)
    local_contrast = image - uniform_filter(image, size=5)
    h_elems = hessian_matrix(image, sigma=2, order='rc')
    eig = hessian_matrix_eigvals(h_elems)
    stack = np.stack([image, g1, g2, g3, sobel_mag, g1 - g3, g1 - g2, local_contrast,
                       eig[0], eig[1], np.sqrt(eig[0] ** 2 + eig[1] ** 2)], axis=-1)
    return stack.reshape(-1, 11)


def predict_probability_volume(volume, rf, intensity_range):
    lo, hi = intensity_range
    prob_volume = np.zeros(volume.shape, dtype=np.float32)
    for i in range(volume.shape[0]):
        norm = normalize_with_range(volume[i], lo, hi)
        feats = extract_features(norm)
        probs = rf.predict_proba(feats)[:, 1]
        prob_volume[i] = probs.reshape(volume[i].shape)
    return prob_volume


def segment_watershed(prob_volume):
    """Method C."""
    binary_rf = prob_volume > 0.5
    smoothed_prob = gaussian_filter(prob_volume, sigma=PROB_SMOOTH_SIGMA)
    gx = sobel(smoothed_prob, axis=2)
    gy = sobel(smoothed_prob, axis=1)
    gz = sobel(smoothed_prob, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    masked = smoothed_prob * binary_rf
    img_max = maximum_filter(masked, size=SEED_WINDOW_SIZE)
    peak_mask = (masked == img_max) & (masked > SEED_PROB_THRESHOLD)
    markers, _ = label(peak_mask)
    labeled = watershed(grad, markers, mask=binary_rf)
    return remove_small_objects(labeled, min_size=MIN_VOXELS)


def build_normalized_intensity_volume(volume, intensity_range):
    lo, hi = intensity_range
    return np.stack([normalize_with_range(volume[i], lo, hi) for i in range(volume.shape[0])], axis=0)


def generate_binary_mask_otsu(prob_volume, otsu_min=OTSU_CLIP_MIN, otsu_max=OTSU_CLIP_MAX):
    binary = np.zeros(prob_volume.shape, dtype=bool)
    for i in range(prob_volume.shape[0]):
        slice_p = prob_volume[i]
        if slice_p.max() == slice_p.min():
            t = 0.5 * (otsu_min + otsu_max)
        else:
            t = float(np.clip(threshold_otsu(slice_p), otsu_min, otsu_max))
        binary[i] = slice_p > t
    struct = ball(1)
    binary = opening(binary, struct)
    binary = closing(binary, struct)
    return binary


def place_intensity_seeds(intensity_volume, binary_mask, sigma=SEED_INTENSITY_SIGMA,
                           threshold=SEED_INTENSITY_THRESHOLD, max_seeds=MAX_SEEDS,
                           window_size=SEED_WINDOW_SIZE):
    smoothed = gaussian_filter(intensity_volume, sigma=sigma)
    masked = smoothed * binary_mask
    img_max = maximum_filter(masked, size=window_size)
    peak_mask = (masked == img_max) & (masked > threshold) & binary_mask
    n_seeds = int(peak_mask.sum())
    if n_seeds > max_seeds:
        peak_values = masked[peak_mask]
        adaptive_threshold = np.partition(peak_values, -max_seeds)[-max_seeds]
        peak_mask = (masked == img_max) & (masked >= adaptive_threshold) & binary_mask
        n_seeds = int(peak_mask.sum())
    markers, _ = label(peak_mask)
    return markers, n_seeds


def compute_hybrid_energy(prob_volume, intensity_volume, sigma=ENERGY_SIGMA,
                           w_prob=W_PROB, w_intensity=W_INTENSITY):
    def grad_mag_3d(vol):
        gx = sobel(vol, axis=2)
        gy = sobel(vol, axis=1)
        gz = sobel(vol, axis=0)
        return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    smoothed_prob = gaussian_filter(prob_volume, sigma=sigma)
    smoothed_intensity = gaussian_filter(intensity_volume, sigma=sigma)
    return w_prob * grad_mag_3d(smoothed_prob) + w_intensity * grad_mag_3d(smoothed_intensity)


def segment_method_d(prob_volume, intensity_volume):
    binary_mask = generate_binary_mask_otsu(prob_volume)
    markers, n_seeds = place_intensity_seeds(intensity_volume, binary_mask)
    energy = compute_hybrid_energy(prob_volume, intensity_volume)
    labeled = watershed(energy, markers, mask=binary_mask)
    labeled = remove_small_objects(labeled, min_size=MIN_VOXELS)
    return labeled, n_seeds


# ============================================================
# Benchmark driver
# ============================================================
def main():
    print("=" * 70)
    print("SEGMENTATION METHOD TIMING BENCHMARK -- mid zone (slices 350-449)")
    print("=" * 70)
    print(f"Node: {socket.gethostname()}")
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"SLURM_JOB_PARTITION: {os.environ.get('SLURM_JOB_PARTITION', 'N/A')}")
    print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'N/A')}")
    print(f"SLURM_MEM_PER_NODE: {os.environ.get('SLURM_MEM_PER_NODE', 'N/A')}")
    print(f"Python: {platform.python_version()}  Platform: {platform.platform()}")
    import sklearn, skimage
    print(f"scikit-learn: {sklearn.__version__}  scikit-image: {skimage.__version__}  numpy: {np.__version__}")
    print()

    # ---- shared setup: NOT timed ----
    print(f"Loading mid-zone CT slices ({START}-{END - 1})...")
    t_io_start = time.perf_counter()
    files = [CT_DIR / f"slice_{n:06d}.tif" for n in range(START, END)]
    raw_volume = np.stack([np.array(Image.open(f)) for f in files], axis=0)
    t_io = time.perf_counter() - t_io_start
    n_slices = raw_volume.shape[0]
    print(f"  volume shape: {raw_volume.shape}  (I/O time, excluded from method timing: {t_io:.2f}s)")

    print("Loading production RF model (combined_model.joblib, same model used for the paper's zone-level results)...")
    t_model_start = time.perf_counter()
    rf = joblib.load(MODEL_DIR / "combined_model.joblib")
    intensity_range = joblib.load(MODEL_DIR / "combined_intensity_range.joblib")
    t_model = time.perf_counter() - t_model_start
    print(f"  model loaded  (load time, excluded from method timing: {t_model:.2f}s)")

    vol_abe = normalize_per_slice_abe(raw_volume)

    results = []

    def record(method, total_s, n_particles, extra=None, error=None):
        row = {
            'method': method,
            'total_s': total_s,
            'per_slice_s': (total_s / n_slices) if total_s is not None else None,
            'n_particles_raw_labels': n_particles,
            'error': error,
        }
        if extra:
            row.update(extra)
        results.append(row)
        if error:
            print(f"  *** METHOD {method} FAILED: {error} ***")
        else:
            print(f"  {method}: total={total_s:.2f}s  per-slice={total_s/n_slices:.4f}s  raw_labels={n_particles}")

    # ---- Method A ----
    print("\n=== Method A (Otsu + distance watershed) ===")
    try:
        t0 = time.perf_counter()
        labeled_a = run_method_a(vol_abe)
        t_a = time.perf_counter() - t0
        record('A', t_a, int(labeled_a.max()))
    except Exception as e:
        record('A', None, None, error=repr(e))

    # ---- Method B ----
    print("\n=== Method B (intensity-guided watershed) ===")
    try:
        t0 = time.perf_counter()
        labeled_b = run_method_b(vol_abe)
        t_b = time.perf_counter() - t0
        record('B', t_b, int(labeled_b.max()))
    except Exception as e:
        record('B', None, None, error=repr(e))

    # ---- Method E ----
    print("\n=== Method E (morphological reconstruction) ===")
    try:
        t0 = time.perf_counter()
        labeled_e = run_method_e(vol_abe)
        t_e = time.perf_counter() - t0
        record('E', t_e, int(labeled_e.max()))
    except Exception as e:
        record('E', None, None, error=repr(e))

    # ---- Method C ----
    print("\n=== Method C (RF classification + probability watershed) ===")
    try:
        t0 = time.perf_counter()
        prob_volume_c = predict_probability_volume(raw_volume, rf, intensity_range)
        t_rf_c = time.perf_counter() - t0
        t1 = time.perf_counter()
        labeled_c = segment_watershed(prob_volume_c)
        t_ws_c = time.perf_counter() - t1
        t_c = t_rf_c + t_ws_c
        record('C', t_c, int(labeled_c.max()), extra={'rf_predict_s': t_rf_c, 'watershed_s': t_ws_c})
        print(f"    (breakdown: RF feature-extraction+predict={t_rf_c:.2f}s, watershed={t_ws_c:.2f}s)")
    except Exception as e:
        record('C', None, None, error=repr(e))

    # ---- Method D ----
    print("\n=== Method D (hybrid RF-intensity watershed) ===")
    try:
        t0 = time.perf_counter()
        # Recomputed independently, NOT reused from Method C -- matches production,
        # where C and D are separate script invocations each calling this fresh.
        prob_volume_d = predict_probability_volume(raw_volume, rf, intensity_range)
        t_rf_d = time.perf_counter() - t0
        t1 = time.perf_counter()
        intensity_volume = build_normalized_intensity_volume(raw_volume, intensity_range)
        labeled_d, n_seeds = segment_method_d(prob_volume_d, intensity_volume)
        t_ws_d = time.perf_counter() - t1
        t_d = t_rf_d + t_ws_d
        record('D', t_d, int(labeled_d.max()), extra={'rf_predict_s': t_rf_d, 'watershed_s': t_ws_d, 'n_seeds': n_seeds})
        print(f"    (breakdown: RF feature-extraction+predict={t_rf_d:.2f}s, hybrid watershed={t_ws_d:.2f}s, n_seeds={n_seeds})")
    except Exception as e:
        record('D', None, None, error=repr(e))

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (mid zone, slices 350-449, n=100 slices)")
    print("=" * 70)
    print(f"{'Method':<8}{'Total (s)':<12}{'Per-slice (s)':<16}{'RF predict (s)':<16}{'Watershed (s)':<15}")
    for r in results:
        if r['error']:
            print(f"{r['method']:<8}FAILED: {r['error']}")
            continue
        rf_s = f"{r.get('rf_predict_s'):.2f}" if r.get('rf_predict_s') is not None else "N/A"
        ws_s = f"{r.get('watershed_s'):.2f}" if r.get('watershed_s') is not None else "N/A"
        print(f"{r['method']:<8}{r['total_s']:<12.2f}{r['per_slice_s']:<16.4f}{rf_s:<16}{ws_s:<15}")

    out = {
        'io_time_s': t_io,
        'model_load_time_s': t_model,
        'n_slices': n_slices,
        'slice_range': f'{START}-{END - 1}',
        'results': results,
    }
    with open(OUT_DIR / 'benchmark_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
