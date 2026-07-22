"""
Shared pipeline for zone-level and full-volume particle segmentation.

Feature extraction and RF hyperparameters are ported directly from
RandomForest_Validation.ipynb (11 real feature channels, incl. actual Hessian
eigenvalues -- the old method_c*.py scripts had only 8 channels, one of which
was a zero placeholder never actually computed). Labeling uses the notebook's
final decision: USE_BACKGROUND_MASK=False, closed-world (particle mask -> 1,
everything else -> 0), trained as ONE combined model on all 6 annotated
slices (151, 350, 500, 700, 800, 900) rather than per-fold LOSO models --
this is the single production model used for all zone/full-volume inference
below.

The watershed segmentation stage (probability volume -> smoothed gradient ->
seeded watershed -> size filter -> diameter) is ported from
method_c_full_volume_final.py, same algorithm and parameters, since that part
was already validated in the user's prior work (zone-level output verified
against Dragonfly ground truth). Only the RF/feature stage changed.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import (gaussian_filter, uniform_filter, maximum_filter,
                            label as nd_label)
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import label as nd_label_filter
from sklearn.ensemble import RandomForestClassifier

# ---- Paths ------------------------------------------------------------
CT_DIR = Path("/Volumes/My Passport/ct_slices")
ANN_DIR = Path("/Volumes/My Passport/annotations")

# ---- Physical calibration (kept identical to the old method_c scripts) ----
VOXEL_UM = 19.8
MIN_DIAM_UM = 100.0
MIN_VOXELS = int((MIN_DIAM_UM / VOXEL_UM) ** 3 * (np.pi / 6))

# ---- RF config (from RandomForest_Validation.ipynb) --------------------
MIN_PARTICLE_SIZE = 5  # blob-cleaning threshold applied to particle masks
RF_PARAMS = dict(
    n_estimators=150, max_depth=20, min_samples_leaf=3, min_samples_split=2,
    class_weight='balanced', n_jobs=-1, random_state=42,
)
ANNOTATED_SLICES = [151, 350, 500, 700, 800, 900]
FEATURE_NAMES = ['raw_intensity', 'gaussian_sigma_1', 'gaussian_sigma_2', 'gaussian_sigma_3',
                  'sobel_magnitude', 'log_approx', 'dog', 'local_contrast',
                  'hessian_lambda_max', 'hessian_lambda_min', 'hessian_frobenius']

# ---- Watershed params (from method_c_full_volume_final.py, unchanged) ----
SEED_PROB_THRESHOLD = 0.6
SEED_WINDOW_SIZE = 13
PROB_SMOOTH_SIGMA = 1
BATCH_SIZE = 100
OVERLAP = 10


def resolve_ct_path(slice_num: int) -> Path:
    candidates = [
        CT_DIR / f'Copy of slice_{slice_num:06d}.tif',
        CT_DIR / f'slice_{slice_num:06d}.tif',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f'No CT file found for slice {slice_num}')


def resolve_ann_path(slice_num: int, kind: str) -> Path:
    for ext in ('tif', 'tiff'):
        c = ANN_DIR / f'annotation_slice_{slice_num:04d}_{kind}.{ext}'
        if c.exists():
            return c
    raise FileNotFoundError(f'No annotation_{kind} file found for slice {slice_num}')


def load_tiff(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def remove_small_blobs(mask: np.ndarray, min_size: int) -> np.ndarray:
    labeled, _ = nd_label(mask)
    counts = np.bincount(labeled.ravel())
    cleaned = mask.copy()
    for label_id, size in enumerate(counts):
        if label_id != 0 and size < min_size:
            cleaned[labeled == label_id] = False
    return cleaned


def load_particle_labels(slice_num: int) -> np.ndarray:
    """Closed-world labels for one annotated slice: particle_mask -> 1, else -> 0.
    Matches the notebook's USE_BACKGROUND_MASK=False final configuration."""
    particle_path = resolve_ann_path(slice_num, 'particle')
    particle = load_tiff(particle_path)
    if particle.dtype.kind == 'f':
        particle_mask = particle > 0.001
    else:
        particle_mask = particle > 127 if particle.max() > 1 else particle == 1
    particle_mask = remove_small_blobs(particle_mask, MIN_PARTICLE_SIZE)
    labels = np.zeros(particle_mask.shape, dtype=np.int8)
    labels[particle_mask] = 1
    return labels


def extract_features(image: np.ndarray) -> np.ndarray:
    """Identical to RandomForest_Validation.ipynb Cell 6 -- 11 real feature channels."""
    image = image.astype(np.float32, copy=False)
    g1, g2, g3 = gaussian_filter(image, 1), gaussian_filter(image, 2), gaussian_filter(image, 3)
    sobel_mag = np.sqrt(sobel(image, axis=1) ** 2 + sobel(image, axis=0) ** 2)
    local_contrast = image - uniform_filter(image, size=5)
    h_elems = hessian_matrix(image, sigma=2, order='rc')
    eig = hessian_matrix_eigvals(h_elems)
    stack = np.stack([image, g1, g2, g3, sobel_mag, g1 - g3, g1 - g2, local_contrast,
                       eig[0], eig[1], np.sqrt(eig[0] ** 2 + eig[1] ** 2)], axis=-1)
    return stack.reshape(-1, 11)


def normalize_with_range(image: np.ndarray, lo: float, hi: float, eps: float = 1e-8) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    return (image - lo) / (hi - lo + eps)


def compute_training_intensity_range():
    """Global min/max computed ONLY from the 6 annotated (training) slices' raw pixel values --
    leak-free by construction, since no zone/full-volume slice ever contributes to this range."""
    lo, hi = np.inf, -np.inf
    for n in ANNOTATED_SLICES:
        arr = load_tiff(resolve_ct_path(n)).astype(np.float32)
        lo = min(lo, float(arr.min()))
        hi = max(hi, float(arr.max()))
    return lo, hi


def train_combined_model():
    """Train ONE RF on all 6 annotated slices combined (not per-fold LOSO) -- the single
    production model used for every zone and the full volume."""
    lo, hi = compute_training_intensity_range()
    X_all, y_all = [], []
    for n in ANNOTATED_SLICES:
        raw = load_tiff(resolve_ct_path(n))
        norm = normalize_with_range(raw, lo, hi)
        feats = extract_features(norm)
        labels = load_particle_labels(n).ravel()
        X_all.append(feats)
        y_all.append(labels)
        print(f'  slice {n}: {int((labels == 1).sum())} particle px, {int((labels == 0).sum())} background px')
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f'Combined training set: {len(y):,} pixels ({int((y == 1).sum()):,} particle, {int((y == 0).sum()):,} background)')
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X, y)
    return rf, (lo, hi)


def predict_probability_volume(volume: np.ndarray, rf, intensity_range) -> np.ndarray:
    lo, hi = intensity_range
    prob_volume = np.zeros(volume.shape, dtype=np.float32)
    for i in range(volume.shape[0]):
        norm = normalize_with_range(volume[i], lo, hi)
        feats = extract_features(norm)
        probs = rf.predict_proba(feats)[:, 1]
        prob_volume[i] = probs.reshape(volume[i].shape)
    return prob_volume


def segment_watershed(prob_volume: np.ndarray) -> np.ndarray:
    """Identical algorithm to method_c_full_volume_final.py's segment_method_c_chunk."""
    binary_rf = prob_volume > 0.5
    smoothed_prob = gaussian_filter(prob_volume, sigma=PROB_SMOOTH_SIGMA)

    gx = sobel(smoothed_prob, axis=2)
    gy = sobel(smoothed_prob, axis=1)
    gz = sobel(smoothed_prob, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    masked = smoothed_prob * binary_rf
    img_max = maximum_filter(masked, size=SEED_WINDOW_SIZE)
    peak_mask = (masked == img_max) & (masked > SEED_PROB_THRESHOLD)

    markers, _ = nd_label(peak_mask)
    labeled = watershed(grad, markers, mask=binary_rf)
    labeled = remove_small_objects(labeled, min_size=MIN_VOXELS)
    return labeled


def compute_diameters(labeled: np.ndarray) -> np.ndarray:
    diams = []
    counts = np.bincount(labeled.ravel())
    for region_id, voxels in enumerate(counts):
        if region_id == 0:
            continue
        if voxels >= MIN_VOXELS:
            d = ((6 * voxels / np.pi) ** (1 / 3)) * VOXEL_UM
            diams.append(d)
    return np.array(diams)


def compute_diameters_central(labeled: np.ndarray, central_start: int, central_end: int) -> np.ndarray:
    """Only counts particles whose z-centroid falls in [central_start, central_end) --
    used for full-volume chunking to avoid double-counting particles spanning chunk seams."""
    diams = []
    for region_id in np.unique(labeled):
        if region_id == 0:
            continue
        coords = np.where(labeled == region_id)
        centroid_z = coords[0].mean()
        if not (central_start <= centroid_z < central_end):
            continue
        voxels = len(coords[0])
        if voxels >= MIN_VOXELS:
            d = ((6 * voxels / np.pi) ** (1 / 3)) * VOXEL_UM
            diams.append(d)
    return np.array(diams)
