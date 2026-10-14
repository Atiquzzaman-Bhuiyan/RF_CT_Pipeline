# GT validation: Methods A-E on slices 400-489

## Purpose
Instance-level validation of Methods A-E against a held-out manual ground truth
(Dragonfly, LHS-1 CT). The GT stack (`roi5`) has 100 slices; the first 90 are scored,
i.e. CT slices 400-489 (slice index 0 = CT slice 400, checked with `--check-offset`).

## Files
- `run_400_499.py`, `run_400_499.sh` - SLURM job that produced the method label TIFFs (Jubail).
  Settings: `OTSU_LOWER=0.475`, `CORE_END=490`, `ABE_ALSO_ORIGINAL=1`.
- `score_instances.py` - instance-level scoring. `overlay_check.py` - visual GT/method overlays.
- `results/{standard,oracle,roi,roi_oracle}/` - `summary.csv`, `foreground_agreement.csv`, `per_grain.csv`.

## Scoring commands (all with `--floors 67 227`; `--edge-ratio 0.2` and `--iou 0.5` are the script defaults)
```
# standard
python3 score_instances.py --gt "/Volumes/My Passport/roi5" --methods-dir ~/gt400_out --gt-slices 90 --out ~/gt400_scores --floors 67 227 --check-offset
# oracle
python3 score_instances.py --gt "/Volumes/My Passport/roi5" --methods-dir ~/gt400_out --gt-slices 90 --out ~/gt400_scores_oracle --floors 67 227 --oracle-fg
# roi
python3 score_instances.py --gt "/Volumes/My Passport/roi5" --methods-dir ~/gt400_out --gt-slices 90 --out ~/gt400_scores_roi --floors 67 227 --roi auto --check-offset
# roi + oracle
python3 score_instances.py --gt "/Volumes/My Passport/roi5" --methods-dir ~/gt400_out --gt-slices 90 --out ~/gt400_scores_roi_oracle --floors 67 227 --roi auto --oracle-fg
```
Note: `results/standard` and `results/oracle` were produced with an earlier revision of
`score_instances.py` that had no `--roi` option and no `merged_true`/`merged_leak` columns.
`results/roi` and `results/roi_oracle` were produced with the revision committed here.

## Parameters
- Voxel size 19.8 um. Size floors: 67 voxels (100 um) and 227 voxels (150 um).
- Methods A/B/E: `normalize_volume` + `sample_mask` (Methods_ABE) with `OTSU_LOWER=0.475`,
  calibrated on annotated slices 350/700/800, not on 400-489. `_orig` variants are the superseded
  min-max pipeline, saved for comparison.
- Methods C/D: combined RF model. The processing window is capped at slice 499, so training
  slice 500 is never loaded.

## Rules
- Edge grains (cross-section ratio > 0.2 on the first/last slice, or touching the frame border)
  are excluded from size statistics only, not from matching.
- `precision_lb` is a lower bound because the GT is incomplete; recall is the primary metric.
- `--oracle-fg` runs mask method labels to the GT foreground. They are a diagnostic upper bound,
  not headline results.
- `--roi auto`: score only inside the disc where the GT was traced. The radius is derived from
  the GT alone and applied identically to the GT and every method.

## Data
Not in this repo (GT stack `roi5` and method label TIFFs). Data: TODO (Zenodo DOI)
