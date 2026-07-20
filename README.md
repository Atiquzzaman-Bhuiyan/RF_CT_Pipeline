# RF_CT_Pipeline

Random Forest pipeline scaffold for validation, feature extraction, training, evaluation, visualization, and export.

This version matches the CT TIFF workflow used in the Colab script:
- CT slices: `slice_000150.tif` style files in `ct_slices/`
- Annotation masks: `annotation_slice_0150_particle.tif` and `annotation_slice_0150_background.tif` in `annotations/`
- Feature extraction: 11 per-pixel features from the CT slice image

## Files
- `RandomForest_Validation.ipynb`: main notebook
- `config.py`: parameters and configuration
- `feature_extraction.py`: feature extraction helpers
- `dataset.py`: CT slice and annotation loading utilities
- `validation.py`: validation strategies
- `training.py`: RF training entry point
- `evaluation.py`: evaluation metrics helpers
- `visualization.py`: plotting helpers
- `export.py`: result export helpers
- `utils.py`: shared utilities

## Next steps
1. Put CT slices in `ct_slices/` and annotation masks in `annotations/`.
2. Run the notebook from the project folder so local imports resolve correctly.
3. Adjust `slice_config` in `config.py` if you want different train/test/validate slices.
