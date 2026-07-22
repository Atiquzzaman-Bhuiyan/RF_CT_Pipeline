"""Train the single combined RF model (all 6 annotated slices) and save it + the
training-derived intensity range to disk, so zone/full-volume scripts can load
it without retraining."""
import time
import joblib
from pathlib import Path
from pipeline_common import train_combined_model

OUT_DIR = Path(__file__).parent / 'Results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print('Training combined RF on all 6 annotated slices (151, 350, 500, 700, 800, 900)...')
    rf, intensity_range = train_combined_model()
    print(f'Training time: {time.time() - t0:.1f}s')
    joblib.dump(rf, OUT_DIR / 'combined_model.joblib')
    joblib.dump(intensity_range, OUT_DIR / 'combined_intensity_range.joblib')
    print(f'Saved model to {OUT_DIR / "combined_model.joblib"}')
    print(f'Intensity range: {intensity_range}')


if __name__ == '__main__':
    main()
