"""Central configuration for the RF_CT_Pipeline project."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    ct_dir: Path = Path("ct_slices")
    ann_dir: Path = Path("annotations")
    output_dir: Path = Path("outputs")
    artifact_dir: Path = Path("outputs") / "artifacts"
    random_state: int = 42
    n_estimators: int = 150
    max_depth: int | None = 20
    min_samples_split: int = 2
    min_samples_leaf: int = 3
    n_jobs: int = -1
    validation_mode: str = "loso"
    fold_counts: tuple[int, ...] = field(default_factory=lambda: (5, 6))
    supported_extensions: tuple[str, ...] = (".tif", ".tiff")
    voxel_um: float = 19.8
    slice_config: dict[int, str] = field(
        default_factory=lambda: {
            150: "train",
            350: "train",
            800: "train",
            120: "test",
            450: "test",
            600: "validate",
        }
    )
    feature_names: tuple[str, ...] = (
        "raw_intensity",
        "gaussian_sigma_1",
        "gaussian_sigma_2",
        "gaussian_sigma_3",
        "sobel_magnitude",
        "log_sigma_1_minus_sigma_3",
        "dog_sigma_1_minus_sigma_2",
        "local_contrast",
        "hessian_lambda_max",
        "hessian_lambda_min",
        "hessian_frobenius",
    )

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
