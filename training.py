"""Random Forest training for the RF_CT_Pipeline project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config import PipelineConfig


@dataclass
class TrainedModel:
    model: RandomForestClassifier
    config: PipelineConfig
    feature_names: tuple[str, ...]
    is_fitted: bool = False

    def predict(self, features):
        return self.model.predict(features)

    def predict_proba(self, features):
        return self.model.predict_proba(features)


def train_random_forest(features, labels, config: PipelineConfig | None = None) -> TrainedModel:
    config = config or PipelineConfig()
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )
    model.fit(np.asarray(features), np.asarray(labels))
    return TrainedModel(model=model, config=config, feature_names=config.feature_names, is_fitted=True)
