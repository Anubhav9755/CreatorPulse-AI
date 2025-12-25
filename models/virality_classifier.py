# models/virality_classifier.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Keep feature names aligned with performance_predictor
NUMERIC_FEATURES = [
    "subscriber_count",
    "upload_hour",
    "day_of_week",
    "video_duration_sec",
    "title_length",
    "thumbnail_text_length",
    "sentiment_score",
    "like_ratio_24h",
    "comment_ratio_24h",
    "engagement_velocity_24h",
    "retention_avg_pct",
    "retention_p25_pct",
    "retention_p50_pct",
    "retention_p75_pct",
]

EMBEDDING_FEATURES = [f"topic_emb_{i}" for i in range(16)]

CATEGORICAL_FEATURES = ["niche", "country", "topic"]

TARGET = "virality_label"


@dataclass
class ViralityClassifierConfig:
    numeric_features: List[str]
    embedding_features: List[str]
    categorical_features: List[str]
    target: str = TARGET
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 0.8
    random_state: int = 42


class ViralityClassifier:
    """
    Tree-based virality classifier with feature importance for explainability.

    Training target:
      - virality_label (0/1)
    """

    def __init__(self, config: Optional[ViralityClassifierConfig] = None):
        self.config = config or ViralityClassifierConfig(
            numeric_features=NUMERIC_FEATURES,
            embedding_features=EMBEDDING_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )

        self.pipeline: Optional[Pipeline] = None
        self._feature_names_out: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Preprocessing + model
    # ------------------------------------------------------------------
    def _build_pipeline(self, df: pd.DataFrame) -> Pipeline:
        cfg = self.config

        numeric_features = cfg.numeric_features + cfg.embedding_features
        categorical_features = cfg.categorical_features

        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        clf = GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            random_state=cfg.random_state,
        )

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        # Fit once to populate feature names (for importance later)
        pipe.fit(df[numeric_features + categorical_features], df[cfg.target])

        # Extract feature names after preprocessing
        col_trans: ColumnTransformer = pipe.named_steps["preprocessor"]  # type: ignore
        feature_names: List[str] = []

        # numeric
        num_features_out = col_trans.named_transformers_["num"][
            "scaler"
        ].get_feature_names_out(numeric_features)
        feature_names.extend(num_features_out.tolist())

        # categorical
        ohe: OneHotEncoder = col_trans.named_transformers_["cat"]["onehot"]  # type: ignore
        cat_features_out = ohe.get_feature_names_out(categorical_features)
        feature_names.extend(cat_features_out.tolist())

        self._feature_names_out = feature_names

        return pipe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit the virality classifier.

        Returns a metrics dict with optional validation AUC.
        """
        cfg = self.config

        numeric_features = cfg.numeric_features + cfg.embedding_features
        categorical_features = cfg.categorical_features

        # Build and fit pipeline
        self.pipeline = self._build_pipeline(train_df)

        metrics: Dict[str, Any] = {}

        # Training AUC
        y_train = train_df[cfg.target].values
        y_train_pred = self.pipeline.predict_proba(
            train_df[numeric_features + categorical_features]
        )[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred)
        metrics["train_auc"] = float(train_auc)

        if val_df is not None:
            y_val = val_df[cfg.target].values
            y_val_pred = self.pipeline.predict_proba(
                val_df[numeric_features + categorical_features]
            )[:, 1]
            val_auc = roc_auc_score(y_val, y_val_pred)
            metrics["val_auc"] = float(val_auc)

        if verbose:
            print(f"[ViralityClassifier] train AUC: {metrics['train_auc']:.4f}")
            if "val_auc" in metrics:
                print(f"[ViralityClassifier]  val  AUC: {metrics['val_auc']:.4f}")

        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not trained or loaded.")

        cfg = self.config
        numeric_features = cfg.numeric_features + cfg.embedding_features
        categorical_features = cfg.categorical_features

        proba = self.pipeline.predict_proba(
            df[numeric_features + categorical_features]
        )[:, 1]
        return proba

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)

    def feature_importance(self, top_k: int = 20) -> pd.DataFrame:
        """
        Returns a DataFrame of global feature importance (top_k).
        """
        if self.pipeline is None or self._feature_names_out is None:
            raise RuntimeError("Model not trained or loaded.")

        clf: GradientBoostingClassifier = self.pipeline.named_steps["classifier"]  # type: ignore
        importances = clf.feature_importances_
        features = np.array(self._feature_names_out)

        df_imp = pd.DataFrame(
            {"feature": features, "importance": importances}
        ).sort_values("importance", ascending=False)

        return df_imp.head(top_k).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model pipeline + config + feature names
        joblib.dump(self.pipeline, path / "virality_pipeline.joblib")

        meta = {
            "config": asdict(self.config),
            "feature_names_out": self._feature_names_out,
        }
        joblib.dump(meta, path / "metadata.joblib")

    @classmethod
    def load(cls, path: str | Path) -> "ViralityClassifier":
        path = Path(path)

        meta = joblib.load(path / "metadata.joblib")
        cfg_dict = meta["config"]
        feature_names_out = meta["feature_names_out"]

        config = ViralityClassifierConfig(**cfg_dict)
        pipeline = joblib.load(path / "virality_pipeline.joblib")

        obj = cls(config=config)
        obj.pipeline = pipeline
        obj._feature_names_out = feature_names_out
        return obj
