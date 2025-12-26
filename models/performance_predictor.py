# models/performance_predictor.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================
# Feature definitions
# =========================

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

TARGET_REGRESSION = ["views_24h", "views_7d", "engagement_velocity_24h"]
TARGET_CLASSIFICATION = "virality_label"


# =========================
# Config dataclass
# =========================

@dataclass
class PerformancePredictorConfig:
    numeric_features: List[str]
    embedding_features: List[str]
    categorical_features: List[str]
    target_regression: List[str]
    target_classification: str
    learning_rate: float = 1e-3
    hidden_units_num: Tuple[int, ...] = (128, 64)
    hidden_units_emb: Tuple[int, ...] = (64,)
    hidden_units_combined: Tuple[int, ...] = (128, 64)
    dropout_rate: float = 0.2


# =========================
# Main class
# =========================

class PerformancePredictor:
    """
    Multi-task model:
      - Regression: views_24h, views_7d, engagement_velocity_24h
      - Classification: virality_label (0/1)

    Uses:
      - sklearn ColumnTransformer for preprocessing
      - TensorFlow Keras model for training & inference
    """

    def __init__(self, config: Optional[PerformancePredictorConfig] = None):
        self.config = config or PerformancePredictorConfig(
            numeric_features=NUMERIC_FEATURES,
            embedding_features=EMBEDDING_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            target_regression=TARGET_REGRESSION,
            target_classification=TARGET_CLASSIFICATION,
        )

        self.preprocessor: Optional[ColumnTransformer] = None
        self.model: Optional[keras.Model] = None
        self._feature_names_out: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        numeric_features = self.config.numeric_features + self.config.embedding_features
        categorical_features = self.config.categorical_features

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

        preprocessor.fit(df[numeric_features + categorical_features])

        if hasattr(preprocessor, "get_feature_names_out"):
            self._feature_names_out = preprocessor.get_feature_names_out().tolist()
        else:
            self._feature_names_out = None

        return preprocessor

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not fitted or loaded.")

        numeric_features = self.config.numeric_features + self.config.embedding_features
        categorical_features = self.config.categorical_features
        X = self.preprocessor.transform(df[numeric_features + categorical_features])
        return X.astype(np.float32)

    def _extract_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        y_reg = df[self.config.target_regression].values.astype(np.float32)
        y_cls = df[self.config.target_classification].values.astype(np.float32)
        return y_reg, y_cls

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def _build_model(self, input_dim: int) -> keras.Model:
        cfg = self.config

        inputs = keras.Input(shape=(input_dim,), name="features")
        x = inputs

        for units in cfg.hidden_units_combined:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(cfg.dropout_rate)(x)

        reg_output = layers.Dense(
            len(cfg.target_regression),
            name="regression_head"
        )(x)

        cls_x = layers.Dense(64, activation="relu")(x)
        cls_x = layers.Dropout(cfg.dropout_rate)(cls_x)
        cls_output = layers.Dense(
            1,
            activation="sigmoid",
            name="classification_head"
        )(cls_x)

        model = keras.Model(
            inputs=inputs,
            outputs=[reg_output, cls_output],
            name="performance_predictor",
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss={
                "regression_head": "mse",
                "classification_head": "binary_crossentropy",
            },
            metrics={
                "regression_head": [keras.metrics.MeanAbsoluteError(name="mae")],
                "classification_head": [keras.metrics.AUC(name="auc")],
            },
        )
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        batch_size: int = 256,
        epochs: int = 20,
        verbose: int = 1,
    ):
        self.preprocessor = self._build_preprocessor(train_df)

        X_train = self._transform_features(train_df)
        y_reg_train, y_cls_train = self._extract_targets(train_df)

        validation_data = None
        if val_df is not None:
            X_val = self._transform_features(val_df)
            y_reg_val, y_cls_val = self._extract_targets(val_df)
            validation_data = (
                X_val,
                {
                    "regression_head": y_reg_val,
                    "classification_head": y_cls_val,
                },
            )

        self.model = self._build_model(input_dim=X_train.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_regression_head_loss",
                patience=3,
                restore_best_weights=True,
            )
        ]

        self.model.fit(
            X_train,
            {
                "regression_head": y_reg_train,
                "classification_head": y_cls_train,
            },
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model not trained or loaded.")

        X = self._transform_features(df)
        reg_pred, cls_pred = self.model.predict(X, verbose=0)

        return {
            "views_24h_pred": reg_pred[:, 0],
            "views_7d_pred": reg_pred[:, 1],
            "engagement_velocity_24h_pred": reg_pred[:, 2],
            "virality_prob": cls_pred[:, 0],
        }

    # ------------------------------------------------------------------
    # Persistence (Keras 3 SAFE)
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        """
        Saves:
          - TensorFlow SavedModel (Keras 3 compatible)
          - sklearn preprocessor
          - config JSON
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # --- Save model as SavedModel ---
        model_path = path / "tf_model_saved"
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)

        self.model.export(str(model_path))

        # --- Save preprocessor ---
        import joblib
        joblib.dump(self.preprocessor, path / "preprocessor.joblib")

        # --- Save config ---
        with (path / "config.json").open("w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PerformancePredictor":
        path = Path(path)

        # Load config
        with (path / "config.json").open("r") as f:
            cfg_dict = json.load(f)
        config = PerformancePredictorConfig(**cfg_dict)

        # Load SavedModel (Keras 3 safe)
        model = keras.models.load_model(
            str(path / "tf_model_saved"),
            compile=False
        )

        # Load preprocessor
        import joblib
        preprocessor = joblib.load(path / "preprocessor.joblib")

        obj = cls(config=config)
        obj.model = model
        obj.preprocessor = preprocessor
        return obj
