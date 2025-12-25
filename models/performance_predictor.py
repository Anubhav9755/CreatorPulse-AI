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


class PerformancePredictor:
    """
    Multi-task model:
      - Regression: views_24h, views_7d, engagement_velocity_24h
      - Classification: virality_label (0/1)

    Uses:
      - sklearn ColumnTransformer for preprocessing (scaling + one-hot)
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

        # Cache feature order after preprocessing
        self._feature_names_out: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        numeric_features = self.config.numeric_features + self.config.embedding_features
        categorical_features = self.config.categorical_features

        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

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
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

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

        # For clarity, split numeric+emb vs cat is already done in preprocessing,
        # here we just feed the dense vector.
        x = inputs

        # Shared trunk
        for units in cfg.hidden_units_combined:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(cfg.dropout_rate)(x)

        # Regression head
        reg_output = layers.Dense(len(cfg.target_regression), name="regression_head")(x)

        # Classification head (binary)
        cls_x = x
        cls_x = layers.Dense(64, activation="relu")(cls_x)
        cls_x = layers.Dropout(cfg.dropout_rate)(cls_x)
        cls_output = layers.Dense(1, activation="sigmoid", name="classification_head")(cls_x)

        model = keras.Model(inputs=inputs, outputs=[reg_output, cls_output], name="performance_predictor")

        losses = {
            "regression_head": "mse",
            "classification_head": "binary_crossentropy",
        }
        loss_weights = {
            "regression_head": 1.0,
            "classification_head": 1.0,
        }

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics={
                "regression_head": [keras.metrics.MeanAbsoluteError(name="mae")],
                "classification_head": [keras.metrics.AUC(name="auc")],
            },
        )
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        batch_size: int = 256,
        epochs: int = 20,
        verbose: int = 1,
    ):
        # Build preprocessor on train
        self.preprocessor = self._build_preprocessor(train_df)

        # Transform features and targets
        X_train = self._transform_features(train_df)
        y_reg_train, y_cls_train = self._extract_targets(train_df)

        if val_df is not None:
            X_val = self._transform_features(val_df)
            y_reg_val, y_cls_val = self._extract_targets(val_df)
            validation_data = (X_val, {"regression_head": y_reg_val, "classification_head": y_cls_val})
        else:
            validation_data = None

        # Build model
        self.model = self._build_model(input_dim=X_train.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_regression_head_loss",  # watch validation loss
                mode="min",                          # smaller is better
                patience=3,
                restore_best_weights=True,
            )
        ]
        
        self.model.fit(
            X_train,
            {"regression_head": y_reg_train, "classification_head": y_cls_train},
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model not trained or loaded.")

        X = self._transform_features(df)
        reg_pred, cls_pred = self.model.predict(X, verbose=0)

        # reg_pred: columns follow TARGET_REGRESSION order
        out = {
            "views_24h_pred": reg_pred[:, 0],
            "views_7d_pred": reg_pred[:, 1],
            "engagement_velocity_24h_pred": reg_pred[:, 2],
            "virality_prob": cls_pred[:, 0],
        }
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        """
        Saves:
          - Keras model (SavedModel format)
          - sklearn preprocessor (joblib)
          - config JSON
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        # Save model in native Keras format

        model_path = path / "tf_model.keras"
        self.model.save(model_path)

    

        # Save preprocessor
        import joblib

        preproc_path = path / "preprocessor.joblib"
        joblib.dump(self.preprocessor, preproc_path)

        # Save config
        cfg_path = path / "config.json"
        with cfg_path.open("w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PerformancePredictor":
        path = Path(path)

        # Load config
        cfg_path = path / "config.json"
        with cfg_path.open("r") as f:
            cfg_dict = json.load(f)
        config = PerformancePredictorConfig(**cfg_dict)

        # Load model

        # Load model
        model_path = path / "tf_model.keras"
        model = keras.models.load_model(model_path)
        
        # Load preprocessor
        import joblib

        preproc_path = path / "preprocessor.joblib"
        preprocessor = joblib.load(preproc_path)

        obj = cls(config=config)
        obj.model = model
        obj.preprocessor = preprocessor
        return obj
