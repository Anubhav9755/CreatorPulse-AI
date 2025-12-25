# features/engineering.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = [
    "subscriber_count",
    "upload_hour",
    "day_of_week",
    "video_duration_sec",
    "title_length",
    "thumbnail_text_length",
    "sentiment_score",
    "views_24h",        # for some label-engineering or baselines
    "likes_24h",
    "comments_24h",
    "shares_24h",
    "engagement_velocity_24h",
    "retention_avg_pct",
    "retention_p25_pct",
    "retention_p50_pct",
    "retention_p75_pct",
]

EMBEDDING_FEATURES = [f"topic_emb_{i}" for i in range(16)]

CATEGORICAL_FEATURES = ["niche", "country", "topic"]


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["published_at"])
    return df


def train_val_test_split(df: pd.DataFrame,
                         test_size: float = 0.2,
                         val_size: float = 0.1,
                         random_state: int = 42):
    # Time-based split is better; for now simple random split for synthetic data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
    return train_df, val_df, test_df
