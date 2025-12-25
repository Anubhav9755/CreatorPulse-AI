import os
import pandas as pd

from features.engineering import load_dataset, train_val_test_split
from models.performance_predictor import PerformancePredictor
from models.virality_classifier import ViralityClassifier


def main():
    # 1) Load dataset
    data_path = os.path.join("data", "youtube_data.csv")
    print(f"Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    print("Dataset shape:", df.shape)

    # 2) Split
    train_df, val_df, test_df = train_val_test_split(df)
    print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

    # 3) Train PerformancePredictor
    print("\n=== Training PerformancePredictor (TensorFlow) ===")
    perf_model = PerformancePredictor()
    perf_model.fit(train_df, val_df, epochs=10, batch_size=512)

    # Quick sanity check
    test_subset = test_df.iloc[:1000].copy()
    perf_preds = perf_model.predict(test_subset)
    print("Sample predicted 24h views:", perf_preds["views_24h_pred"][:5])
    print("Sample virality probabilities:", perf_preds["virality_prob"][:5])

    # Save
    perf_out_path = os.path.join("models", "artifacts", "performance_predictor")
    perf_model.save(perf_out_path)
    print(f"✅ Saved PerformancePredictor to {perf_out_path}")

    # 4) Train ViralityClassifier
    print("\n=== Training ViralityClassifier (Sklearn) ===")
    vc = ViralityClassifier()
    metrics = vc.fit(train_df, val_df)

    print("Metrics:", metrics)
    print("Top 10 feature importances:")
    print(vc.feature_importance(top_k=10))

    # Save
    viral_out_path = os.path.join("models", "artifacts", "virality_classifier")
    vc.save(viral_out_path)
    print(f"✅ Saved ViralityClassifier to {viral_out_path}")


if __name__ == "__main__":
    main()
