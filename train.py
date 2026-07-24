"""
train.py
Full training pipeline for the Multi-Modal Disaster Prediction System.
Loads data, processes text + sensor, trains fusion model, saves everything.
"""

import os
import sys
import io
import time
import json
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

from data_generator import generate_dataset
from text_processor import TextProcessor
from sensor_processor import SensorProcessor, SENSOR_COLUMNS
from model import MultiModalDisasterModel, DISASTER_LABELS, SEVERITY_LABELS


def load_data():
    csv_path = os.path.join(DATA_DIR, "disaster_data.csv")
    if not os.path.exists(csv_path):
        print("Dataset not found. Generating...")
        generate_dataset(10000)
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Loaded {len(df):,} samples")
    return df


def train_pipeline():
    print("=" * 60)
    print("  MULTI-MODAL DISASTER PREDICTION - TRAINING")
    print("=" * 60)
    print()

    total_start = time.time()

    df = load_data()
    print(f"Columns: {list(df.columns)}")
    print(f"Disaster types: {df['disaster_type'].value_counts().to_dict()}")
    print(f"Severity: {df['severity'].value_counts().to_dict()}")
    print()

    print("[1/5] Processing text data...")
    t0 = time.time()
    text_processor = TextProcessor(max_features=1500)
    text_features = text_processor.fit_transform(df["text_report"].fillna(""))
    print(f"  TF-IDF matrix: {text_features.shape}")
    print(f"  Time: {time.time() - t0:.2f}s")

    print("[2/5] Processing sensor data...")
    t0 = time.time()
    sensor_processor = SensorProcessor()
    sensor_features = sensor_processor.fit_transform(df)
    print(f"  Sensor matrix: {sensor_features.shape}")
    print(f"  Time: {time.time() - t0:.2f}s")

    print("[3/5] Training multi-modal fusion model...")
    t0 = time.time()
    model = MultiModalDisasterModel(n_estimators=100, max_depth=15)
    metrics = model.train(
        text_features, sensor_features,
        df["disaster_type"].values,
        df["severity"].values,
    )
    print(f"  Disaster classification CV accuracy: {metrics['disaster_accuracy']:.4f} (+/- {metrics['disaster_std']:.4f})")
    print(f"  Severity classification CV accuracy: {metrics['severity_accuracy']:.4f} (+/- {metrics['severity_std']:.4f})")
    print(f"  Time: {time.time() - t0:.2f}s")

    print("[4/5] Evaluating on full dataset...")
    eval_metrics = model.evaluate(
        text_features, sensor_features,
        df["disaster_type"].values,
        df["severity"].values,
    )

    print("  Disaster Classification Report:")
    for label in DISASTER_LABELS:
        if label in eval_metrics["disaster_report"]:
            p = eval_metrics["disaster_report"][label]["precision"]
            r = eval_metrics["disaster_report"][label]["recall"]
            f1 = eval_metrics["disaster_report"][label]["f1-score"]
            print(f"    {label:12s}: P={p:.3f} R={r:.3f} F1={f1:.3f}")

    print("  Severity Classification Report:")
    for label in SEVERITY_LABELS:
        if label in eval_metrics["severity_report"]:
            p = eval_metrics["severity_report"][label]["precision"]
            r = eval_metrics["severity_report"][label]["recall"]
            f1 = eval_metrics["severity_report"][label]["f1-score"]
            print(f"    {label:12s}: P={p:.3f} R={r:.3f} F1={f1:.3f}")

    print("[5/5] Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save()

    import joblib
    joblib.dump(text_processor, os.path.join(MODEL_DIR, "text_processor.joblib"), compress=3)
    joblib.dump(sensor_processor, os.path.join(MODEL_DIR, "sensor_processor.joblib"), compress=3)

    sensor_stats = sensor_processor.get_feature_stats(df)
    feature_importances = {
        "disaster": dict(zip(
            list(range(text_processor.n_features + sensor_processor.n_features)),
            model.disaster_clf.feature_importances_.tolist()
        ))
    }

    metadata = {
        "total_samples": len(df),
        "disaster_types": DISASTER_LABELS,
        "severity_levels": SEVERITY_LABELS,
        "text_features": text_processor.n_features,
        "sensor_features": sensor_processor.n_features,
        "total_features": text_processor.n_features + sensor_processor.n_features,
        "cv_metrics": metrics,
        "sensor_stats": sensor_stats,
    }

    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Model saved to: {MODEL_DIR}")

    model_size = sum(
        os.path.getsize(os.path.join(MODEL_DIR, f))
        for f in os.listdir(MODEL_DIR)
    )
    print(f"Model size: {model_size / (1024*1024):.2f} MB")
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    train_pipeline()
