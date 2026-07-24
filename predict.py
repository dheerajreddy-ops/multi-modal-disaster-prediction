"""
predict.py
Prediction pipeline for the Multi-Modal Disaster Prediction System.
Loads trained model and provides prediction interface.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


class DisasterPredictor:
    def __init__(self):
        self.model = None
        self.text_processor = None
        self.sensor_processor = None
        self.metadata = None
        self.is_loaded = False

    def load(self):
        from model import MultiModalDisasterModel

        self.model = MultiModalDisasterModel()
        self.model.load()

        self.text_processor = joblib.load(os.path.join(MODEL_DIR, "text_processor.joblib"))
        self.sensor_processor = joblib.load(os.path.join(MODEL_DIR, "sensor_processor.joblib"))

        with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
            self.metadata = json.load(f)

        self.is_loaded = True
        return True

    def predict(self, text_report, sensor_data):
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load() first.")

        text_features = self.text_processor.transform([text_report])

        sensor_df = pd.DataFrame([sensor_data])
        sensor_features = self.sensor_processor.transform(sensor_df)

        result = self.model.predict(text_features, sensor_features)

        disaster_pred = result["disaster_type"][0]
        disaster_proba = result["disaster_proba"][0]
        severity_pred = result["severity"][0]
        severity_proba = result["severity_proba"][0]

        disaster_probs = {
            self.model.disaster_clf.classes_[i]: round(float(disaster_proba[i]), 4)
            for i in range(len(self.model.disaster_clf.classes_))
        }
        severity_probs = {
            self.model.severity_clf.classes_[i]: round(float(severity_proba[i]), 4)
            for i in range(len(self.model.severity_clf.classes_))
        }

        return {
            "disaster_type": disaster_pred,
            "disaster_confidence": round(float(max(disaster_proba)), 4),
            "disaster_probabilities": disaster_probs,
            "severity": severity_pred,
            "severity_confidence": round(float(max(severity_proba)), 4),
            "severity_probabilities": severity_probs,
        }

    def predict_batch(self, texts, sensor_readings):
        text_features = self.text_processor.transform(texts)
        sensor_df = pd.DataFrame(sensor_readings)
        sensor_features = self.sensor_processor.transform(sensor_df)

        result = self.model.predict(text_features, sensor_features)

        results = []
        for i in range(len(texts)):
            disaster_probs = {
                self.model.disaster_clf.classes_[j]: round(float(result["disaster_proba"][i][j]), 4)
                for j in range(len(self.model.disaster_clf.classes_))
            }
            severity_probs = {
                self.model.severity_clf.classes_[j]: round(float(result["severity_proba"][i][j]), 4)
                for j in range(len(self.model.severity_clf.classes_))
            }
            results.append({
                "disaster_type": result["disaster_type"][i],
                "disaster_confidence": round(float(max(result["disaster_proba"][i])), 4),
                "disaster_probabilities": disaster_probs,
                "severity": result["severity"][i],
                "severity_confidence": round(float(max(result["severity_proba"][i])), 4),
                "severity_probabilities": severity_probs,
            })

        return results
