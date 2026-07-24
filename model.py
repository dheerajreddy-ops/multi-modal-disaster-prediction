"""
model.py
Multi-modal fusion model for disaster prediction.
Combines text features (TF-IDF) and sensor features (normalized)
using an early fusion approach with Random Forest classifier.
"""

import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

DISASTER_LABELS = ["earthquake", "flood", "hurricane", "wildfire", "tornado", "tsunami"]
SEVERITY_LABELS = ["low", "moderate", "high", "critical"]


class MultiModalDisasterModel:
    def __init__(self, n_estimators=200, max_depth=None):
        self.disaster_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.severity_clf = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.2,
            random_state=42,
            subsample=0.8,
        )
        self.is_trained = False

    def _combine_features(self, text_features, sensor_features):
        if hasattr(text_features, "toarray"):
            text_dense = text_features.toarray()
        else:
            text_dense = np.array(text_features)

        if hasattr(sensor_features, "toarray"):
            sensor_dense = sensor_features.toarray()
        else:
            sensor_dense = np.array(sensor_features)

        return np.hstack([text_dense, sensor_dense])

    def train(self, text_features, sensor_features, disaster_labels, severity_labels):
        X = self._combine_features(text_features, sensor_features)
        disaster_labels = np.array(disaster_labels)
        severity_labels = np.array(severity_labels)

        self.disaster_clf.fit(X, disaster_labels)
        self.severity_clf.fit(X, severity_labels)

        self.is_trained = True

        return {
            "disaster_classes": list(self.disaster_clf.classes_),
            "severity_classes": list(self.severity_clf.classes_),
        }

    def predict(self, text_features, sensor_features):
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = self._combine_features(text_features, sensor_features)

        disaster_pred = self.disaster_clf.predict(X)
        disaster_proba = self.disaster_clf.predict_proba(X)

        severity_pred = self.severity_clf.predict(X)
        severity_proba = self.severity_clf.predict_proba(X)

        return {
            "disaster_type": disaster_pred,
            "disaster_proba": disaster_proba,
            "severity": severity_pred,
            "severity_proba": severity_proba,
        }

    def evaluate(self, text_features, sensor_features, disaster_labels, severity_labels):
        X = self._combine_features(text_features, sensor_features)

        disaster_pred = self.disaster_clf.predict(X)
        severity_pred = self.severity_clf.predict(X)

        disaster_report = classification_report(
            disaster_labels, disaster_pred,
            labels=DISASTER_LABELS,
            output_dict=True,
            zero_division=0,
        )
        severity_report = classification_report(
            severity_labels, severity_pred,
            labels=SEVERITY_LABELS,
            output_dict=True,
            zero_division=0,
        )

        disaster_cm = confusion_matrix(disaster_labels, disaster_pred, labels=DISASTER_LABELS).tolist()
        severity_cm = confusion_matrix(severity_labels, severity_pred, labels=SEVERITY_LABELS).tolist()

        return {
            "disaster_report": disaster_report,
            "severity_report": severity_report,
            "disaster_confusion_matrix": disaster_cm,
            "severity_confusion_matrix": severity_cm,
        }

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(self.disaster_clf, os.path.join(MODEL_DIR, "disaster_classifier.joblib"), compress=3)
        joblib.dump(self.severity_clf, os.path.join(MODEL_DIR, "severity_classifier.joblib"), compress=3)
        self.is_trained = True

    def load(self):
        self.disaster_clf = joblib.load(os.path.join(MODEL_DIR, "disaster_classifier.joblib"))
        self.severity_clf = joblib.load(os.path.join(MODEL_DIR, "severity_classifier.joblib"))
        self.is_trained = True
