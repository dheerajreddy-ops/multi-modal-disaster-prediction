"""
sensor_processor.py
Processes numerical sensor data (weather, seismic, environmental).
Normalizes features and prepares them for model input.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SENSOR_COLUMNS = [
    "seismic_activity", "ground_vibration", "temperature_c", "humidity_pct",
    "wind_speed_kmh", "air_pressure_hpa", "rainfall_mm", "water_level_m",
    "visibility_km",
]


class SensorProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df):
        sensor_data = df[SENSOR_COLUMNS].values
        self.scaler.fit(sensor_data)
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("SensorProcessor not fitted. Call fit() first.")
        sensor_data = df[SENSOR_COLUMNS].values
        return self.scaler.transform(sensor_data)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_feature_stats(self, df):
        stats = {}
        for col in SENSOR_COLUMNS:
            if col in df.columns:
                stats[col] = {
                    "mean": round(df[col].mean(), 2),
                    "std": round(df[col].std(), 2),
                    "min": round(df[col].min(), 2),
                    "max": round(df[col].max(), 2),
                }
        return stats

    @property
    def n_features(self):
        return len(SENSOR_COLUMNS)
