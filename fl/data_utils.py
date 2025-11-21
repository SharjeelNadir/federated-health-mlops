import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_node_1_data(base_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = os.path.join(base_dir, "data", "node_1", "wearables.csv")
    df = pd.read_csv(path)

    feature_cols = [
        "heart_rate",
        "spo2",
        "steps",
        "sleep_hours",
        "age",
        "smoker",
        "chronic",
        "aqi",
    ]

    X = df[feature_cols].values               # 8 features (already perfect)
    y = df["health_risk"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
    )


def load_node_2_data(base_dir: str):
    path = os.path.join(base_dir, "data", "node_2", "env_weather.csv")
    df = pd.read_csv(path)

    feature_cols = [
        "pm25",
        "pm10",
        "no2",
        "o3",
        "temperature",
        "humidity",
        "wind_speed",
    ]

    X = df[feature_cols].values               # 7 features
    y = df["health_risk"].values

    # ---- PAD TO 8 FEATURES ----
    if X.shape[1] < 8:
        pad_width = 8 - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad_width))])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
    )


def load_node_3_data(base_dir: str):
    path = os.path.join(base_dir, "data", "node_3", "clinic.csv")
    df = pd.read_csv(path)

    feature_cols = [
        "heart_rate",
        "spo2",
        "steps",
        "age",
        "cough_flag",
        "breathless_flag",
    ]

    X = df[feature_cols].values               # 6 features
    y = df["health_risk"].values

    # ---- PAD TO 8 FEATURES ----
    if X.shape[1] < 8:
        pad_width = 8 - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad_width))])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
    )
