import os
import numpy as np
import pandas as pd

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def simulate_node_1(n_samples=2000):
    """
    Hospital A: wearable vitals + basic demographic + risk label
    """
    user_ids = np.random.randint(1, 201, size=n_samples)
    heart_rate = np.random.normal(80, 10, size=n_samples).clip(50, 140)
    spo2 = np.random.normal(96, 2, size=n_samples).clip(85, 100)
    steps = np.random.normal(6000, 2500, size=n_samples).clip(0, 20000)
    sleep_hours = np.random.normal(6.5, 1.5, size=n_samples).clip(2, 12)
    age = np.random.randint(18, 80, size=n_samples)
    smoker = np.random.binomial(1, 0.25, size=n_samples)
    chronic = np.random.binomial(1, 0.30, size=n_samples)

    # Simple AQI "exposure" proxy
    aqi = np.random.normal(110, 40, size=n_samples).clip(20, 300)

    # Risk rule (you can tune this)
    # High AQI + chronic/smoker + poor sleep/low spo2 → higher risk
    risk_score = (
        0.02 * (aqi - 80)
        + 0.5 * smoker
        + 0.7 * chronic
        + 0.03 * (90 - spo2)
        + 0.1 * (7 - sleep_hours)
    )

    prob = 1 / (1 + np.exp(-risk_score))
    y = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "heart_rate": heart_rate,
            "spo2": spo2,
            "steps": steps,
            "sleep_hours": sleep_hours,
            "age": age,
            "smoker": smoker,
            "chronic": chronic,
            "aqi": aqi,
            "health_risk": y,
        }
    )
    out_dir = os.path.join(BASE_DIR, "node_1")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "wearables.csv"), index=False)


def simulate_node_2(n_samples=2000):
    """
    City Env Dept: air quality + weather + city-level risk
    """
    city_ids = np.random.choice(["CityA", "CityB", "CityC"], size=n_samples)
    pm25 = np.random.normal(80, 30, size=n_samples).clip(5, 300)
    pm10 = np.random.normal(120, 40, size=n_samples).clip(10, 400)
    no2 = np.random.normal(40, 15, size=n_samples).clip(5, 150)
    o3 = np.random.normal(30, 10, size=n_samples).clip(3, 100)
    temperature = np.random.normal(25, 7, size=n_samples).clip(-5, 50)
    humidity = np.random.normal(55, 20, size=n_samples).clip(10, 100)
    wind_speed = np.random.normal(8, 3, size=n_samples).clip(0, 30)

    # City risk mainly from air quality
    risk_score = 0.02 * (pm25 - 50) + 0.01 * (pm10 - 80) + 0.05 * (no2 - 30)
    prob = 1 / (1 + np.exp(-risk_score))
    y = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "city_id": city_ids,
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "o3": o3,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "health_risk": y,
        }
    )
    out_dir = os.path.join(BASE_DIR, "node_2")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "env_weather.csv"), index=False)


def simulate_node_3(n_samples=2000):
    """
    Clinic C: wearables + simple text-like symptom flag (encoded as numeric for now)
    """
    user_ids = np.random.randint(1, 151, size=n_samples)
    heart_rate = np.random.normal(78, 9, size=n_samples).clip(50, 140)
    spo2 = np.random.normal(97, 1.5, size=n_samples).clip(88, 100)
    steps = np.random.normal(7000, 2800, size=n_samples).clip(0, 20000)
    age = np.random.randint(18, 80, size=n_samples)
    cough_flag = np.random.binomial(1, 0.2, size=n_samples)
    breathless_flag = np.random.binomial(1, 0.15, size=n_samples)

    # Risk rule
    risk_score = (
        0.04 * (heart_rate - 80)
        + 0.05 * (95 - spo2)
        + 0.8 * cough_flag
        + 1.0 * breathless_flag
    )
    prob = 1 / (1 + np.exp(-risk_score))
    y = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "heart_rate": heart_rate,
            "spo2": spo2,
            "steps": steps,
            "age": age,
            "cough_flag": cough_flag,
            "breathless_flag": breathless_flag,
            "health_risk": y,
        }
    )
    out_dir = os.path.join(BASE_DIR, "node_3")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "clinic.csv"), index=False)


if __name__ == "__main__":
    simulate_node_1()
    simulate_node_2()
    simulate_node_3()
    print("Synthetic data generated for node_1, node_2, node_3.")
