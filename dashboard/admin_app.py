# -----------------------------------------------------
# ADMIN / PUBLIC HEALTH DASHBOARD
# -----------------------------------------------------
import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------
# API Endpoint
# -----------------------------------------------------
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Public Health Monitoring Dashboard",
    layout="wide",
)

st.title("🏥 Public Health Risk Monitoring Dashboard")
st.markdown("AI-powered real-time health risk insights using Federated Learning.")

# -----------------------------------------------------
# SIMULATED CITY INPUT DATA
# -----------------------------------------------------
CITY_LIST = [
    "Karachi", "Lahore", "Islamabad", "Quetta", "Peshawar",
    "Multan", "Faisalabad", "Rawalpindi", "Hyderabad"
]

def simulate_city_wearable_data():
    """Simulate live city wearable + environmental data."""
    import random
    return {
        "heart_rate": random.randint(70, 110),
        "spo2": random.randint(90, 100),
        "steps": random.randint(1000, 8000),
        "sleep_hours": round(random.uniform(4, 9), 1),
        "age": random.randint(20, 70),
        "smoker": random.randint(0, 1),
        "chronic": random.randint(0, 1),
        "aqi": random.randint(40, 300),
    }

# -----------------------------------------------------
# FETCH LIVE CITY PREDICTIONS
# -----------------------------------------------------
def get_city_predictions():
    results = []

    for city in CITY_LIST:
        payload = simulate_city_wearable_data()
        try:
            response = requests.post(API_URL, json=payload)
            pred = response.json()
            results.append({
                "City": city,
                "Heart Rate": payload["heart_rate"],
                "SpO2": payload["spo2"],
                "AQI": payload["aqi"],
                "Risk Score": pred.get("risk_score", 0),
                "High Risk": pred.get("high_risk", False)
            })
        except:
            results.append({
                "City": city,
                "Heart Rate": payload["heart_rate"],
                "SpO2": payload["spo2"],
                "AQI": payload["aqi"],
                "Risk Score": None,
                "High Risk": None
            })

    return pd.DataFrame(results)

# -----------------------------------------------------
# AUTO-REFRESH OPTION
# -----------------------------------------------------
refresh = st.sidebar.checkbox("Auto-refresh every 10 seconds", value=True)

df = get_city_predictions()

if refresh:
    time.sleep(10)
    st.rerun()

# -----------------------------------------------------
# HIGH-RISK ALERTS
# -----------------------------------------------------
st.subheader("🚨 High-Risk Alerts")

high_risk_cities = df[df["High Risk"] == True]

if high_risk_cities.empty:
    st.success("All cities are currently in safe range.")
else:
    for _, row in high_risk_cities.iterrows():
        st.error(f"⚠ {row['City']} is HIGH RISK — Score {row['Risk Score']}")

# -----------------------------------------------------
# CITY RISK TABLE
# -----------------------------------------------------
st.subheader("📊 City Risk Overview")
st.dataframe(df, use_container_width=True)

# -----------------------------------------------------
# RISK MAP (AQI vs RISK SCORE)
# -----------------------------------------------------
st.subheader("🌍 Risk Map (AQI vs Risk Score)")

fig_map = px.scatter(
    df,
    x="AQI",
    y="Risk Score",
    size="Risk Score",
    color="High Risk",
    hover_name="City",
    title="City Risk Visualization",
)
st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------------------------------
# RISK TREND (SIMULATED)
# -----------------------------------------------------
st.subheader("📈 Risk Trend Simulation (Round-based)")

# Work only with rows that actually have a numeric Risk Score
df_trend = df[["City", "Risk Score"]].copy()
df_trend = df_trend.dropna(subset=["Risk Score"])

if df_trend.empty:
    st.info("No valid risk scores available yet to show trend (API might be down).")
else:
    # Simulate 3 rounds to make it look like time-series
    rounds = []
    for r in range(1, 4):
        tmp = df_trend.copy()
        tmp["Round"] = r
        # (optional) add tiny random noise so lines don't look completely flat
        tmp["Risk Score"] = tmp["Risk Score"] + (0.02 * (r - 2))
        rounds.append(tmp)

    trend_df = pd.concat(rounds, ignore_index=True)

    # Optional: show small table for debugging
    st.caption("Sample of trend data used below:")
    st.dataframe(trend_df.head(), use_container_width=True)

    fig_trend = px.line(
        trend_df,
        x="Round",
        y="Risk Score",
        color="City",
        markers=True,
        title="Simulated Risk Trend Across Rounds per City",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------------------------------
# GLOBAL TRAINING METRICS (REAL)
# -----------------------------------------------------
st.subheader("🧠 Global Federated Model Metrics")

global_rounds = [0, 1, 2, 3]
global_loss = [0.6933, 1.0031, 1.2762, 1.5598]
global_acc  = [0.50, 0.5234, 0.5156, 0.5352]

fig_global = go.Figure()
fig_global.add_trace(go.Scatter(x=global_rounds, y=global_loss, name="Loss", mode="lines+markers"))
fig_global.add_trace(go.Scatter(x=global_rounds, y=global_acc, name="Accuracy", mode="lines+markers"))
fig_global.update_layout(
    title="Global Model Performance per Round",
    xaxis_title="Round",
    yaxis_title="Value",
)
st.plotly_chart(fig_global, use_container_width=True)

# -----------------------------------------------------
# CLIENT-SIDE TRAINING LOSS (REAL)
# -----------------------------------------------------
st.subheader("🏥 Local Client Training Loss")

client_rounds = [1, 2, 3]
node1_loss = [0.38, 0.24, 0.16]
node2_loss = [0.26, 0.17, 0.13]
node3_loss = [0.45, 0.29, 0.19]

fig_client = go.Figure()
fig_client.add_trace(go.Scatter(x=client_rounds, y=node1_loss, name="Node 1"))
fig_client.add_trace(go.Scatter(x=client_rounds, y=node2_loss, name="Node 2"))
fig_client.add_trace(go.Scatter(x=client_rounds, y=node3_loss, name="Node 3"))
fig_client.update_layout(
    title="Client-Side Local Training Loss",
    xaxis_title="Round",
    yaxis_title="Loss",
)
st.plotly_chart(fig_client, use_container_width=True)

# -----------------------------------------------------
# AQI DISTRIBUTION (NODE 2 — FIXED)
# -----------------------------------------------------
st.subheader("🌫 Air Pollution Distribution (Node 2 Env Data)")

node2_df = pd.read_csv("data/node_2/env_weather.csv")

fig_aqi = px.histogram(
    node2_df,
    x="pm25",
    nbins=30,
    title="PM2.5 Distribution (Air Quality Indicator)"
)
st.plotly_chart(fig_aqi, use_container_width=True)

# -----------------------------------------------------
# HEALTH RISK DISTRIBUTION (NODE 1 WEARABLES)
# -----------------------------------------------------
st.subheader("🩺 Health Risk Breakdown (Wearables Data)")

node1_df = pd.read_csv("data/node_1/wearables.csv")
risk_counts = node1_df["health_risk"].value_counts()

fig_risk = px.pie(
    values=risk_counts.values,
    names=["Low Risk", "High Risk"],
    title="Health Risk Categories (Wearables)"
)
st.plotly_chart(fig_risk, use_container_width=True)
# -----------------------------------------------------
# DATA DRIFT MONITORING
# -----------------------------------------------------
st.subheader("📉 Data Drift Monitor")

# Baseline distributions from historical data
# (Node 2: pollution, Node 1: heart rate)

# These were already loaded above:
# node2_df = pd.read_csv("data/node_2/env_weather.csv")
# node1_df = pd.read_csv("data/node_1/wearables.csv")

baseline_pm25_mean = node2_df["pm25"].mean()
baseline_pm25_std = node2_df["pm25"].std()

baseline_hr_mean = node1_df["heart_rate"].mean()
baseline_hr_std = node1_df["heart_rate"].std()

# Live "streaming" stats from current dashboard data (df)
current_aqi_mean = df["AQI"].mean()
current_hr_mean = df["Heart Rate"].mean()

# Simple drift scores (z-score style)
z_aqi = (current_aqi_mean - baseline_pm25_mean) / (baseline_pm25_std + 1e-6)
z_hr = (current_hr_mean - baseline_hr_mean) / (baseline_hr_std + 1e-6)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Live AQI Mean (Stream)",
        f"{current_aqi_mean:.1f}",
        f"Δ {current_aqi_mean - baseline_pm25_mean:+.1f} vs baseline",
    )
    st.caption(
        f"Baseline PM2.5 mean (historical Node 2): {baseline_pm25_mean:.1f} "
        f" | Drift score (z): {z_aqi:.2f}"
    )

with col2:
    st.metric(
        "Live Heart Rate Mean (Stream)",
        f"{current_hr_mean:.1f}",
        f"Δ {current_hr_mean - baseline_hr_mean:+.1f} vs baseline",
    )
    st.caption(
        f"Baseline heart rate mean (historical Node 1): {baseline_hr_mean:.1f} "
        f" | Drift score (z): {z_hr:.2f}"
    )

# Simple drift rule: flag if any metric is > 2 std away
DRIFT_THRESHOLD = 2.0
drift_flags = []

if abs(z_aqi) > DRIFT_THRESHOLD:
    drift_flags.append("Pollution (AQI/PM2.5)")

if abs(z_hr) > DRIFT_THRESHOLD:
    drift_flags.append("Heart Rate")

if drift_flags:
    st.error(
        "⚠ Data drift detected in: " + ", ".join(drift_flags) +
        ". This may indicate a change in population behaviour or environment. "
        "You may need to retrain or fine-tune the global model."
    )
else:
    st.success(
        "✅ No significant data drift detected. "
        "Live streams are consistent with historical training data."
    )
