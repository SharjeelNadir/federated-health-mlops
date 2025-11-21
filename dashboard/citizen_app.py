import streamlit as st
import requests

st.set_page_config(page_title="Citizen Health Dashboard", page_icon="🩺")

st.title("🩺 Citizen Health Risk Checker")
st.write("Enter your daily wearable data to check health risk.")

# Input fields
heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 88)
spo2 = st.number_input("SpO2 (%)", 70, 100, 97)
steps = st.number_input("Steps Walked Today", 0, 30000, 4500)
sleep_hours = st.number_input("Sleep Hours", 0.0, 15.0, 6.0)
age = st.number_input("Age", 1, 100, 30)
smoker = st.selectbox("Do you smoke?", ["No", "Yes"])
chronic = st.selectbox("Chronic Illness?", ["No", "Yes"])
aqi = st.number_input("City AQI (Pollution)", 0, 500, 60)

payload = {
    "heart_rate": heart_rate,
    "spo2": spo2,
    "steps": steps,
    "sleep_hours": sleep_hours,
    "age": age,
    "smoker": 1 if smoker == "Yes" else 0,
    "chronic": 1 if chronic == "Yes" else 0,
    "aqi": aqi
}

if st.button("Check Risk"):
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        result = response.json()

        st.subheader("🔍 Prediction Result")
        st.write(f"**Risk Score:** {result['risk_score']}")
        st.write(f"**High Risk:** {'❗ YES' if result['high_risk'] else '✔ NO'}")

        if result["high_risk"]:
            st.error("⚠ You are at HIGH risk. Please consult a doctor.")
        else:
            st.success("You are safe. Keep monitoring daily!")

    except Exception as e:
        st.error(f"API Error: {e}")
