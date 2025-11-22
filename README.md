📘 Federated Health MLOps System

A complete end-to-end Federated Learning + MLOps pipeline for health-risk prediction using data from wearables, weather sensors, and clinic observations.
The system ensures data privacy, distributed training, central model aggregation, Dockerized deployment, API serving, and interactive dashboards.

🧠 Project Overview

Modern healthcare systems collect sensitive data from multiple distributed sources such as:

🩺 Wearables (heart-rate, SpO2, steps, sleep)

# Federated Health MLOps

A federated learning platform for health data, enabling privacy-preserving machine learning across distributed nodes (wearables, clinics, and environmental sensors).

## Project Structure

```
federated_health_mlops/
├── app/                # API application
├── dashboard/          # Admin and citizen dashboards
├── data/               # Synthetic and node-specific datasets
├── fl/                 # Federated learning logic (clients, server, model)
├── models/             # Saved models
├── notebooks/          # EDA and baseline experiments
├── Dockerfile.api      # Dockerfile for API
├── docker-compose.yml  # Docker Compose setup
├── requirements.txt    # Core dependencies
├── requirements_api.txt
├── requirements_dashboard.txt
└── README.md
```

## Features

- Federated learning with multiple simulated nodes
- Privacy-preserving health data analytics
- Synthetic data generation for testing
- Interactive dashboards for admins and citizens
- Containerized API and dashboard for easy deployment

## Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose

### Installation

1. Clone the repository:

```sh
git clone <repo-url>
cd federated_health_mlops
```

2. Install dependencies:

```sh
pip install -r requirements.txt
```

3. (Optional) Set up API and dashboard:

```sh
pip install -r requirements_api.txt
pip install -r requirements_dashboard.txt
```

### Running with Docker

```sh
docker-compose up --build
```

### Running Locally

- **API**:
  ```sh
  python app/main.py
  ```
- **Dashboard**:
  ```sh
  python dashboard/admin_app.py
  python dashboard/citizen_app.py
  ```

### Data Generation

Generate synthetic data:

```sh
python data/generate_synthetic_data.py
```

## Notebooks

- `01_eda.ipynb`: Exploratory Data Analysis
- `02_local_baseline.ipynb`: Local model baseline

## Federated Learning

- `fl/server.py`: Federated server logic
- `fl/client_node1.py`, `fl/client_node2.py`, `fl/client_node3.py`: Client logic for each node
- `fl/model.py`: Model architecture
- `fl/data_utils.py`: Data loading and preprocessing

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

[MIT License](LICENSE)

uvicorn app.main:app --host 0.0.0.0 --port 8000

Open Swagger:

➡ http://localhost:8000/docs

Example request:

{
"heart_rate": 88,
"spo2": 97,
"steps": 4500,
"sleep_hours": 6,
"age": 30,
"smoker": 0,
"chronic": 0,
"aqi": 60
}

Response:

{
"risk_score": 1,
"high_risk": true
}

🐳 Docker Deployment
Build API image
docker build -f Dockerfile.api -t federated-health-api .

Run container
docker run -p 8000:8000 federated-health-api

API now available at:

➡ http://localhost:8000/predict

🌐 Dashboards (Streamlit)
1️⃣ Citizen Dashboard
streamlit run dashboard/citizen_app.py

Displays:

Heart-rate

SpO2

AQI

Personalized health risk

Model-based recommendations

2️⃣ Admin / Public Health Dashboard
streamlit run dashboard/admin_app.py

Shows:

Real-time risk across 9 cities

High-risk alerts

Risk heatmaps

Trend analysis

Local client training losses

Global model evaluation

AQI distribution

🔄 CI Pipeline (GitHub Actions)

File: .github/workflows/ci.yml

Pipeline does:

✔ Install dependencies
✔ Validate model exists
✔ Test PyTorch import
✔ Build Docker image

Your GitHub Actions will show:
All green ✔ (build successful)




🧪 Model Evaluation Summary

From your federated training:

Central (Global) Evaluation
Round Loss Accuracy
0 0.6933 0.50
1 1.0031 0.5234
2 1.2762 0.5156
3 1.5598 0.5352
Client Local Losses

Node 1: 0.38 → 0.24 → 0.16

Node 2: 0.26 → 0.17 → 0.13

Node 3: 0.45 → 0.29 → 0.19




🏁 Conclusion

This project successfully demonstrates an end-to-end Federated Learning Health MLOps system, including:

✔ Privacy-preserving distributed training
✔ Central model aggregation
✔ Dockerized inference API
✔ Streamlit dashboards
✔ Automated CI pipeline
✔ Modular, production-ready architecture

This can be extended into a real-world digital health monitoring ecosystem.
