# Dengue_Taiwan_Forecast

End-to-End Dengue Forecasting Platform (Taiwan 2010–2024)

Machine Learning • Deep Learning • Cloud Pipeline • Streamlit Web App

Project Snapshot

Region:
Southern Taiwan — Kaohsiung City, Tainan City, Pingtung County

Forecast Target:
Weekly dengue case counts

Data Coverage:
2010–2024

Dengue surveillance (Taiwan CDC)

Climate and rainfall (NCHC)

Mosquito vector indices: BI, HI, CI

Goal:
Develop an early-warning forecasting system to support public health action, outbreak preparedness, and vector control operations.

Models Implemented:
Random Forest, XGBoost, LSTM (Lag Features), Hybrid LSTM + Transformer

Technology Stack:
Python, Pandas, NumPy
scikit-learn, XGBoost
TensorFlow / Keras
Supabase Storage (Model Artifacts)
Streamlit UI
Render Cloud Deployment

Project Overview

This project builds a complete machine learning and deep learning pipeline for forecasting weekly dengue cases in Taiwan’s three highest-risk regions.
The workflow spans from data engineering (ETL), to model training and evaluation, to cloud artifact storage, to real-time visualization through a Streamlit app.

Architecture and Pipeline
      ┌────────────────────────────────────────────────────────────┐
      │                    1. Raw Data Sources                     │
      │  Taiwan CDC Dengue | CDC Mosquito Indices | NCHC Weather   │
      └────────────────────────────────────────────────────────────┘
                               │
                               ▼
      ┌────────────────────────────────────────────────────────────┐
      │              2. ETL and Data Preprocessing (Local)         │
      │  Weekly aggregation | Missing value handling | Lags        │
      │  Scaling | Feature engineering | QC filters                │
      └────────────────────────────────────────────────────────────┘
                               │
                               ▼
      ┌────────────────────────────────────────────────────────────┐
      │             3. Model Training (Local Python)               │
      │  Random Forest | XGBoost | LSTM | LSTM-Transformer         │
      │  Hyperparameter tuning                                     │
      └────────────────────────────────────────────────────────────┘
                               │
                               ▼
      ┌────────────────────────────────────────────────────────────┐
      │             4. Artifact Storage on Supabase (Cloud)        │
      │  predictions.csv | metrics.json | residuals.csv | history  │
      └────────────────────────────────────────────────────────────┘
                               │
                               ▼
      ┌────────────────────────────────────────────────────────────┐
      │               5. Streamlit Web Application                 │
      │  Reads artifacts → Visualizes forecasts and diagnostics    │
      │  Deployed on Render                                        │
      └────────────────────────────────────────────────────────────┘
                               │
                               ▼
      ┌────────────────────────────────────────────────────────────┐
      │                       6. End User                          │
      │  Model selection | City selection | Interactive analysis   │
      └────────────────────────────────────────────────────────────┘

Repository Structure
Dengue_Taiwan_Forecast/
│
├── app.py                      # Streamlit web application
├── data/                       # Raw and cleaned input data
│   └── 3_df_merged_cleaned.csv
│
├── artifacts/                  # Model artifacts (local copies)
│   ├── rf_ks_predictions.csv
│   ├── xg_pt_metrics.json
│   ├── lstm_lag_tn_history.csv
│   └── lstm_att_ks_residuals.csv
│
├── notebooks/                  # Training notebooks
│   ├── rforest_ks.ipynb
│   ├── xgboost_ks.ipynb
│   ├── lstm_ks_lag.ipynb
│   └── lstm_ks_att.ipynb
│
├── requirements.txt
├── README.md
└── LICENSE

Key Features
Multi-City Forecasting

Models trained independently for Kaohsiung, Tainan, and Pingtung.

Combined ML and DL Approaches

Comparative evaluation across classical ML (RF, XGB) and sequence models (LSTM, LSTM-Transformer).

Real-Time Interactive Dashboard

Streamlit interface includes:

EDA visualizations

Seasonal and climate patterns

ACF/PACF diagnostics

Predicted vs. actual time-series plots

Feature importance

Training and validation loss curves

Residual diagnostics

Model comparison tables

Cloud Integration

All model artifacts stored in Supabase Storage:

predictions.csv

metrics.json

residuals.csv

history.csv

feature_importance.csv

Streamlit retrieves data dynamically via the Supabase Python client.

Model Training Workflow

Install dependencies:

pip install -r requirements.txt


Run any training notebook:

jupyter notebook


Generate model outputs (artifacts).

Upload artifacts to Supabase:

supabase.storage.from_("artifacts").upload(path, data, upsert=True)

Run the Streamlit App Locally
streamlit run app.py

Deployment

The application is deployed on Render.
It retrieves model artifacts directly from Supabase to ensure seamless updates without redeploying the app.

Data Sources
Source	Provider	Description
Dengue Surveillance	Taiwan CDC	Daily confirmed dengue cases
Vector Indices	Taiwan CDC	BI, HI, CI mosquito indices
Meteorological Data	NCHC	Temperature, humidity, pressure, rainfall
License

MIT License

Author

Aichu Tan
Master of Science in Big Data Analytics
San Diego State University
