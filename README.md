# End-to-End Dengue Forecasting Platform (Taiwan 2010–2024)

Machine Learning · Deep Learning · Cloud Pipeline · Streamlit Web App

---

## Project Snapshot

### Region
- Southern Taiwan — Kaohsiung City, Tainan City, Pingtung County

### Forecast Target
- Weekly dengue case counts

### Data Coverage
- 2010–2024
- Dengue surveillance (Taiwan CDC)
- Climate and rainfall (NCHC)
- Mosquito vector indices: BI, HI, CI
- Taiwan Population Data - ArCGIS enriched data 2024

### Goal
Develop an early-warning forecasting system to support public health action, outbreak preparedness, and vector control operations.

### Models Implemented
- Random Forest  
- XGBoost  
- LSTM (Lag Features)  
- Hybrid LSTM + Transformer  

### Technology Stack
- Python, Pandas, NumPy  
- scikit-learn, XGBoost  
- TensorFlow / Keras  
- Supabase Storage (Model Artifacts)  
- Streamlit UI  
- Render Cloud Deployment  

---

## Project Overview

This project builds a complete machine learning and deep learning pipeline for forecasting weekly dengue cases in Taiwan’s three highest-risk regions.

The workflow spans from data engineering (ETL), to model training and evaluation, to cloud artifact storage, to real-time visualization through a Streamlit app.

---

## Architecture and Pipeline

```text
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

```
Author

Aichu Tan
Master of Science in Big Data Analytics
San Diego State University

### Project Website  
https://aichutan.github.io/Dengue_Taiwan_Forecast/

### ArcGIS Spatiotemporal Dashboard  
https://experience.arcgis.com/experience/1eebab4280a549d294e274392d64625f

### Live Streamlit Forecasting App  
https://dengue-taiwan-forecast.onrender.com/

### Video Presentation  
(Insert link once available)

---

## License  
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
