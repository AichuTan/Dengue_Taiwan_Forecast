import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
import io
from supabase import create_client, Client


import os


SUPABASE_URL = os.getenv("SUPABASE_URL", st.secrets.get("SUPABASE_URL", ""))
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", st.secrets.get("SUPABASE_ANON_KEY", ""))
SUPABASE_BUCKET_ARTIFACTS = os.getenv("SUPABASE_BUCKET_ARTIFACTS", st.secrets.get("SUPABASE_BUCKET_ARTIFACTS", "artifacts"))


# ------------------------------------------------------------------
# Supabase setup
# ------------------------------------------------------------------
@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

ARTIFACTS_BUCKET = SUPABASE_BUCKET_ARTIFACTS


# ARTIFACTS_BUCKET = st.secrets.get("SUPABASE_BUCKET_ARTIFACTS", "artifacts")


def read_csv_from_supabase(path: str, parse_dates=None) -> pd.DataFrame:
    """
    Download a CSV file from Supabase Storage and return as DataFrame.
    Expects objects like 'rf_ks_predictions.csv' in bucket 'artifacts'.
    """
    sb = get_supabase_client()
    data: bytes = sb.storage.from_(ARTIFACTS_BUCKET).download(path)
    buffer = io.BytesIO(data)
    return pd.read_csv(buffer, parse_dates=parse_dates)


def read_json_from_supabase(path: str) -> dict:
    """
    Download a JSON file from Supabase Storage and return as dict.
    """
    sb = get_supabase_client()
    data: bytes = sb.storage.from_(ARTIFACTS_BUCKET).download(path)
    text = data.decode("utf-8")
    return json.loads(text)


# ------------------------------------------------------------------
# 0.  Utility: prepare_weekly_city
#     -> Paste your real function here instead of this stub
# ------------------------------------------------------------------
def prepare_weekly_city(df_raw, city_name):
    df_city = df_raw[df_raw["City"] == city_name].copy()
    df_city["Date"] = pd.to_datetime(df_city["Date"])
    df_city = df_city.set_index("Date").sort_index()

    agg_dict = {
        "Cases": "sum",
        "Precip": "sum",
        "Pressure": "mean",
        "Tmean": "mean",
        "Tmin": "mean",
        "Tmax": "mean",
        "Humidity": "mean",
        "Windspeed": "mean",
        "BI": "mean",
        "HI": "mean",
        "CI": "mean",
        "PopDensity_km2": "mean",
    }

    df_weekly = df_city.resample("W").agg(agg_dict).asfreq("W")

    # cases: NaN -> 0 (no report)
    df_weekly["Cases"] = df_weekly["Cases"].fillna(0)

    # weather: interpolate
    weather_cols = ["Precip", "Pressure", "Tmean", "Tmin", "Tmax",
                    "Humidity", "Windspeed"]
    df_weekly[weather_cols] = (
        df_weekly[weather_cols]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    # mosquito indices: carry forward/backward
    index_cols = ["BI", "HI", "CI"]
    df_weekly[index_cols] = df_weekly[index_cols].ffill().bfill()

    return df_weekly


# ------------------------------------------------------------------
# 1. Data loader
# ------------------------------------------------------------------
@st.cache_data
def load_raw():
    # File in bucket: artifacts / 3_df_merged_cleaned.csv
    df = read_csv_from_supabase("3_df_merged_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df




CITY_CODE = {
    "Kaohsiung City": "ks",
    "Tainan City": "tn",
    "Pingtung County": "pt",
}

@st.cache_data
def load_rf_results(city):
    code = CITY_CODE[city]
    pred_df = read_csv_from_supabase(
        f"rf_{code}_predictions.csv",
        parse_dates=["Date"]
    )
    metrics = read_json_from_supabase(f"rf_{code}_metrics.json")
    fi_df = read_csv_from_supabase(f"rf_{code}_feature_importance.csv")
    return pred_df, metrics, fi_df



@st.cache_data
def load_xgb_results(city: str):
    code = CITY_CODE[city]
    pred_df = read_csv_from_supabase(
        f"xg_{code}_predictions.csv",
        parse_dates=["Date"]
    )
    metrics = read_json_from_supabase(f"xg_{code}_metrics.json")
    fi_df = read_csv_from_supabase(f"xg_{code}_feature_importance.csv")
    return pred_df, metrics, fi_df
@st.cache_data
def load_lstm_lag_artifacts(city: str):
    code = CITY_CODE[city]

    pred = read_csv_from_supabase(
        f"lstm_lag_{code}_predictions.csv",
        parse_dates=["Date"]
    )
    metrics = read_json_from_supabase(f"lstm_lag_{code}_metrics.json")
    resid = read_csv_from_supabase(
        f"lstm_lag_{code}_residuals.csv",
        parse_dates=["Date"]
    )
    return pred, metrics, resid


@st.cache_data
def load_lstm_att_results(city: str):
    code = CITY_CODE[city]

    pred = read_csv_from_supabase(
        f"lstm_att_{code}_predictions.csv",
        parse_dates=["Date"]
    )
    metrics = read_json_from_supabase(f"lstm_att_{code}_metrics.json")
    resid = read_csv_from_supabase(
        f"lstm_att_{code}_residuals.csv",
        parse_dates=["Date"]
    )
    return pred, metrics, resid


@st.cache_data
def load_lstm_history(city: str, model_name: str):
    """
    model_name e.g. "lstm_lag", "lstm_att"
    expects {model_name}_{code}_history.csv in bucket 'artifacts'
    """
    code = CITY_CODE[city]
    path = f"{model_name}_{code}_history.csv"
    return read_csv_from_supabase(path)


# ------------------------------------------------------------------
# 2. EDA plots
# ------------------------------------------------------------------
def plot_cases_timeseries(df_city, city):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_city.index, df_city["Cases"], marker=".", linestyle="-")
    ax.set_title(f"Dengue Cases – {city}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Cases")
    ax.grid(True, alpha=0.3)
    return fig


def plot_monthly_climate_boxplots(df_raw, city):
    df_city = prepare_weekly_city(df_raw, city)

    df_city["month"] = df_city.index.month
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    df_city["month_name"] = pd.Categorical(
        df_city["month"].map(month_map),
        categories=list(month_map.values()),
        ordered=True
    )

    features = [
        "Pressure", "Tmean", "Tmin", "Tmax",
        "Humidity", "Windspeed",
        "Precip", "BI", "HI", "CI", "PopDensity_km2"
    ]

    scaler = MinMaxScaler()
    df_scaled = df_city.copy()
    df_scaled[features] = scaler.fit_transform(df_city[features])

    long_df = df_scaled.melt(
        id_vars=["month_name"],
        value_vars=features,
        var_name="Feature",
        value_name="Value",
    )

    sns.set(style="whitegrid")
    g = sns.catplot(
        data=long_df,
        x="month_name", y="Value",
        col="Feature",
        kind="box",
        col_wrap=3,
        height=3.0, aspect=1.2,
        sharey=False, showfliers=False,
        palette="Blues",
    )
    g.set_axis_labels("Month", "Scaled Value")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=0)

    g.fig.suptitle(
        f"Monthly Distribution of Scaled Weather & Climate Variables ({city})",
        y=1.02, fontsize=14
    )
    return g.fig


def plot_monthly_cases_log(df_raw, city):
    df_city = prepare_weekly_city(df_raw, city)

    df_city["month"] = df_city.index.month
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    df_city["month_name"] = pd.Categorical(
        df_city["month"].map(month_map),
        categories=list(month_map.values()),
        ordered=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df_city.reset_index(),
        x="month_name",
        y="Cases",
        ax=ax,
        palette="Blues"
    )
    ax.set_yscale("log")
    ax.set_title(f"Monthly Dengue Cases (Log Scale) – {city}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Weekly Dengue Cases (log scale)")
    fig.tight_layout()
    return fig


def plot_acf_pacf(df_city, lags=30):
    series = df_city["Cases"]

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title("ACF of Weekly Dengue Cases")

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    plot_pacf(series, lags=lags, ax=ax2)
    ax2.set_title("PACF of Weekly Dengue Cases")

    return fig1, fig2


def plot_rf_test(dates, y_test, y_pred, city, title_prefix="Tuned Random Forest"):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(dates, y_test, label="Actual Cases", color="black")
    ax.plot(dates, y_pred, label="Predicted Cases", color="red")
    ax.set_title(f"{title_prefix} – Weekly Dengue Cases ({city})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Cases")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_feature_importance(fi, title):
    fig, ax = plt.subplots(figsize=(6,6))
    fi.head(20).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_model_test(dates, y_test, y_pred, city, title_prefix):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(dates, y_test, label="Actual Cases", color="black")
    ax.plot(dates, y_pred, label="Predicted Cases", color="red")
    ax.set_title(f"{title_prefix} – Weekly Dengue Cases ({city})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Cases")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_metric_bar(df_metrics_city, metric_name, city):
    """
    df_metrics_city:  DataFrame with index=model_name, columns=metrics
    metric_name: one of ["MSE", "RMSE", "MAE", "R2"]
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    df_metrics_city[metric_name].plot(kind="bar", ax=ax)

    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by Model – {city}")
    ax.set_xticklabels(df_metrics_city.index, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 3. Model results – metrics + pred vs actual (from artifacts)
# ------------------------------------------------------------------

# Map the display names in the UI to the file prefixes you used
MODEL_FILE_PREFIX = {
    "Random Forest":      "rf",
    "XGBoost":            "xg",
    "LSTM":               "lstm_lag",
    "LSTM-Transformer":   "lstm_att",
}

def _empty_metrics():
    """Fallback if metrics file is missing."""
    return {"MSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "R2": 0.0}

def build_model_metrics() -> dict:

    model_metrics = {}

    for city, code in CITY_CODE.items():
        model_metrics[city] = {}
        for model_label, prefix in MODEL_FILE_PREFIX.items():
            metrics_path = f"{prefix}_{code}_metrics.json"
            try:
                metrics = read_json_from_supabase(metrics_path)
            except Exception:
                metrics = _empty_metrics()


            # Ensure all keys exist (avoid KeyError later)
            for k in ["MSE", "RMSE", "MAE", "R2"]:
                metrics.setdefault(k, 0.0)

            model_metrics[city][model_label] = metrics

    return model_metrics

# Build once at import time (you can also wrap this with @st.cache_data if you like)
MODEL_METRICS = build_model_metrics()






# ------------------------------------------------------------------
# 4. Streamlit layout
# ------------------------------------------------------------------
def main():
    st.title("Dengue Forecasting in Southern Taiwan")

    df_raw = load_raw()

    # ----- Sidebar -----
    city = st.sidebar.selectbox(
        "Select city",
        ["Kaohsiung City", "Tainan City", "Pingtung County"]
    )
    st.sidebar.write(f"Current city: **{city}**")

    df_city_weekly = prepare_weekly_city(df_raw, city)

    tab_eda, tab_diag, tab_model, tab_compare = st.tabs(
        ["EDA", "ACF/PACF", "Single Model View", "Model Comparison"]
    )

    # ---------- EDA ----------
    with tab_eda:
        st.subheader("Epidemiological & Climate Patterns")
        st.pyplot(plot_cases_timeseries(df_city_weekly, city))
        st.pyplot(plot_monthly_climate_boxplots(df_raw, city))
        st.pyplot(plot_monthly_cases_log(df_raw, city))

    # ---------- Time-series diagnostics ----------
    with tab_diag:
        st.subheader("Autocorrelation of Weekly Dengue Cases")
        fig_acf, fig_pacf = plot_acf_pacf(df_city_weekly)
        st.pyplot(fig_acf)
        st.pyplot(fig_pacf)

    # ---------- Single model ----------
    with tab_model:
        st.subheader(f"Model Performance – {city}")
        model_name = st.selectbox(
            "Choose model",
            ["Random Forest", "XGBoost", "LSTM", "LSTM-Transformer"]
        )

        if model_name == "Random Forest":
            pred_df, metrics, fi_df = load_rf_results(city)

            dates  = pred_df["Date"]
            y_test = pred_df["y_test"].values
            y_pred = pred_df["y_pred"].values

            # metric cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{metrics['RMSE']:.2f}")
            c2.metric("MAE",  f"{metrics['MAE']:.2f}")
            c3.metric("MSE",  f"{metrics['MSE']:.1f}")
            c4.metric("R²",   f"{metrics['R2']:.3f}")

            # time-series plot
            st.pyplot(plot_rf_test(dates, y_test, y_pred, city))

            # feature importance
            fi = fi_df.set_index(fi_df.columns[0])["importance"]
            st.pyplot(
                plot_feature_importance(
                    fi, "Top 20 Feature Importances (Random Forest)"
                )
            )

        elif model_name == "XGBoost":
            pred_df, metrics, fi_df = load_xgb_results(city)

            dates  = pred_df["Date"]
            y_test = pred_df["y_test"].values
            y_pred = pred_df["y_pred"].values

            # metric cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{metrics['RMSE']:.2f}")
            c2.metric("MAE",  f"{metrics['MAE']:.2f}")
            c3.metric("MSE",  f"{metrics['MSE']:.1f}")
            c4.metric("R²",   f"{metrics['R2']:.3f}")

            # time-series plot
            st.pyplot(
                plot_model_test(dates, y_test, y_pred, city,
                                "Tuned XGBoost")
            )

            # feature importance
            fi = fi_df.set_index(fi_df.columns[0])["importance"]
            st.pyplot(
                plot_feature_importance(
                    fi, "Top 20 Feature Importances (XGBoost)"
                )
            )

        elif model_name == "LSTM":
            # ✅ FIX: unpack 3 values (pred, metrics, resid)
            pred_df, metrics, resid_df = load_lstm_lag_artifacts(city)

            dates  = pred_df["Date"]
            y_test = pred_df["y_test"].values
            y_pred = pred_df["y_pred"].values

            # Metric cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{metrics['RMSE']:.2f}")
            c2.metric("MAE",  f"{metrics['MAE']:.2f}")
            c3.metric("MSE",  f"{metrics['MSE']:.1f}")
            c4.metric("R²",   f"{metrics['R2']:.3f}")

            # Actual vs Predicted
            st.pyplot(
                plot_model_test(dates, y_test, y_pred, city, "LSTM (Lag)")
            )

            # Training history
            hist_df = load_lstm_history(city, "lstm_lag")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.semilogy(hist_df["loss"], label="Train loss")
            ax.semilogy(hist_df["val_loss"], label="Val loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log scale)")
            ax.set_title(f"LSTM (Lag) – Training & Validation Loss ({city})")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Optional: residuals
            fig_r, ax_r = plt.subplots(figsize=(10, 3))
            ax_r.plot(resid_df["Date"], resid_df["residual"])
            ax_r.axhline(0, color="black")
            ax_r.set_xlabel("Date")
            ax_r.set_ylabel("Residual")
            ax_r.set_title("LSTM (Lag) – Residuals Over Time")
            ax_r.grid(True)
            st.pyplot(fig_r)

        elif model_name == "LSTM-Transformer":
            # ✅ FIX: use LSTM-attention artifacts, not lag
            pred_df, metrics, resid_df = load_lstm_att_results(city)

            dates  = pred_df["Date"]
            y_test = pred_df["y_test"].values
            y_pred = pred_df["y_pred"].values

            # Metric cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{metrics['RMSE']:.2f}")
            c2.metric("MAE",  f"{metrics['MAE']:.2f}")
            c3.metric("MSE",  f"{metrics['MSE']:.1f}")
            c4.metric("R²",   f"{metrics['R2']:.3f}")

            # Actual vs Predicted
            st.pyplot(
                plot_model_test(dates, y_test, y_pred, city,
                                "LSTM + Transformer")
            )

            # Training history
            hist_df = load_lstm_history(city, "lstm_att")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.semilogy(hist_df["loss"], label="Train loss")
            ax.semilogy(hist_df["val_loss"], label="Val loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log scale)")
            ax.set_title(f"LSTM + Transformer – Training & Validation Loss ({city})")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Residuals
            fig_r, ax_r = plt.subplots(figsize=(10, 3))
            ax_r.plot(resid_df["Date"], resid_df["residual"])
            ax_r.axhline(0, color="black")
            ax_r.set_xlabel("Date")
            ax_r.set_ylabel("Residual")
            ax_r.set_title("LSTM + Transformer – Residuals Over Time")
            ax_r.grid(True)
            st.pyplot(fig_r)


    # ---------- Model comparison ----------
    with tab_compare:
        st.subheader(f"Model Comparison (Test Set Metrics) – {city}")

        df_metrics_city = pd.DataFrame(MODEL_METRICS[city]).T
        st.dataframe(
            df_metrics_city.style.format(
                {"MSE": "{:.1f}", "RMSE": "{:.2f}",
                "MAE": "{:.2f}", "R2": "{:.3f}"}
            )
        )

        st.pyplot(plot_metric_bar(df_metrics_city, "RMSE", city))
        st.pyplot(plot_metric_bar(df_metrics_city, "MSE", city))
        st.pyplot(plot_metric_bar(df_metrics_city, "MAE", city))
        st.pyplot(plot_metric_bar(df_metrics_city, "R2", city))



if __name__ == "__main__":
    main()