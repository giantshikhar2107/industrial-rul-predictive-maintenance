from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="NASA C-MAPSS Engine Monitoring",
    page_icon="✈️",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "ml" / "nasa_fd001_test_predictions.csv"

st.title("✈️ NASA C-MAPSS Engine Monitoring")
st.caption("LSTM Remaining Useful Life Prediction | NASA C-MAPSS FD001")

st.info(
    "DEMO MODE: This dashboard displays historical evaluation results "
    "from 100 NASA FD001 test engines. These are not live engine readings."
)


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    required = [
        "engine_id",
        "actual_rul",
        "predicted_rul",
        "absolute_error"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required).copy()
    df["engine_id"] = df["engine_id"].astype(int)
    df["absolute_error"] = (
        df["actual_rul"] - df["predicted_rul"]
    ).abs()

    return df.sort_values("engine_id")


if not CSV_PATH.exists():
    st.error(f"Prediction CSV not found: {CSV_PATH}")
    st.stop()

try:
    data = load_data(str(CSV_PATH))
except Exception as error:
    st.error(f"Unable to load evaluation data: {error}")
    st.stop()

if data.empty:
    st.error("No valid evaluation records found.")
    st.stop()


with st.sidebar:
    st.header("Monitoring Settings")

    engine_id = st.selectbox(
        "Select Engine",
        data["engine_id"].tolist()
    )

    st.write(f"Total engines: {len(data)}")

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()


engine = data[data["engine_id"] == engine_id].iloc[0]

actual = float(engine["actual_rul"])
predicted = float(engine["predicted_rul"])
error = float(engine["absolute_error"])

st.subheader(f"Engine {engine_id} Status")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Actual RUL", f"{actual:.2f} cycles")
c2.metric("Predicted RUL", f"{predicted:.2f} cycles")
c3.metric("Absolute Error", f"{error:.2f} cycles")
c4.metric("Prediction Difference", f"{predicted - actual:+.2f} cycles")

if predicted <= 10:
    st.error("CRITICAL — Predicted RUL is 10 cycles or less.")
elif predicted <= 30:
    st.warning("WARNING — Predicted RUL is between 11 and 30 cycles.")
else:
    st.success("NORMAL — Predicted RUL is above 30 cycles.")

st.caption(
    "These health thresholds are illustrative, not validated maintenance "
    "limits or safety criteria."
)

st.divider()

st.subheader("Overall Model Performance")

mae = data["absolute_error"].mean()

mse = (
    (data["actual_rul"] - data["predicted_rul"]) ** 2
).mean()

rmse = mse ** 0.5

bias = (
    data["predicted_rul"] - data["actual_rul"]
).mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Engines Evaluated", len(data))
c2.metric("MAE", f"{mae:.2f} cycles")
c3.metric("RMSE", f"{rmse:.2f} cycles")
c4.metric("Mean Bias", f"{bias:+.2f} cycles")

st.divider()

st.subheader("Actual vs Predicted RUL")

fig = px.scatter(
    data,
    x="actual_rul",
    y="predicted_rul",
    color="absolute_error",
    hover_data=["engine_id"],
    labels={
        "actual_rul": "Actual RUL (cycles)",
        "predicted_rul": "Predicted RUL (cycles)",
        "absolute_error": "Absolute Error"
    }
)

maximum = max(
    data["actual_rul"].max(),
    data["predicted_rul"].max()
)

fig.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=maximum,
    y1=maximum,
    line=dict(dash="dash", color="gray")
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Prediction Error by Engine")

error_fig = px.bar(
    data,
    x="engine_id",
    y="absolute_error",
    labels={
        "engine_id": "Engine ID",
        "absolute_error": "Absolute Error (cycles)"
    }
)

st.plotly_chart(error_fig, use_container_width=True)

st.divider()

st.subheader("Prediction Error Distribution")

hist = px.histogram(
    data,
    x="absolute_error",
    nbins=15
)

st.plotly_chart(hist, use_container_width=True)

st.divider()

st.subheader("Complete Evaluation Records")

st.dataframe(
    data,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download Evaluation Results",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="nasa_fd001_test_predictions.csv",
    mime="text/csv"
)

st.caption(
    "Source: NASA C-MAPSS FD001 offline test evaluation. "
    "Live MQTT readings and live alert history are not displayed in Demo Mode."
)