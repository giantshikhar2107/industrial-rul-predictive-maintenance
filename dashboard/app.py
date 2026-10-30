from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ml"

RESULTS_PATH = ML_DIR / "nasa_fd001_test_predictions.csv"
ACTUAL_PREDICTED_PATH = ML_DIR / "nasa_fd001_actual_vs_predicted.png"
ERROR_PATH = ML_DIR / "nasa_fd001_prediction_error.png"


st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Industrial Predictive Maintenance System")

st.markdown(
    """
    Welcome to the Industrial Machine Predictive Maintenance platform.

    Monitor simulated machine readings and explore NASA C-MAPSS
    Remaining Useful Life (RUL) predictions.
    """
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Simulator Monitoring")
    st.write(
        "Monitor simulated machine sensor readings and predicted "
        "Remaining Useful Life (RUL)."
    )
    st.page_link(
        "pages/1_Simulator_Monitoring.py",
        label="Open Simulator Monitoring",
        icon="📡",
    )

with col2:
    st.subheader("🚀 NASA C-MAPSS Analytics")
    st.write(
        "Explore LSTM predictions, official test-set performance, "
        "and engine-level prediction errors."
    )

    st.write("Dataset: NASA C-MAPSS FD001")
    st.write("Model: LSTM")
    st.write("Input window: 30 cycles")

st.divider()

st.header("📊 NASA FD001 Model Evaluation")

if RESULTS_PATH.exists():
    results = pd.read_csv(RESULTS_PATH)

    required_columns = {
        "engine_id",
        "actual_rul",
        "predicted_rul",
        "absolute_error",
    }

    if results.empty or not required_columns.issubset(results.columns):
        st.error("The NASA prediction results file is empty or invalid.")
        st.stop()

    if results[list(required_columns)].isna().any().any():
        st.error("The NASA prediction results contain missing values.")
        st.stop()

    actual = results["actual_rul"].to_numpy(dtype=float)
    predicted = results["predicted_rul"].to_numpy(dtype=float)

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        st.error("The NASA prediction results contain invalid numeric values.")
        st.stop()

    if (actual < 0).any() or (predicted < 0).any():
        st.error("RUL values must be nonnegative.")
        st.stop()

    errors = predicted - actual

    nasa_score = float(
        np.sum(
            np.where(
                errors < 0,
                np.exp(-errors / 13.0) - 1.0,
                np.exp(errors / 10.0) - 1.0,
            )
        )
    )

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Test Engines", len(results))
    metric2.metric("MAE", f"{mae:.3f} cycles")
    metric3.metric("RMSE", f"{rmse:.3f} cycles")
    metric4.metric("NASA Score", f"{nasa_score:.3f}")

    st.caption(
        "These metrics describe the offline NASA FD001 test set. "
        "Actual RUL is available here because this is a labelled evaluation dataset."
    )

    st.subheader("Engine-Level Predictions")

    selected_engine = st.selectbox(
        "Select an engine",
        sorted(results["engine_id"].unique()),
    )

    engine_result = results[
        results["engine_id"] == selected_engine
    ].iloc[0]

    engine_col1, engine_col2, engine_col3 = st.columns(3)

    engine_col1.metric(
        "Actual RUL",
        f"{engine_result['actual_rul']:.1f} cycles",
    )

    engine_col2.metric(
        "Predicted RUL",
        f"{engine_result['predicted_rul']:.2f} cycles",
    )

    engine_col3.metric(
        "Absolute Error",
        f"{abs(engine_result['actual_rul'] - engine_result['predicted_rul']):.2f} cycles",
    )

    st.dataframe(
        results.sort_values("engine_id").round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Prediction Results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="nasa_fd001_test_predictions.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("📈 Actual vs. Predicted RUL")

    if ACTUAL_PREDICTED_PATH.exists():
        st.image(
            str(ACTUAL_PREDICTED_PATH),
            caption="Actual versus predicted RUL for the NASA FD001 test engines",
            use_container_width=True,
        )
    else:
        st.info("Run visualize_nasa_predictions.py to generate this graph.")

    st.subheader("⚠️ Prediction Error by Engine")

    if ERROR_PATH.exists():
        st.image(
            str(ERROR_PATH),
            caption="Signed prediction error for each test engine",
            use_container_width=True,
        )
    else:
        st.info("Run visualize_nasa_predictions.py to generate this graph.")

else:
    st.warning(
        "NASA evaluation results were not found. Run "
        "python evaluate_nasa_lstm.py from the project root first."
    )

st.divider()

st.caption(
    "NASA C-MAPSS analytics uses offline test predictions. "
    "Simulator Monitoring continues to use the existing simulator pipeline."
)