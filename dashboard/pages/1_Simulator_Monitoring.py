import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from supabase import create_client
from streamlit_autorefresh import st_autorefresh


st.set_page_config(
    page_title="Industrial RUL Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.html(
    """
    <style>

    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #9ca3af;
    }

    [data-testid="stMetricValue"] {
        color: #f9fafb;
    }

    .dashboard-header {
        background: linear-gradient(
            135deg,
            #111827,
            #172554
        );
        padding: 25px 30px;
        border-radius: 16px;
        border: 1px solid #263244;
        margin-bottom: 20px;
    }

    .dashboard-title {
        font-size: 32px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        font-size: 15px;
        color: #94a3b8;
    }

    .status-card {
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 1px solid #263244;
    }

    .status-online {
        background-color: #052e16;
        border-color: #166534;
        color: #86efac;
    }

    .status-warning {
        background-color: #422006;
        border-color: #92400e;
        color: #fcd34d;
    }

    .status-danger {
        background-color: #450a0a;
        border-color: #991b1b;
        color: #fca5a5;
    }

    .section-title {
        font-size: 21px;
        font-weight: 600;
        color: #f8fafc;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .info-box {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 10px;
    }

    </style>
    """
)


st_autorefresh(
    interval=2000,
    key="dashboard_refresh"
)


def get_supabase_client():

    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_PUBLISHABLE_KEY"]

    except Exception:

        from dotenv import load_dotenv

        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )

        env_file = os.path.join(
            project_root,
            ".env"
        )

        load_dotenv(env_file)

        supabase_url = os.getenv(
            "SUPABASE_URL"
        )

        supabase_key = os.getenv(
            "SUPABASE_PUBLISHABLE_KEY"
        )

    if not supabase_url or not supabase_key:

        st.error(
            "Supabase credentials are not configured."
        )

        st.stop()

    return create_client(
        supabase_url,
        supabase_key
    )


supabase = get_supabase_client()


def get_machine_info():

    response = (
        supabase
        .table("machines")
        .select(
            "machine_id,machine_name,"
            "machine_type,location"
        )
        .limit(1)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    return pd.DataFrame(
        response.data
    )


def get_latest_reading():

    response = (
        supabase
        .table("sensor_readings")
        .select(
            "id,machine_id,timestamp,cycle,"
            "temperature,vibration,current,"
            "pressure,rpm,degradation,rul,"
            "predicted_rul"
        )
        .order(
            "id",
            desc=True
        )
        .limit(1)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    return pd.DataFrame(
        response.data
    )


def get_recent_readings():

    response = (
        supabase
        .table("sensor_readings")
        .select(
            "id,machine_id,timestamp,cycle,"
            "temperature,vibration,current,"
            "pressure,rpm,degradation,rul,"
            "predicted_rul"
        )
        .order(
            "id",
            desc=True
        )
        .limit(100)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    data = pd.DataFrame(
        response.data
    )

    return data.sort_values(
        "id"
    )


def get_reading_count():

    response = (
        supabase
        .table("sensor_readings")
        .select(
            "id",
            count="exact"
        )
        .execute()
    )

    return (
        response.count
        if response.count is not None
        else 0
    )


try:

    machine_info = get_machine_info()
    latest_reading = get_latest_reading()
    recent_readings = get_recent_readings()
    reading_count = get_reading_count()

except Exception as e:

    st.error(
        "Unable to connect to Supabase."
    )

    st.code(
        str(e)
    )

    st.stop()


if machine_info.empty:

    st.warning(
        "No machine information found."
    )

    st.stop()


if latest_reading.empty:

    st.warning(
        "No sensor readings found."
    )

    st.stop()


machine = machine_info.iloc[0]
latest = latest_reading.iloc[0]


st.html(
    """
    <div class="dashboard-header">

        <div class="dashboard-title">
            🏭 Industrial Machine RUL Monitor
        </div>

        <div class="dashboard-subtitle">
            Real-time predictive maintenance and machine
            condition monitoring system
        </div>

    </div>
    """
)


with st.sidebar:

    st.markdown(
        "## 🏭 Machine System"
    )

    st.divider()

    st.markdown(
        f"""
        **Machine ID**

        `{machine["machine_id"]}`

        **Machine Name**

        {machine["machine_name"]}

        **Machine Type**

        {machine["machine_type"]}

        **Location**

        {machine["location"]}
        """
    )

    st.divider()

    st.markdown(
        "### 📊 Data Pipeline"
    )

    st.write("🟢 Machine Simulator")
    st.write("🟢 HiveMQ Cloud")
    st.write("🟢 MQTT Subscriber")
    st.write("🟢 Supabase")
    st.write("🟢 Streamlit")

    st.divider()

    if st.button(
        "🔄 Refresh Now",
        use_container_width=True
    ):

        st.rerun()

    st.caption(
        "Dashboard refresh interval: 2 seconds"
    )


predicted_rul = latest.get(
    "predicted_rul"
)


if pd.isna(predicted_rul):

    predicted_rul = None


actual_rul = float(
    latest["rul"]
)


degradation = float(
    latest["degradation"]
)


if predicted_rul is not None:

    predicted_rul = max(
        0,
        float(predicted_rul)
    )


if degradation < 0.40:

    machine_status = "NORMAL"
    status_class = "status-online"
    status_icon = "🟢"

elif degradation < 0.70:

    machine_status = "WARNING"
    status_class = "status-warning"
    status_icon = "🟡"

else:

    machine_status = "CRITICAL"
    status_class = "status-danger"
    status_icon = "🔴"


st.html(
    f"""
    <div class="status-card {status_class}">

        <strong>
            {status_icon} MACHINE STATUS:
            {machine_status}
        </strong>

        &nbsp;&nbsp;|&nbsp;&nbsp;

        Degradation:
        <strong>
            {degradation * 100:.1f}%
        </strong>

    </div>
    """
)


st.html(
    '<div class="section-title">📈 Machine Overview</div>'
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Current Cycle",
        f'{int(latest["cycle"]):,}'
    )


with col2:

    st.metric(
        "Actual RUL",
        f'{actual_rul:,.0f} cycles'
    )


with col3:

    if predicted_rul is not None:

        st.metric(
            "🤖 LSTM Predicted RUL",
            f'{predicted_rul:,.1f} cycles'
        )

    else:

        st.metric(
            "🤖 LSTM Predicted RUL",
            "Waiting..."
        )


with col4:

    st.metric(
        "Stored Readings",
        f'{reading_count:,}'
    )


st.divider()


st.html(
    '<div class="section-title">🌡️ Live Sensor Measurements</div>'
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "🌡️ Temperature",
        f'{latest["temperature"]:.2f} °C'
    )


with col2:

    st.metric(
        "📳 Vibration",
        f'{latest["vibration"]:.2f}'
    )


with col3:

    st.metric(
        "⚡ Current",
        f'{latest["current"]:.2f} A'
    )


with col4:

    st.metric(
        "🔧 Pressure",
        f'{latest["pressure"]:.2f}'
    )


with col5:

    st.metric(
        "🔄 RPM",
        f'{latest["rpm"]:.2f}'
    )


st.divider()


if not recent_readings.empty:

    recent_readings["timestamp"] = pd.to_datetime(
        recent_readings["timestamp"]
    )

    chart_data = recent_readings.copy()

    st.html(
        '<div class="section-title">📉 Sensor Trends</div>'
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🌡️ Temperature",
            "📳 Vibration",
            "⚡ Current",
            "🔧 Pressure"
        ]
    )


    def create_chart(
        dataframe,
        column,
        title,
        unit
    ):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dataframe["timestamp"],
                y=dataframe[column],
                mode="lines",
                name=title,
                line=dict(
                    width=2
                )
            )
        )

        fig.update_layout(
            title=title,
            height=350,
            template="plotly_dark",
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),
            xaxis_title="Time",
            yaxis_title=unit,
            hovermode="x unified"
        )

        return fig


    with tab1:

        st.plotly_chart(
            create_chart(
                chart_data,
                "temperature",
                "Temperature Trend",
                "°C"
            ),
            use_container_width=True
        )


    with tab2:

        st.plotly_chart(
            create_chart(
                chart_data,
                "vibration",
                "Vibration Trend",
                "Vibration"
            ),
            use_container_width=True
        )


    with tab3:

        st.plotly_chart(
            create_chart(
                chart_data,
                "current",
                "Current Trend",
                "A"
            ),
            use_container_width=True
        )


    with tab4:

        st.plotly_chart(
            create_chart(
                chart_data,
                "pressure",
                "Pressure Trend",
                "Pressure"
            ),
            use_container_width=True
        )


    col1, col2 = st.columns(2)


    with col1:

        st.html(
            '<div class="section-title">🔄 RPM Trend</div>'
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_data["timestamp"],
                y=chart_data["rpm"],
                mode="lines",
                name="RPM"
            )
        )

        fig.update_layout(
            height=350,
            template="plotly_dark",
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        st.html(
            '<div class="section-title">🤖 RUL Prediction</div>'
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_data["timestamp"],
                y=chart_data["rul"],
                mode="lines",
                name="Actual RUL"
            )

        )

        if "predicted_rul" in chart_data.columns:

            prediction_data = chart_data.dropna(
                subset=["predicted_rul"]
            )

            if not prediction_data.empty:

                fig.add_trace(
                    go.Scatter(
                        x=prediction_data["timestamp"],
                        y=prediction_data["predicted_rul"],
                        mode="lines",
                        name="LSTM Predicted RUL"
                    )
                )


        fig.update_layout(
            height=350,
            template="plotly_dark",
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            xaxis_title="Time",
            yaxis_title="Cycles",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


st.divider()


st.html(
    '<div class="section-title">⚠️ Predictive Maintenance Status</div>'
)


if predicted_rul is None:

    st.info(
        "The LSTM model is waiting for enough sensor "
        "readings to generate a prediction."
    )

elif predicted_rul > 5000:

    st.success(
        f"🟢 LSTM prediction: approximately "
        f"{predicted_rul:,.0f} cycles of useful life remaining."
    )

elif predicted_rul > 2000:

    st.warning(
        f"🟡 LSTM prediction: approximately "
        f"{predicted_rul:,.0f} cycles remaining. "
        f"Continue monitoring machine condition."
    )

else:

    st.error(
        f"🔴 LSTM prediction: approximately "
        f"{predicted_rul:,.0f} cycles remaining. "
        f"Maintenance attention is recommended."
    )


st.divider()


st.html(
    '<div class="section-title">📋 Latest Sensor Readings</div>'
)


if not recent_readings.empty:

    display_data = (
        recent_readings
        .sort_values(
            "id",
            ascending=False
        )
        .head(10)
        .copy()
    )

    display_data = display_data[
        [
            "timestamp",
            "machine_id",
            "cycle",
            "temperature",
            "vibration",
            "current",
            "pressure",
            "rpm",
            "rul",
            "predicted_rul"
        ]
    ]

    display_data = display_data.rename(
        columns={
            "timestamp": "Timestamp",
            "machine_id": "Machine",
            "cycle": "Cycle",
            "temperature": "Temperature",
            "vibration": "Vibration",
            "current": "Current",
            "pressure": "Pressure",
            "rpm": "RPM",
            "rul": "Actual RUL",
            "predicted_rul": "LSTM Predicted RUL"
        }
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


st.divider()


st.caption(
    f"🕒 Last reading: {latest['timestamp']}"
)

st.caption(
    "📊 Data pipeline: "
    "Machine Simulator ΓåÆ HiveMQ Cloud ΓåÆ MQTT ΓåÆ "
    "Subscriber ΓåÆ Supabase ΓåÆ Streamlit"
)

st.caption("Dashboard UI build: HTML rendering fix")
