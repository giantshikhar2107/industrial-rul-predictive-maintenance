import os
import pandas as pd
import streamlit as st
from supabase import create_client
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Industrial Machine Monitoring",
    page_icon="⚙️",
    layout="wide"
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

        load_dotenv()

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_PUBLISHABLE_KEY")

    if not supabase_url or not supabase_key:
        st.error("Supabase credentials are not configured.")
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
        .select("machine_id,machine_name,machine_type,location")
        .limit(1)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    return pd.DataFrame(response.data)


def get_latest_reading():
    response = (
        supabase
        .table("sensor_readings")
        .select(
            "id,machine_id,timestamp,cycle,"
            "temperature,vibration,current,pressure,rpm,"
            "degradation,rul"
        )
        .order("id", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    return pd.DataFrame(response.data)


def get_recent_readings():
    response = (
        supabase
        .table("sensor_readings")
        .select(
            "id,machine_id,timestamp,cycle,"
            "temperature,vibration,current,pressure,rpm,"
            "degradation,rul"
        )
        .order("id", desc=True)
        .limit(100)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    data = pd.DataFrame(response.data)

    data = data.sort_values("id")

    return data


def get_reading_count():
    response = (
        supabase
        .table("sensor_readings")
        .select("id", count="exact")
        .execute()
    )

    return response.count if response.count is not None else 0


st.title("⚙️ Industrial Machine Monitoring")

st.write(
    "Real-time monitoring of industrial machine sensor data "
    "using MQTT, HiveMQ Cloud, Supabase and Streamlit."
)


try:
    machine_info = get_machine_info()
    latest_reading = get_latest_reading()
    recent_readings = get_recent_readings()
    reading_count = get_reading_count()

except Exception as e:
    st.error("Unable to connect to Supabase.")
    st.code(str(e))
    st.stop()


if machine_info.empty:
    st.warning("No machine information found in Supabase.")
    st.stop()


if latest_reading.empty:
    st.warning("No sensor readings found in Supabase.")
    st.stop()


machine = machine_info.iloc[0]
latest = latest_reading.iloc[0]


st.subheader("Machine Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Machine ID",
        machine["machine_id"]
    )

with col2:
    st.metric(
        "Machine Name",
        machine["machine_name"]
    )

with col3:
    st.metric(
        "Machine Type",
        machine["machine_type"]
    )

with col4:
    st.metric(
        "Location",
        machine["location"]
    )


st.divider()


st.subheader("Current Machine Condition")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Temperature",
        f'{latest["temperature"]:.2f} °C'
    )

with col2:
    st.metric(
        "Vibration",
        f'{latest["vibration"]:.2f}'
    )

with col3:
    st.metric(
        "Current",
        f'{latest["current"]:.2f} A'
    )

with col4:
    st.metric(
        "Pressure",
        f'{latest["pressure"]:.2f}'
    )

with col5:
    st.metric(
        "RPM",
        f'{latest["rpm"]:.2f}'
    )


st.divider()


st.subheader("Machine Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Cycle",
        int(latest["cycle"])
    )

with col2:
    st.metric(
        "Remaining Useful Life",
        f'{latest["rul"]:.0f} cycles'
    )

with col3:
    st.metric(
        "Stored Readings",
        reading_count
    )


st.divider()


if not recent_readings.empty:

    st.subheader("Sensor Trends")

    recent_readings["timestamp"] = pd.to_datetime(
        recent_readings["timestamp"]
    )

    chart_data = recent_readings.set_index("timestamp")

    st.line_chart(
        chart_data[
            [
                "temperature",
                "vibration",
                "current",
                "pressure"
            ]
        ]
    )


    st.subheader("RPM Trend")

    st.line_chart(
        chart_data[
            ["rpm"]
        ]
    )


    st.subheader("Remaining Useful Life Trend")

    st.line_chart(
        chart_data[
            ["rul"]
        ]
    )


st.divider()


st.subheader("Latest Sensor Readings")

if not recent_readings.empty:

    display_data = recent_readings.sort_values(
        "id",
        ascending=False
    ).head(10)

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
            "rul"
        ]
    ]

    st.dataframe(
        display_data,
        use_container_width=True
    )


st.caption(
    f"Last reading timestamp: {latest['timestamp']}"
)

st.caption(
    "Data source: HiveMQ Cloud → MQTT → Supabase → Streamlit"
)

st.button(
    "Refresh Dashboard"
)