import os
import sqlite3
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_FILE = os.path.join(
    BASE_DIR,
    "database",
    "machine_data.db"
)


def get_connection():
    return sqlite3.connect(DATABASE_FILE)


def get_machine_info():

    connection = get_connection()

    query = """
    SELECT machine_id, machine_name, machine_type, location
    FROM machines
    LIMIT 1
    """

    data = pd.read_sql_query(query, connection)

    connection.close()

    return data


def get_latest_reading():

    connection = get_connection()

    query = """
    SELECT
        machine_id,
        timestamp,
        cycle,
        temperature,
        vibration,
        current,
        pressure,
        rpm,
        rul
    FROM sensor_readings
    ORDER BY id DESC
    LIMIT 1
    """

    data = pd.read_sql_query(query, connection)

    connection.close()

    return data


def get_recent_readings():

    connection = get_connection()

    query = """
    SELECT
        timestamp,
        cycle,
        temperature,
        vibration,
        current,
        pressure,
        rpm,
        rul
    FROM sensor_readings
    ORDER BY id DESC
    LIMIT 100
    """

    data = pd.read_sql_query(query, connection)

    connection.close()

    return data.sort_values("cycle")


def get_reading_count():

    connection = get_connection()

    query = """
    SELECT COUNT(*) AS total
    FROM sensor_readings
    """

    data = pd.read_sql_query(query, connection)

    connection.close()

    return int(data.iloc[0]["total"])


st.set_page_config(
    page_title="Industrial Machine Monitoring",
    page_icon="⚙️",
    layout="wide"
)
st_autorefresh(
    interval=2000,
    key="dashboard_refresh"
)


st.title("Industrial Machine Monitoring Dashboard")

st.write(
    "Real-time monitoring of machine condition using "
    "HiveMQ Cloud, MQTT and SQLite."
)


if not os.path.exists(DATABASE_FILE):

    st.error(
        "SQLite database not found. "
        "Start the MQTT subscriber first."
    )

    st.stop()


machine_info = get_machine_info()
latest = get_latest_reading()
recent = get_recent_readings()
total_readings = get_reading_count()


if machine_info.empty or latest.empty:

    st.warning(
        "No machine data is available in the database yet."
    )

    st.stop()


machine = machine_info.iloc[0]
reading = latest.iloc[0]


st.subheader("Machine Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Machine ID",
        machine["machine_id"]
    )

with col2:

    st.metric(
        "Machine",
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


st.subheader("Current Machine Condition")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Temperature",
        f"{reading['temperature']:.2f} °C"
    )

with col2:

    st.metric(
        "Vibration",
        f"{reading['vibration']:.2f}"
    )

with col3:

    st.metric(
        "Current",
        f"{reading['current']:.2f} A"
    )

with col4:

    st.metric(
        "Pressure",
        f"{reading['pressure']:.2f} bar"
    )

with col5:

    st.metric(
        "RPM",
        f"{reading['rpm']:.2f}"
    )


st.subheader("Machine Status")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Current Cycle",
        int(reading["cycle"])
    )

with col2:

    st.metric(
        "Remaining Useful Life",
        f"{reading['rul']:.0f} cycles"
    )

with col3:

    st.metric(
        "Stored Readings",
        total_readings
    )


st.subheader("Machine Sensor Trends")

chart_data = recent.set_index("cycle")

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
        [
            "rpm"
        ]
    ]
)


st.subheader("RUL Trend")

st.line_chart(
    chart_data[
        [
            "rul"
        ]
    ]
)


st.subheader("Latest Database Reading")

st.dataframe(
    latest,
    use_container_width=True
)


st.caption(
    f"Data source: SQLite database | "
    f"Last cycle: {int(reading['cycle'])} | "
    f"Last update: {reading['timestamp']}"
)


if st.button("Refresh Dashboard"):

    st.rerun()