import os
import time

import pandas as pd
import streamlit as st
import plotly.express as px


CSV_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "machine_data.csv"
    )
)


st.set_page_config(
    page_title="Industrial Machine Monitoring",
    page_icon="⚙️",
    layout="wide"
)
time.sleep(1)


st.title("Industrial Machine Monitoring Dashboard")

st.write(
    "Real-time monitoring of machine condition using MQTT sensor data."
)


if not os.path.exists(CSV_FILE):

    st.error("Machine data file not found.")

    st.stop()


data = pd.read_csv(CSV_FILE)


if data.empty:

    st.warning("No machine data available.")

    st.stop()


data["timestamp"] = pd.to_datetime(
    data["timestamp"]
)


latest = data.iloc[-1]


st.subheader("Machine Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Machine ID",
        latest["machine_id"]
    )

with col2:
    st.metric(
        "Machine Type",
        latest["machine_type"]
    )

with col3:
    st.metric(
        "Current Cycle",
        int(latest["cycle"])
    )


st.subheader("Current Machine Condition")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Temperature",
        f"{latest['temperature']:.2f} °C"
    )

with col2:
    st.metric(
        "Vibration",
        f"{latest['vibration']:.2f}"
    )

with col3:
    st.metric(
        "Current",
        f"{latest['current']:.2f} A"
    )

with col4:
    st.metric(
        "Pressure",
        f"{latest['pressure']:.2f} bar"
    )

with col5:
    st.metric(
        "RPM",
        f"{latest['rpm']:.0f}"
    )


st.subheader("Remaining Useful Life")

st.metric(
    "RUL",
    f"{int(latest['rul'])} cycles"
)


st.subheader("Machine Sensor Trends")


temperature_fig = px.line(
    data,
    x="timestamp",
    y="temperature",
    title="Temperature vs Time"
)

st.plotly_chart(
    temperature_fig,
    use_container_width=True
)


vibration_fig = px.line(
    data,
    x="timestamp",
    y="vibration",
    title="Vibration vs Time"
)

st.plotly_chart(
    vibration_fig,
    use_container_width=True
)


current_fig = px.line(
    data,
    x="timestamp",
    y="current",
    title="Current vs Time"
)

st.plotly_chart(
    current_fig,
    use_container_width=True
)


rpm_fig = px.line(
    data,
    x="timestamp",
    y="rpm",
    title="RPM vs Time"
)

st.plotly_chart(
    rpm_fig,
    use_container_width=True
)


rul_fig = px.line(
    data,
    x="timestamp",
    y="rul",
    title="Remaining Useful Life vs Time"
)

st.plotly_chart(
    rul_fig,
    use_container_width=True
)


st.subheader("Latest Sensor Data")

st.dataframe(
    data.tail(20),
    use_container_width=True
)
time.sleep(2)
st.rerun()