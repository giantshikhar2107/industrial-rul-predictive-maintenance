import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NASA C-MAPSS Live Monitoring",
    page_icon="✈️",
    layout="wide"
)

MACHINE_ID = "NASA_FD001_001"
REFRESH_SECONDS = 10

PAGE_DIR = Path(__file__).resolve().parent

# Find the project root and existing database.
DB_PATH = None

for directory in [PAGE_DIR, *PAGE_DIR.parents]:
    candidate = directory / "database" / "machine_data.db"

    if candidate.is_file():
        DB_PATH = candidate
        break


# ==================================================
# DATABASE FUNCTIONS
# ==================================================

@st.cache_data(ttl=3)
def load_nasa_data(db_path, machine_id):
    """Load NASA engine readings from SQLite."""

    if not Path(db_path).is_file():
        return pd.DataFrame(), "Database file not found."

    try:
        with sqlite3.connect(db_path, timeout=10) as connection:

            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                connection
            )["name"].tolist()

            if "sensor_readings" not in tables:
                return (
                    pd.DataFrame(),
                    "The sensor_readings table was not found."
                )

            columns = pd.read_sql_query(
                "PRAGMA table_info(sensor_readings)",
                connection
            )["name"].tolist()

            required_columns = {
                "id",
                "machine_id",
                "timestamp",
                "cycle",
                "rul",
                "predicted_rul"
            }

            missing_columns = required_columns - set(columns)

            if missing_columns:
                return (
                    pd.DataFrame(),
                    "Missing columns: "
                    + ", ".join(sorted(missing_columns))
                )

            data = pd.read_sql_query(
                """
                SELECT *
                FROM sensor_readings
                WHERE machine_id = ?
                ORDER BY id ASC
                """,
                connection,
                params=(machine_id,)
            )

        if data.empty:
            return (
                data,
                f"No readings found for {machine_id}."
            )

        for column in [
            "cycle",
            "rul",
            "predicted_rul",
            "max_cycles",
            "engine_id"
        ]:
            if column in data.columns:
                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
                )

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
            utc=True
        )

        data = data.dropna(subset=["cycle"])
        data = data.sort_values("id")

        return data, None

    except (sqlite3.Error, ValueError, OSError) as error:
        return pd.DataFrame(), f"Database error: {error}"


def initialize_alert_table(db_path):
    """Create the alert table without modifying sensor readings."""

    with sqlite3.connect(db_path, timeout=10) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_reading_id INTEGER NOT NULL,
                machine_id TEXT NOT NULL,
                cycle INTEGER NOT NULL,
                predicted_rul REAL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_reading_id, alert_type)
            )
        """)


def save_alert_history(db_path, data):
    """Save each reading's warning or critical status once."""

    if data.empty:
        return

    initialize_alert_table(db_path)

    required = {
        "id",
        "machine_id",
        "cycle",
        "predicted_rul"
    }

    if not required.issubset(data.columns):
        return

    alerts_to_save = []

    for _, row in data.iterrows():
        predicted_rul = row["predicted_rul"]

        if pd.isna(predicted_rul):
            continue

        if predicted_rul <= 10:
            alert_type = "CRITICAL"
            message = (
                f"Predicted RUL is {predicted_rul:.2f} cycles."
            )

        elif predicted_rul <= 30:
            alert_type = "WARNING"
            message = (
                f"Predicted RUL is {predicted_rul:.2f} cycles."
            )

        else:
            continue

        alerts_to_save.append((
            int(row["id"]),
            str(row["machine_id"]),
            int(row["cycle"]),
            float(predicted_rul),
            alert_type,
            message
        ))

    if not alerts_to_save:
        return

    with sqlite3.connect(db_path, timeout=10) as connection:
        connection.executemany("""
            INSERT OR IGNORE INTO alert_history (
                source_reading_id,
                machine_id,
                cycle,
                predicted_rul,
                alert_type,
                message
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, alerts_to_save)


def load_alert_history(db_path, limit=100):
    """Load recent alerts and total alert counts."""

    with sqlite3.connect(db_path, timeout=10) as connection:

        initialize_alert_table(db_path)

        alerts = pd.read_sql_query(
            """
            SELECT
                created_at,
                machine_id,
                cycle,
                predicted_rul,
                alert_type,
                message
            FROM alert_history
            ORDER BY id DESC
            LIMIT ?
            """,
            connection,
            params=(limit,)
        )

        totals = pd.read_sql_query(
            """
            SELECT
                alert_type,
                COUNT(*) AS total
            FROM alert_history
            GROUP BY alert_type
            """,
            connection
        )

    return alerts, totals


# ==================================================
# PAGE HEADER
# ==================================================

st.title("✈️ NASA C-MAPSS Engine Monitoring")

st.caption(
    "LSTM Remaining Useful Life prediction | "
    "HiveMQ MQTT | SQLite"
)

st.info(
    "Demonstration mode: NASA C-MAPSS training data is replayed "
    "through MQTT. These are recorded dataset cycles, not live "
    "measurements from a physical engine."
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("Monitoring Settings")

    st.write(f"**Engine:** {MACHINE_ID}")
    st.write(f"**Refresh interval:** {REFRESH_SECONDS} seconds")

    if DB_PATH is not None:
        st.success("Database file found")
    else:
        st.error("Database file not found")

    if st.button("Refresh data now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ==================================================
# LIVE DASHBOARD
# ==================================================

@st.fragment(run_every=REFRESH_SECONDS)
def live_dashboard():

    if DB_PATH is None:
        st.error(
            "Could not find database/machine_data.db. "
            "Check your project folder."
        )
        return

    data, error = load_nasa_data(
        str(DB_PATH),
        MACHINE_ID
    )

    with st.expander("Database diagnostics"):

        st.write("Database path:", str(DB_PATH))
        st.write("Database exists:", DB_PATH.is_file())

        if error:
            st.write("Query status:", error)
        else:
            st.success("Database query successful")
            st.write("NASA readings found:", len(data))

    if error:
        st.warning(error)
        return

    # Save warning and critical events.
    try:
        save_alert_history(DB_PATH, data)
    except sqlite3.Error as error:
        st.error(f"Could not save alert history: {error}")

    latest = data.iloc[-1]

    current_cycle = int(latest["cycle"])
    actual_rul = latest["rul"]
    predicted_rul = latest["predicted_rul"]

    # --------------------------------------------------
    # CURRENT ENGINE STATUS
    # --------------------------------------------------

    st.subheader("Current Engine Status")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Cycle",
        current_cycle
    )

    col2.metric(
        "Predicted RUL",
        f"{predicted_rul:.2f} cycles"
        if pd.notna(predicted_rul)
        else "N/A"
    )

    col3.metric(
        "Actual RUL",
        f"{actual_rul:.1f} cycles"
        if pd.notna(actual_rul)
        else "N/A"
    )

    if pd.notna(actual_rul) and pd.notna(predicted_rul):
        prediction_error = abs(actual_rul - predicted_rul)

        col4.metric(
            "Absolute Prediction Error",
            f"{prediction_error:.2f} cycles"
        )
    else:
        col4.metric(
            "Absolute Prediction Error",
            "N/A"
        )

    st.caption(
        "Actual RUL is derived from the recorded training engine's "
        "known final cycle. In real deployment, this ground truth "
        "would generally not be known during operation."
    )

    st.divider()

    # --------------------------------------------------
    # ENGINE HEALTH
    # --------------------------------------------------

    st.subheader("Engine Health Indicator")

    if pd.isna(predicted_rul):
        st.warning("A valid RUL prediction is not available.")

    elif predicted_rul <= 10:
        st.error(
            "CRITICAL — Predicted RUL is 10 cycles or less."
        )

    elif predicted_rul <= 30:
        st.warning(
            "WARNING — Predicted RUL is between 11 and 30 cycles."
        )

    else:
        st.success(
            "NORMAL — Predicted RUL is above 30 cycles."
        )

    st.caption(
        "These thresholds are illustrative and are not validated "
        "maintenance limits or safety criteria."
    )

    st.divider()

    # --------------------------------------------------
    # RUL TREND
    # --------------------------------------------------

    st.subheader("Actual vs Predicted RUL")

    trend_data = data.dropna(
        subset=["rul", "predicted_rul"]
    ).copy()

    if not trend_data.empty:

        trend = (
            trend_data.groupby("cycle", as_index=False)
            .agg(
                Actual_RUL=("rul", "last"),
                Predicted_RUL=("predicted_rul", "last")
            )
            .sort_values("cycle")
        )

        st.line_chart(
            trend.set_index("cycle")[
                ["Actual_RUL", "Predicted_RUL"]
            ],
            height=350
        )

    else:
        st.info("RUL trend is not available yet.")

    st.divider()

    # --------------------------------------------------
    # SENSOR TREND
    # --------------------------------------------------

    st.subheader("Sensor Trend Analysis")

    sensor_columns = [
        f"sensor_{number}"
        for number in range(1, 22)
    ]

    available_sensors = [
        column
        for column in sensor_columns
        if column in data.columns
        and data[column].notna().any()
    ]

    if available_sensors:

        selected_sensor = st.selectbox(
            "Select sensor",
            available_sensors,
            key="nasa_sensor_selector"
        )

        sensor_data = data[
            ["cycle", selected_sensor]
        ].copy()

        sensor_data[selected_sensor] = pd.to_numeric(
            sensor_data[selected_sensor],
            errors="coerce"
        )

        sensor_data = sensor_data.dropna(
            subset=["cycle", selected_sensor]
        )

        sensor_data = (
            sensor_data.groupby("cycle", as_index=False)
            .last()
            .sort_values("cycle")
        )

        st.line_chart(
            sensor_data.set_index("cycle"),
            height=300
        )

    else:
        st.info("No populated NASA sensor columns were found.")

    st.divider()

    # --------------------------------------------------
    # ALERT HISTORY
    # --------------------------------------------------

    st.subheader("🚨 Alert History")

    try:
        alerts, totals = load_alert_history(DB_PATH)

        critical_count = 0
        warning_count = 0

        if not totals.empty:
            for _, item in totals.iterrows():
                if item["alert_type"] == "CRITICAL":
                    critical_count = int(item["total"])

                elif item["alert_type"] == "WARNING":
                    warning_count = int(item["total"])

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Critical Alerts",
            critical_count
        )

        col2.metric(
            "Total Warnings",
            warning_count
        )

        if alerts.empty:
            st.success("No warning or critical alerts recorded.")

        else:
            st.dataframe(
                alerts,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Recent Alert History",
                data=alerts.to_csv(index=False).encode("utf-8"),
                file_name="nasa_alert_history.csv",
                mime="text/csv",
                key="download_nasa_alerts"
            )

    except sqlite3.Error as error:
        st.error(f"Could not load alert history: {error}")

    st.divider()

    # --------------------------------------------------
    # RECENT READINGS
    # --------------------------------------------------

    st.subheader("Recent Engine Readings")

    display_columns = [
        column
        for column in [
            "timestamp",
            "engine_id",
            "cycle",
            "rul",
            "predicted_rul"
        ]
        if column in data.columns
    ]

    recent = (
        data.tail(20)[display_columns]
        .sort_values("cycle", ascending=False)
    )

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # DOWNLOAD READINGS
    # --------------------------------------------------

    download_data = data.drop(
        columns=["id"],
        errors="ignore"
    )

    st.download_button(
        "Download NASA Readings as CSV",
        data=download_data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="nasa_fd001_live_readings.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_nasa_readings"
    )

    timestamp = latest.get("timestamp")

    if pd.notna(timestamp):
        st.caption(
            "Latest saved timestamp: "
            + timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        )

    st.caption(
        f"Total database readings for this engine: {len(data)}"
    )


live_dashboard()

