"""
ADAPTIVE-DMS

Real-Time Monitoring Dashboard

Version:
    v1.4

Features:
    - Memory-safe CSV loading
    - Live processed camera frame
    - Fatigue risk
    - Risk level
    - EAR
    - MAR
    - PERCLOS
    - Eye closure duration
    - Gaze direction
    - Head pose
    - Signal reliability
    - Safety intervention
    - Alert count
    - Live graphs
"""

import os
import glob
import csv
import time

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =============================================================
# CONFIGURATION
# =============================================================

MAX_ROWS = 1500

LIVE_FRAME_PATH = os.path.join(
    "logs",
    "live_frame.jpg",
)


# =============================================================
# PAGE CONFIG
# =============================================================

st.set_page_config(
    page_title="ADAPTIVE-DMS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================
# CUSTOM CSS
# =============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 20px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 115px;
    }

    .alert-banner {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        margin: 10px 0 20px 0;
    }

    .alert-safe {
        background: rgba(0, 200, 83, 0.12);
        border: 1px solid rgba(0, 200, 83, 0.35);
    }

    .alert-warning {
        background: rgba(255, 152, 0, 0.16);
        border: 1px solid rgba(255, 152, 0, 0.40);
    }

    .alert-danger {
        background: rgba(244, 67, 54, 0.18);
        border: 1px solid rgba(244, 67, 54, 0.45);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# FIND LATEST SESSION
# =============================================================

def find_latest_session():

    files = glob.glob(
        os.path.join(
            "logs",
            "session_*.csv",
        )
    )

    if not files:

        return None

    return max(
        files,
        key=os.path.getmtime,
    )


# =============================================================
# MEMORY-SAFE CSV READER
# =============================================================

@st.cache_data(ttl=2)
def load_recent_data(
    file_path,
    file_modified_time,
):

    try:

        if not os.path.exists(
            file_path
        ):

            return pd.DataFrame()

        file_size = os.path.getsize(
            file_path
        )

        if file_size <= 0:

            return pd.DataFrame()

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as file:

            header_line = file.readline()

        if not header_line:

            return pd.DataFrame()

        header_line = header_line.strip()

        if not header_line:

            return pd.DataFrame()

        # -----------------------------------------------------
        # Read only last 512 KB
        # -----------------------------------------------------

        max_bytes = 512 * 1024

        bytes_to_read = min(
            file_size,
            max_bytes,
        )

        with open(
            file_path,
            "rb",
        ) as file:

            file.seek(
                -bytes_to_read,
                os.SEEK_END,
            )

            raw_data = file.read()

        text = raw_data.decode(
            "utf-8",
            errors="ignore",
        )

        lines = text.splitlines()

        if len(lines) < 2:

            return pd.DataFrame()

        # First line can be partial.
        data_lines = lines[1:]

        if not data_lines:

            return pd.DataFrame()

        # -----------------------------------------------------
        # Keep latest rows
        # -----------------------------------------------------

        if len(data_lines) > MAX_ROWS:

            data_lines = data_lines[
                -MAX_ROWS:
            ]

        # -----------------------------------------------------
        # Parse header
        # -----------------------------------------------------

        header_reader = csv.reader(
            [header_line]
        )

        headers = next(
            header_reader
        )

        if not headers:

            return pd.DataFrame()

        header_length = len(
            headers
        )

        # -----------------------------------------------------
        # Parse rows
        # -----------------------------------------------------

        reader = csv.reader(
            data_lines
        )

        valid_rows = []

        for row in reader:

            if not row:

                continue

            if len(row) < 5:

                continue

            if len(row) == header_length:

                valid_rows.append(
                    row
                )

            elif len(row) > header_length:

                valid_rows.append(
                    row[
                        :header_length
                    ]
                )

            else:

                padded_row = (
                    row
                    + [""] * (
                        header_length
                        - len(row)
                    )
                )

                valid_rows.append(
                    padded_row
                )

        if not valid_rows:

            return pd.DataFrame()

        df = pd.DataFrame(
            valid_rows,
            columns=headers,
        )

        if df.empty:

            return df

        # -----------------------------------------------------
        # Timestamp
        # -----------------------------------------------------

        if "timestamp" not in df.columns:

            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "timestamp"
            ]
        )

        if df.empty:

            return df

        # -----------------------------------------------------
        # Numeric columns
        # -----------------------------------------------------

        numeric_columns = [

            "ear",

            "mar",

            "perclos",

            "blink_count",

            "blink_duration",

            "microsleep_duration",

            "pitch",

            "yaw",

            "roll",

            "gaze_away_duration",

            "reliability",

            "fatigue_risk",

            "eye_closure_duration",
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        # -----------------------------------------------------
        # Boolean columns
        # -----------------------------------------------------

        boolean_columns = [

            "eyes_closed",

            "microsleep",

            "alert_triggered",
        ]

        for column in boolean_columns:

            if column in df.columns:

                df[column] = (
                    df[column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map(
                        {
                            "true": 1,
                            "false": 0,
                            "1": 1,
                            "0": 0,
                        }
                    )
                    .fillna(0)
                )

        # -----------------------------------------------------
        # Elapsed seconds
        # -----------------------------------------------------

        df["elapsed_seconds"] = (
            df["timestamp"]
            - df["timestamp"].iloc[0]
        ).dt.total_seconds()

        return df

    except (
        OSError,
        PermissionError,
        MemoryError,
        UnicodeError,
        csv.Error,
    ):

        return pd.DataFrame()


# =============================================================
# SAFE FLOAT
# =============================================================

def safe_float(
    value,
    default=0.0,
):

    try:

        if pd.isna(value):

            return default

        return float(value)

    except Exception:

        return default


# =============================================================
# SAFE TEXT
# =============================================================

def safe_text(
    value,
    default="UNKNOWN",
):

    try:

        if pd.isna(value):

            return default

        return str(value)

    except Exception:

        return default


# =============================================================
# HEADER
# =============================================================

st.markdown(
    '<div class="main-title">'
    '🚗 ADAPTIVE-DMS'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Adaptive Multimodal Driver State Monitoring '
    'and Predictive Safety Intervention System'
    '</div>',
    unsafe_allow_html=True,
)


# =============================================================
# FIND SESSION
# =============================================================

session_file = find_latest_session()

if session_file is None:

    st.warning(
        "No session CSV found. "
        "Start main.py first."
    )

    st.stop()


# =============================================================
# FILE MODIFICATION TIME
# =============================================================

try:

    file_modified_time = os.path.getmtime(
        session_file
    )

except Exception:

    file_modified_time = time.time()


# =============================================================
# LOAD DATA
# =============================================================

df = load_recent_data(
    session_file,
    file_modified_time,
)


if df.empty:

    st.warning(
        "Waiting for readable session data..."
    )

    st.info(
        "Keep main.py running and wait a few seconds."
    )

    st.stop()


# =============================================================
# LIVE CAMERA
# =============================================================

st.divider()

st.subheader(
    "🎥 Live Camera Monitoring"
)

if os.path.exists(
    LIVE_FRAME_PATH
):

    try:

        with open(
            LIVE_FRAME_PATH,
            "rb",
        ) as image_file:

            live_frame = image_file.read()

        if live_frame:

            st.image(
                live_frame,
                caption="Processed live camera feed",
                use_container_width=True,
            )

        else:

            st.info(
                "Waiting for camera frame..."
            )

    except Exception:

        st.warning(
            "Unable to read live camera frame."
        )

else:

    st.info(
        "Waiting for main.py to start the camera..."
    )


# =============================================================
# LATEST DATA
# =============================================================

latest = df.iloc[-1]


# =============================================================
# CURRENT VALUES
# =============================================================

fatigue_risk = safe_float(
    latest.get(
        "fatigue_risk",
        0.0,
    )
)

ear = safe_float(
    latest.get(
        "ear",
        0.0,
    )
)

mar = safe_float(
    latest.get(
        "mar",
        0.0,
    )
)

perclos = safe_float(
    latest.get(
        "perclos",
        0.0,
    )
)

reliability = safe_float(
    latest.get(
        "reliability",
        0.0,
    )
)

eye_closure_duration = safe_float(
    latest.get(
        "eye_closure_duration",
        0.0,
    )
)

pitch = safe_float(
    latest.get(
        "pitch",
        0.0,
    )
)

yaw = safe_float(
    latest.get(
        "yaw",
        0.0,
    )
)

roll = safe_float(
    latest.get(
        "roll",
        0.0,
    )
)


# =============================================================
# CURRENT TEXT VALUES
# =============================================================

risk_level = safe_text(
    latest.get(
        "risk_level",
        "UNKNOWN",
    )
).upper()

temporal_state = safe_text(
    latest.get(
        "temporal_state",
        "UNKNOWN",
    )
)

intervention_level = safe_text(
    latest.get(
        "intervention_level",
        "UNKNOWN",
    )
)

intervention_action = safe_text(
    latest.get(
        "intervention_action",
        "UNKNOWN",
    )
)

gaze_direction = safe_text(
    latest.get(
        "gaze_direction",
        "UNKNOWN",
    )
)


# =============================================================
# ALERT COUNT
# =============================================================

if "alert_triggered" in df.columns:

    alert_count = int(
        pd.to_numeric(
            df[
                "alert_triggered"
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

else:

    alert_count = 0


# =============================================================
# ALERT BANNER
# =============================================================

if risk_level in (
    "HIGH",
    "CRITICAL",
):

    alert_class = "alert-danger"

    alert_text = (
        "🚨 HIGH RISK — "
        "DRIVER ATTENTION REQUIRED"
    )

elif risk_level == "MODERATE":

    alert_class = "alert-warning"

    alert_text = (
        "⚠️ MODERATE RISK — "
        "MONITOR DRIVER STATE"
    )

elif risk_level == "LOW":

    alert_class = "alert-warning"

    alert_text = (
        "🟡 LOW RISK — "
        "DRIVER CONDITION REQUIRES MONITORING"
    )

else:

    alert_class = "alert-safe"

    alert_text = (
        "🟢 DRIVER STATE STABLE"
    )


st.markdown(
    f'<div class="alert-banner '
    f'{alert_class}">'
    f'{alert_text}'
    f'</div>',
    unsafe_allow_html=True,
)


# =============================================================
# RISK OVERVIEW
# =============================================================

left, right = st.columns(
    [1, 2]
)


with left:

    st.subheader(
        "🎯 Current Risk"
    )

    st.metric(
        "Risk Level",
        risk_level,
    )

    st.metric(
        "Fatigue Risk",
        f"{fatigue_risk:.2f}",
    )

    if risk_level in (
        "HIGH",
        "CRITICAL",
    ):

        st.error(
            "🚨 Driver attention required"
        )

    elif risk_level == "MODERATE":

        st.warning(
            "⚠️ Moderate fatigue risk"
        )

    elif risk_level == "LOW":

        st.warning(
            "🟡 Low fatigue risk"
        )

    else:

        st.success(
            "🟢 Driver state normal"
        )


with right:

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:

        st.metric(
            "👁️ EAR",
            f"{ear:.3f}",
        )

    with c2:

        st.metric(
            "📊 PERCLOS",
            f"{perclos:.3f}",
        )

    with c3:

        st.metric(
            "😮 MAR",
            f"{mar:.3f}",
        )

    with c4:

        st.metric(
            "🚨 Alerts",
            alert_count,
        )


st.divider()


# =============================================================
# DRIVER STATUS
# =============================================================

st.subheader(
    "🧠 Driver Monitoring Status"
)

c1, c2, c3, c4 = st.columns(
    4
)


with c1:

    st.markdown(
        f"""
        <div class="status-card">
        <b>Temporal State</b>
        <br><br>
        {temporal_state}
        </div>
        """,
        unsafe_allow_html=True,
    )


with c2:

    st.markdown(
        f"""
        <div class="status-card">
        <b>Intervention</b>
        <br><br>
        {intervention_level}
        </div>
        """,
        unsafe_allow_html=True,
    )


with c3:

    st.markdown(
        f"""
        <div class="status-card">
        <b>Gaze Direction</b>
        <br><br>
        👀 {gaze_direction}
        </div>
        """,
        unsafe_allow_html=True,
    )


with c4:

    st.markdown(
        f"""
        <div class="status-card">
        <b>Signal Reliability</b>
        <br><br>
        📡 {reliability:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================
# EYE CLOSURE
# =============================================================

st.subheader(
    "👁️ Eye Closure Monitoring"
)

eye_col1, eye_col2 = st.columns(
    2
)


with eye_col1:

    st.metric(
        "Current Eye Closure",
        f"{eye_closure_duration:.2f} sec",
    )


with eye_col2:

    if eye_closure_duration >= 1.5:

        st.error(
            "🚨 Eye closure threshold reached"
        )

    elif eye_closure_duration > 0:

        st.warning(
            "⚠️ Eyes currently closed"
        )

    else:

        st.success(
            "🟢 Eyes open"
        )


# =============================================================
# FATIGUE RISK GRAPH
# =============================================================

st.divider()

st.subheader(
    "📈 Live Fatigue Risk"
)

if "fatigue_risk" in df.columns:

    fig_risk = go.Figure()

    fig_risk.add_trace(
        go.Scatter(
            x=df[
                "elapsed_seconds"
            ],
            y=df[
                "fatigue_risk"
            ],
            mode="lines",
            name="Fatigue Risk",
            line=dict(
                width=3
            ),
        )
    )

    for threshold, label in [
        (0.20, "Low"),
        (0.35, "Moderate"),
        (0.55, "High"),
        (0.75, "Critical"),
    ]:

        fig_risk.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text=label,
        )

    fig_risk.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Fatigue Risk",
        yaxis_range=[
            0,
            1,
        ],
        height=420,
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )


# =============================================================
# EAR + PERCLOS
# =============================================================

c1, c2 = st.columns(
    2
)


with c1:

    st.subheader(
        "👁️ Eye Aspect Ratio"
    )

    if "ear" in df.columns:

        fig_ear = go.Figure()

        fig_ear.add_trace(
            go.Scatter(
                x=df[
                    "elapsed_seconds"
                ],
                y=df[
                    "ear"
                ],
                mode="lines",
                name="EAR",
            )
        )

        fig_ear.add_hline(
            y=0.21,
            line_dash="dash",
            annotation_text="EAR Threshold",
        )

        fig_ear.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="EAR",
            height=350,
        )

        st.plotly_chart(
            fig_ear,
            use_container_width=True,
        )


with c2:

    st.subheader(
        "📊 PERCLOS"
    )

    if "perclos" in df.columns:

        fig_perclos = go.Figure()

        fig_perclos.add_trace(
            go.Scatter(
                x=df[
                    "elapsed_seconds"
                ],
                y=df[
                    "perclos"
                ],
                mode="lines",
                name="PERCLOS",
            )
        )

        fig_perclos.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="PERCLOS",
            height=350,
        )

        st.plotly_chart(
            fig_perclos,
            use_container_width=True,
        )


# =============================================================
# HEAD POSE
# =============================================================

st.subheader(
    "🧭 Head Pose"
)

fig_pose = go.Figure()


if "pitch" in df.columns:

    fig_pose.add_trace(
        go.Scatter(
            x=df[
                "elapsed_seconds"
            ],
            y=df[
                "pitch"
            ],
            mode="lines",
            name="Pitch",
        )
    )


if "yaw" in df.columns:

    fig_pose.add_trace(
        go.Scatter(
            x=df[
                "elapsed_seconds"
            ],
            y=df[
                "yaw"
            ],
            mode="lines",
            name="Yaw",
        )
    )


if "roll" in df.columns:

    fig_pose.add_trace(
        go.Scatter(
            x=df[
                "elapsed_seconds"
            ],
            y=df[
                "roll"
            ],
            mode="lines",
            name="Roll",
        )
    )


fig_pose.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Angle (degrees)",
    height=400,
)

st.plotly_chart(
    fig_pose,
    use_container_width=True,
)


# =============================================================
# RELIABILITY
# =============================================================

st.subheader(
    "📡 Signal Reliability"
)

if "reliability" in df.columns:

    fig_reliability = go.Figure()

    fig_reliability.add_trace(
        go.Scatter(
            x=df[
                "elapsed_seconds"
            ],
            y=df[
                "reliability"
            ],
            mode="lines",
            name="Reliability",
        )
    )

    fig_reliability.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Reliability",
        yaxis_range=[
            0,
            1,
        ],
        height=350,
    )

    st.plotly_chart(
        fig_reliability,
        use_container_width=True,
    )


# =============================================================
# SAFETY INTERVENTION
# =============================================================

st.divider()

st.subheader(
    "🚨 Safety Intervention"
)

c1, c2, c3 = st.columns(
    3
)


with c1:

    st.metric(
        "Intervention Level",
        intervention_level,
    )


with c2:

    st.metric(
        "Action",
        intervention_action,
    )


with c3:

    st.metric(
        "Total Alerts",
        alert_count,
    )


# =============================================================
# CURRENT HEAD POSE
# =============================================================

st.subheader(
    "🧭 Current Head Pose"
)

c1, c2, c3 = st.columns(
    3
)


with c1:

    st.metric(
        "Pitch",
        f"{pitch:.1f}°",
    )


with c2:

    st.metric(
        "Yaw",
        f"{yaw:.1f}°",
    )


with c3:

    st.metric(
        "Roll",
        f"{roll:.1f}°",
    )


# =============================================================
# SESSION INFORMATION
# =============================================================

st.divider()

st.subheader(
    "📋 Session Information"
)

duration = safe_float(
    df[
        "elapsed_seconds"
    ].iloc[-1]
)

c1, c2, c3 = st.columns(
    3
)


with c1:

    st.write(
        "**Session:** "
        f"`{os.path.basename(session_file)}`"
    )


with c2:

    st.write(
        f"**Displayed duration:** "
        f"{duration:.1f} seconds"
    )


with c3:

    st.write(
        f"**Displayed samples:** "
        f"{len(df)}"
    )


# =============================================================
# FOOTER
# =============================================================

st.divider()

st.caption(
    "ADAPTIVE-DMS v1.4 | "
    "Live Camera + Memory-Safe Dashboard"
)

st.caption(
    f"Displaying at most {MAX_ROWS} recent samples."
)

st.caption(
    "Camera is owned by main.py. "
    "Dashboard reads the processed frame."
)


# =============================================================
# AUTO REFRESH
# =============================================================

time.sleep(2)

st.rerun()