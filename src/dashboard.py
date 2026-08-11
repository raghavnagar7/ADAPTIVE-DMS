"""
ADAPTIVE-DMS
Real-Time Monitoring Dashboard

v1.2

Reads the latest ADAPTIVE-DMS session CSV and displays:
- Fatigue risk
- EAR
- PERCLOS
- MAR
- Risk level
- Reliability
- Head pose
- Gaze
- Intervention
- Alerts
"""

import os
import glob

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =============================================================
# PAGE CONFIGURATION
# =============================================================

st.set_page_config(
    page_title="ADAPTIVE-DMS Dashboard",
    page_icon="🚗",
    layout="wide",
)


# =============================================================
# FIND LATEST CSV
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
# LOAD DATA
# =============================================================

@st.cache_data(ttl=2)
def load_data(file_path):

    df = pd.read_csv(
        file_path
    )

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["elapsed_seconds"] = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


# =============================================================
# TITLE
# =============================================================

st.title(
    "🚗 ADAPTIVE-DMS"
)

st.subheader(
    "Adaptive Multimodal Driver State Monitoring "
    "and Predictive Safety Intervention System"
)

st.divider()


# =============================================================
# FIND SESSION
# =============================================================

session_file = find_latest_session()

if session_file is None:

    st.warning(
        "No session CSV found. "
        "Run main.py first to create a session log."
    )

    st.stop()


# =============================================================
# LOAD
# =============================================================

df = load_data(
    session_file
)

if df.empty:

    st.error(
        "The session CSV is empty."
    )

    st.stop()


# =============================================================
# LATEST DATA
# =============================================================

latest = df.iloc[-1]


fatigue_risk = float(
    latest.get(
        "fatigue_risk",
        0.0,
    )
)

ear = float(
    latest.get(
        "ear",
        0.0,
    )
)

mar = float(
    latest.get(
        "mar",
        0.0,
    )
)

perclos = float(
    latest.get(
        "perclos",
        0.0,
    )
)

reliability = float(
    latest.get(
        "reliability",
        0.0,
    )
)

risk_level = str(
    latest.get(
        "risk_level",
        "UNKNOWN",
    )
)

temporal_state = str(
    latest.get(
        "temporal_state",
        "UNKNOWN",
    )
)

intervention_level = str(
    latest.get(
        "intervention_level",
        "UNKNOWN",
    )
)

intervention_action = str(
    latest.get(
        "intervention_action",
        "UNKNOWN",
    )
)

gaze_direction = str(
    latest.get(
        "gaze_direction",
        "UNKNOWN",
    )
)

pitch = float(
    latest.get(
        "pitch",
        0.0,
    )
)

yaw = float(
    latest.get(
        "yaw",
        0.0,
    )
)

roll = float(
    latest.get(
        "roll",
        0.0,
    )
)

alert_count = int(
    df["alert_triggered"]
    .astype(bool)
    .sum()
)


# =============================================================
# TOP METRICS
# =============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Fatigue Risk",
        f"{fatigue_risk:.2f}",
    )


with col2:

    st.metric(
        "Risk Level",
        risk_level,
    )


with col3:

    st.metric(
        "EAR",
        f"{ear:.3f}",
    )


with col4:

    st.metric(
        "PERCLOS",
        f"{perclos:.3f}",
    )


with col5:

    st.metric(
        "Alerts",
        alert_count,
    )


st.divider()


# =============================================================
# STATUS COLUMNS
# =============================================================

left, middle, right = st.columns(3)


with left:

    st.markdown(
        "### 🧠 Driver State"
    )

    st.write(
        f"**Temporal State:** {temporal_state}"
    )

    st.write(
        f"**Intervention:** {intervention_level}"
    )

    st.write(
        f"**Action:** {intervention_action}"
    )


with middle:

    st.markdown(
        "### 👀 Vision"
    )

    st.write(
        f"**Gaze:** {gaze_direction}"
    )

    st.write(
        f"**MAR:** {mar:.3f}"
    )

    st.write(
        f"**Reliability:** {reliability:.2f}"
    )


with right:

    st.markdown(
        "### 🧭 Head Pose"
    )

    st.write(
        f"**Pitch:** {pitch:.1f}°"
    )

    st.write(
        f"**Yaw:** {yaw:.1f}°"
    )

    st.write(
        f"**Roll:** {roll:.1f}°"
    )


st.divider()


# =============================================================
# FATIGUE RISK GRAPH
# =============================================================

st.subheader(
    "📈 Fatigue Risk"
)

fig_risk = go.Figure()

fig_risk.add_trace(
    go.Scatter(
        x=df["elapsed_seconds"],
        y=df["fatigue_risk"],
        mode="lines",
        name="Fatigue Risk",
    )
)

fig_risk.add_hline(
    y=0.20,
    line_dash="dash",
    annotation_text="Low",
)

fig_risk.add_hline(
    y=0.35,
    line_dash="dash",
    annotation_text="Moderate",
)

fig_risk.add_hline(
    y=0.55,
    line_dash="dash",
    annotation_text="High",
)

fig_risk.add_hline(
    y=0.75,
    line_dash="dash",
    annotation_text="Critical",
)

fig_risk.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Fatigue Risk",
    yaxis_range=[0, 1],
    height=400,
)

st.plotly_chart(
    fig_risk,
    use_container_width=True,
)


# =============================================================
# EAR + PERCLOS
# =============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "👁️ Eye Aspect Ratio"
    )

    fig_ear = go.Figure()

    fig_ear.add_trace(
        go.Scatter(
            x=df["elapsed_seconds"],
            y=df["ear"],
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


with col2:

    st.subheader(
        "📊 PERCLOS"
    )

    fig_perclos = go.Figure()

    fig_perclos.add_trace(
        go.Scatter(
            x=df["elapsed_seconds"],
            y=df["perclos"],
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

fig_pose.add_trace(
    go.Scatter(
        x=df["elapsed_seconds"],
        y=df["pitch"],
        mode="lines",
        name="Pitch",
    )
)

fig_pose.add_trace(
    go.Scatter(
        x=df["elapsed_seconds"],
        y=df["yaw"],
        mode="lines",
        name="Yaw",
    )
)

fig_pose.add_trace(
    go.Scatter(
        x=df["elapsed_seconds"],
        y=df["roll"],
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

fig_reliability = go.Figure()

fig_reliability.add_trace(
    go.Scatter(
        x=df["elapsed_seconds"],
        y=df["reliability"],
        mode="lines",
        name="Reliability",
    )
)

fig_reliability.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Reliability",
    yaxis_range=[0, 1],
    height=350,
)

st.plotly_chart(
    fig_reliability,
    use_container_width=True,
)


# =============================================================
# SESSION INFORMATION
# =============================================================

st.divider()

st.subheader(
    "📋 Session Information"
)

duration = float(
    df["elapsed_seconds"].iloc[-1]
)

st.write(
    f"**Session file:** `{session_file}`"
)

st.write(
    f"**Session duration:** {duration:.1f} seconds"
)

st.write(
    f"**Samples:** {len(df)}"
)

st.write(
    f"**Safety alerts:** {alert_count}"
)


# =============================================================
# AUTO REFRESH
# =============================================================

st.caption(
    "Dashboard refreshes automatically every 2 seconds."
)

st.markdown(
    """
    <script>
        setTimeout(function(){
            window.location.reload();
        }, 2000);
    </script>
    """,
    unsafe_allow_html=True,
)