"""
=============================================================
ADAPTIVE-DMS
=============================================================

Adaptive Multimodal Driver State Monitoring
and Predictive Safety Intervention System

Dashboard Version:
    v1.7

Features:
    - Live camera frame
    - Session monitoring
    - Fatigue risk
    - EAR
    - MAR
    - PERCLOS
    - Reliability
    - Head pose
    - Gaze
    - Temporal state
    - Safety intervention
    - Alert count
    - Event timeline
    - Risk distribution
    - Risk duration
    - Fatigue trend
    - Driver metric analytics
    - Session report
    - CSV export
    - TXT export
    - Memory-safe session CSV loading

IMPORTANT:
    This dashboard is READ-ONLY.
    main.py controls the camera and writes session data.

=============================================================
"""

import os
import glob
import csv
import io
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =============================================================
# CONFIGURATION
# =============================================================

VERSION = "v1.7"

MAX_ROWS = 1500
MAX_EVENTS = 300

TAIL_BYTES = 512 * 1024
EVENT_TAIL_BYTES = 256 * 1024

LIVE_FRAME_PATH = os.path.join(
    "logs",
    "live_frame.jpg",
)

EVENT_FILE_PATH = os.path.join(
    "logs",
    "events.csv",
)

ANALYSIS_DIRECTORY = os.path.join(
    "analysis",
)

LIVE_FRAME_TIMEOUT = 5.0


# =============================================================
# PAGE CONFIGURATION
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
        min-height: 110px;
    }

    .alert-banner {
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 21px;
        font-weight: 700;
        margin: 10px 0 20px 0;
    }

    .alert-safe {
        background: rgba(0, 200, 83, 0.12);
        border: 1px solid rgba(0, 200, 83, 0.35);
    }

    .alert-warning {
        background: rgba(255, 152, 0, 0.14);
        border: 1px solid rgba(255, 152, 0, 0.40);
    }

    .alert-danger {
        background: rgba(244, 67, 54, 0.16);
        border: 1px solid rgba(244, 67, 54, 0.45);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# UTILITY FUNCTIONS
# =============================================================

def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def safe_string(value, default="UNKNOWN"):
    """
    Safely convert a value to string.
    """

    try:

        if pd.isna(value):
            return default

        text = str(value).strip()

        if not text:
            return default

        return text

    except Exception:

        return default


def get_file_age(file_path):
    """
    Return age of a file in seconds.
    """

    try:

        return (
            time.time()
            - os.path.getmtime(file_path)
        )

    except Exception:

        return float("inf")


def file_exists(file_path):
    """
    Safe file existence check.
    """

    try:

        return os.path.exists(file_path)

    except Exception:

        return False


def is_live_camera_active():
    """
    Check whether live frame is being updated.
    """

    if not file_exists(
        LIVE_FRAME_PATH
    ):

        return False

    return (
        get_file_age(
            LIVE_FRAME_PATH
        )
        <= LIVE_FRAME_TIMEOUT
    )


def find_latest_session():
    """
    Find latest ADAPTIVE-DMS session CSV.
    """

    try:

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

    except Exception:

        return None


# =============================================================
# MEMORY-SAFE CSV READER
# =============================================================

@st.cache_data(ttl=2)
def load_recent_session(
    file_path,
    modified_time,
):
    """
    Memory-safe session CSV loader.

    Important:
        - Reads header from beginning.
        - Reads only recent tail of CSV.
        - Does NOT load entire potentially huge CSV.
        - Handles partial first line.
    """

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
        # Read header from beginning
        # -----------------------------------------------------

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as file:

            header_line = file.readline()

        if not header_line.strip():

            return pd.DataFrame()

        try:

            headers = next(
                csv.reader(
                    [header_line]
                )
            )

        except Exception:

            return pd.DataFrame()

        if not headers:

            return pd.DataFrame()

        # -----------------------------------------------------
        # Read only recent part
        # -----------------------------------------------------

        bytes_to_read = min(
            file_size,
            TAIL_BYTES,
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

        if not lines:

            return pd.DataFrame(
                columns=headers
            )

        # -----------------------------------------------------
        # If we started in the middle of the file,
        # first line can be incomplete.
        # -----------------------------------------------------

        if bytes_to_read < file_size:

            lines = lines[1:]

        # -----------------------------------------------------
        # Keep only recent rows
        # -----------------------------------------------------

        if len(lines) > MAX_ROWS:

            lines = lines[
                -MAX_ROWS:
            ]

        # -----------------------------------------------------
        # Parse CSV rows
        # -----------------------------------------------------

        valid_rows = []

        reader = csv.reader(
            lines
        )

        for row in reader:

            if not row:

                continue

            if len(row) == len(headers):

                valid_rows.append(
                    row
                )

            elif len(row) > len(headers):

                valid_rows.append(
                    row[:len(headers)]
                )

            else:

                padded_row = (
                    row
                    + [""] * (
                        len(headers)
                        - len(row)
                    )
                )

                valid_rows.append(
                    padded_row
                )

        if not valid_rows:

            return pd.DataFrame(
                columns=headers
            )

        # -----------------------------------------------------
        # Create DataFrame
        # -----------------------------------------------------

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

            "horizontal_ratio",
            "vertical_ratio",

            "gaze_away_duration",

            "reliability",
            "overall_reliability",

            "fatigue_risk",

            "eye_closure_duration",

            "alert_count",

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
            "yawning",
            "microsleep",
            "prolonged_gaze_away",
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
                            "yes": 1,
                            "no": 0,
                        }
                    )
                    .fillna(0)
                )

        # -----------------------------------------------------
        # Sort by timestamp
        # -----------------------------------------------------

        df = df.sort_values(
            "timestamp"
        ).reset_index(
            drop=True
        )

        # -----------------------------------------------------
        # Elapsed time
        # -----------------------------------------------------

        if not df.empty:

            df["elapsed_seconds"] = (
                df["timestamp"]
                - df["timestamp"].iloc[0]
            ).dt.total_seconds()

        # -----------------------------------------------------
        # Final row protection
        # -----------------------------------------------------

        if len(df) > MAX_ROWS:

            df = df.iloc[
                -MAX_ROWS:
            ].reset_index(
                drop=True
            )

        return df

    except (
        OSError,
        PermissionError,
        MemoryError,
        UnicodeError,
        csv.Error,
    ):

        return pd.DataFrame()

    except Exception:

        return pd.DataFrame()


# =============================================================
# EVENT CSV LOADER
# =============================================================

@st.cache_data(ttl=2)
def load_events(
    file_path,
    modified_time,
):
    """
    Memory-safe event CSV loader.
    """

    columns = [
        "timestamp",
        "event_type",
        "severity",
        "message",
    ]

    try:

        if not os.path.exists(
            file_path
        ):

            return pd.DataFrame(
                columns=columns
            )

        file_size = os.path.getsize(
            file_path
        )

        if file_size <= 0:

            return pd.DataFrame(
                columns=columns
            )

        bytes_to_read = min(
            file_size,
            EVENT_TAIL_BYTES,
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

        if not lines:

            return pd.DataFrame(
                columns=columns
            )

        if bytes_to_read < file_size:

            lines = lines[1:]

        rows = []

        reader = csv.reader(
            lines
        )

        for row in reader:

            if not row:

                continue

            # Skip header
            if (
                len(row) >= 4
                and row[0].strip().lower()
                == "timestamp"
            ):

                continue

            if len(row) < 4:

                continue

            rows.append(
                [
                    row[0],
                    row[1],
                    row[2],
                    ",".join(
                        row[3:]
                    ),
                ]
            )

        if not rows:

            return pd.DataFrame(
                columns=columns
            )

        df = pd.DataFrame(
            rows,
            columns=columns,
        )

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

            return pd.DataFrame(
                columns=columns
            )

        df = df.sort_values(
            "timestamp",
            ascending=False,
        )

        df = df.head(
            MAX_EVENTS
        )

        return df.reset_index(
            drop=True
        )

    except Exception:

        return pd.DataFrame(
            columns=columns
        )


# =============================================================
# RISK LEVEL CALCULATOR
# =============================================================

def calculate_risk_level(
    risk,
):
    """
    Convert fatigue risk into risk level.

    Thresholds match main.py intervention configuration:
        0.20 -> LOW
        0.35 -> MODERATE
        0.55 -> HIGH
        0.75 -> CRITICAL
    """

    risk = safe_float(
        risk
    )

    if risk >= 0.75:

        return "CRITICAL"

    if risk >= 0.55:

        return "HIGH"

    if risk >= 0.35:

        return "MODERATE"

    if risk >= 0.20:

        return "LOW"

    return "NORMAL"


# =============================================================
# ENSURE DERIVED COLUMNS
# =============================================================

def prepare_dataframe(
    df,
):
    """
    Add useful derived columns when they
    are missing from the session CSV.
    """

    df = df.copy()

    if df.empty:

        return df

    # ---------------------------------------------------------
    # Risk level
    # ---------------------------------------------------------

    if "risk_level" not in df.columns:

        if "fatigue_risk" in df.columns:

            df["risk_level"] = (
                df["fatigue_risk"]
                .fillna(0)
                .apply(
                    calculate_risk_level
                )
            )

        else:

            df["risk_level"] = "NORMAL"

    else:

        df["risk_level"] = (
            df["risk_level"]
            .astype(str)
            .str.upper()
        )

    # ---------------------------------------------------------
    # Overall reliability
    # ---------------------------------------------------------

    if (
        "reliability" not in df.columns
        and "overall_reliability"
        in df.columns
    ):

        df["reliability"] = (
            df[
                "overall_reliability"
            ]
        )

    if (
        "reliability" not in df.columns
    ):

        df["reliability"] = 0.0

    # ---------------------------------------------------------
    # Fatigue risk
    # ---------------------------------------------------------

    if "fatigue_risk" not in df.columns:

        df["fatigue_risk"] = 0.0

    # ---------------------------------------------------------
    # Standard numeric columns
    # ---------------------------------------------------------

    defaults = {

        "ear": 0.0,

        "mar": 0.0,

        "perclos": 0.0,

        "pitch": 0.0,

        "yaw": 0.0,

        "roll": 0.0,

        "eye_closure_duration": 0.0,

        "microsleep_duration": 0.0,

        "gaze_away_duration": 0.0,

        "blink_count": 0,

    }

    for column, default in defaults.items():

        if column not in df.columns:

            df[column] = default

    # ---------------------------------------------------------
    # Event states
    # ---------------------------------------------------------

    event_defaults = {

        "eyes_closed": 0,

        "yawning": 0,

        "microsleep": 0,

        "prolonged_gaze_away": 0,

    }

    for column, default in event_defaults.items():

        if column not in df.columns:

            df[column] = default

    # ---------------------------------------------------------
    # Temporal state
    # ---------------------------------------------------------

    if "temporal_state" not in df.columns:

        df["temporal_state"] = "UNKNOWN"

    # ---------------------------------------------------------
    # Intervention
    # ---------------------------------------------------------

    if "intervention_level" not in df.columns:

        df["intervention_level"] = "NORMAL"

    if "intervention_action" not in df.columns:

        df["intervention_action"] = "NO_ACTION"

    return df


# =============================================================
# SESSION ANALYTICS
# =============================================================

def calculate_session_analytics(
    df,
    events_df,
):
    """
    Calculate all v1.7 analytics locally.

    No external analytics module required.
    """

    result = {}

    if df.empty:

        return result

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------

    risk_series = pd.to_numeric(
        df["fatigue_risk"],
        errors="coerce",
    ).fillna(0)

    result[
        "sample_count"
    ] = len(df)

    result[
        "average_fatigue_risk"
    ] = float(
        risk_series.mean()
    )

    result[
        "maximum_fatigue_risk"
    ] = float(
        risk_series.max()
    )

    result[
        "minimum_fatigue_risk"
    ] = float(
        risk_series.min()
    )

    # ---------------------------------------------------------
    # Session duration
    # ---------------------------------------------------------

    if (
        "elapsed_seconds" in df.columns
        and not df.empty
    ):

        result[
            "session_duration_seconds"
        ] = float(
            df[
                "elapsed_seconds"
            ].max()
        )

    else:

        result[
            "session_duration_seconds"
        ] = 0.0

    # ---------------------------------------------------------
    # Fatigue trend
    # ---------------------------------------------------------

    if len(risk_series) >= 2:

        window = min(
            100,
            len(risk_series) // 2,
        )

        if window < 1:

            window = 1

        first_average = float(
            risk_series
            .iloc[
                :window
            ]
            .mean()
        )

        last_average = float(
            risk_series
            .iloc[
                -window:
            ]
            .mean()
        )

        result[
            "fatigue_trend"
        ] = (
            last_average
            - first_average
        )

    else:

        result[
            "fatigue_trend"
        ] = 0.0

    # ---------------------------------------------------------
    # Average metrics
    # ---------------------------------------------------------

    result[
        "average_ear"
    ] = float(
        pd.to_numeric(
            df["ear"],
            errors="coerce",
        )
        .fillna(0)
        .mean()
    )

    result[
        "average_mar"
    ] = float(
        pd.to_numeric(
            df["mar"],
            errors="coerce",
        )
        .fillna(0)
        .mean()
    )

    result[
        "average_perclos"
    ] = float(
        pd.to_numeric(
            df["perclos"],
            errors="coerce",
        )
        .fillna(0)
        .mean()
    )

    result[
        "average_reliability"
    ] = float(
        pd.to_numeric(
            df["reliability"],
            errors="coerce",
        )
        .fillna(0)
        .mean()
    )

    result[
        "maximum_eye_closure"
    ] = float(
        pd.to_numeric(
            df[
                "eye_closure_duration"
            ],
            errors="coerce",
        )
        .fillna(0)
        .max()
    )

    result[
        "maximum_microsleep"
    ] = float(
        pd.to_numeric(
            df[
                "microsleep_duration"
            ],
            errors="coerce",
        )
        .fillna(0)
        .max()
    )

    # ---------------------------------------------------------
    # Risk distribution
    # ---------------------------------------------------------

    levels = [

        "NORMAL",

        "LOW",

        "MODERATE",

        "HIGH",

        "CRITICAL",

    ]

    risk_distribution = {}

    for level in levels:

        risk_distribution[
            level
        ] = int(
            (
                df[
                    "risk_level"
                ]
                == level
            )
            .sum()
        )

    result[
        "risk_distribution"
    ] = risk_distribution

    # ---------------------------------------------------------
    # Risk duration
    # ---------------------------------------------------------

    risk_duration = {

        "NORMAL": 0.0,

        "LOW": 0.0,

        "MODERATE": 0.0,

        "HIGH": 0.0,

        "CRITICAL": 0.0,

    }

    if len(df) > 1:

        for index in range(
            len(df) - 1
        ):

            current_level = safe_string(
                df.iloc[
                    index
                ][
                    "risk_level"
                ],
                "NORMAL",
            ).upper()

            delta = (
                safe_float(
                    df.iloc[
                        index + 1
                    ][
                        "elapsed_seconds"
                    ]
                )
                - safe_float(
                    df.iloc[
                        index
                    ][
                        "elapsed_seconds"
                    ]
                )
            )

            if delta < 0:

                delta = 0

            if delta > 5:

                delta = 5

            if (
                current_level
                in risk_duration
            ):

                risk_duration[
                    current_level
                ] += delta

    result[
        "risk_duration"
    ] = risk_duration

    # ---------------------------------------------------------
    # Event analytics
    # ---------------------------------------------------------

    event_summary = {

        "total_events": 0,

        "eye_closure": 0,

        "microsleep": 0,

        "yawning": 0,

        "gaze_away": 0,

        "risk_transition": 0,

        "intervention": 0,

    }

    if (
        events_df is not None
        and not events_df.empty
    ):

        event_summary[
            "total_events"
        ] = len(events_df)

        for event_type in events_df[
            "event_type"
        ].astype(str):

            event_type = (
                event_type
                .upper()
                .strip()
            )

            if (
                "EYE"
                in event_type
            ):

                event_summary[
                    "eye_closure"
                ] += 1

            elif (
                "MICROSLEEP"
                in event_type
            ):

                event_summary[
                    "microsleep"
                ] += 1

            elif (
                "YAWN"
                in event_type
            ):

                event_summary[
                    "yawning"
                ] += 1

            elif (
                "GAZE"
                in event_type
            ):

                event_summary[
                    "gaze_away"
                ] += 1

            elif (
                "RISK"
                in event_type
            ):

                event_summary[
                    "risk_transition"
                ] += 1

            elif (
                "INTERVENTION"
                in event_type
            ):

                event_summary[
                    "intervention"
                ] += 1

    result[
        "event_summary"
    ] = event_summary

    # ---------------------------------------------------------
    # Session alert count
    # ---------------------------------------------------------

    result[
        "session_alert_count"
    ] = event_summary[
        "intervention"
    ]

    return result


# =============================================================
# REPORT GENERATOR
# =============================================================

def generate_report(
    analytics,
    current_risk,
    current_level,
    current_intervention,
):
    """
    Generate plain-text session report.
    """

    duration = analytics.get(
        "session_duration_seconds",
        0,
    )

    minutes = int(
        duration // 60
    )

    seconds = int(
        duration % 60
    )

    risk_distribution = analytics.get(
        "risk_distribution",
        {},
    )

    events = analytics.get(
        "event_summary",
        {},
    )

    lines = [

        "=" * 70,

        "ADAPTIVE-DMS SESSION REPORT",

        "=" * 70,

        "",

        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",

        "",

        "SESSION SUMMARY",

        "-" * 70,

        f"Duration: {minutes:02d}:{seconds:02d}",

        f"Samples: {analytics.get('sample_count', 0)}",

        f"Average Fatigue Risk: "
        f"{analytics.get('average_fatigue_risk', 0):.3f}",

        f"Maximum Fatigue Risk: "
        f"{analytics.get('maximum_fatigue_risk', 0):.3f}",

        f"Minimum Fatigue Risk: "
        f"{analytics.get('minimum_fatigue_risk', 0):.3f}",

        f"Fatigue Trend: "
        f"{analytics.get('fatigue_trend', 0):+.3f}",

        "",

        "CURRENT DRIVER STATE",

        "-" * 70,

        f"Current Fatigue Risk: {current_risk:.3f}",

        f"Current Risk Level: {current_level}",

        f"Intervention Level: {current_intervention}",

        "",

        "DRIVER METRICS",

        "-" * 70,

        f"Average EAR: "
        f"{analytics.get('average_ear', 0):.3f}",

        f"Average MAR: "
        f"{analytics.get('average_mar', 0):.3f}",

        f"Average PERCLOS: "
        f"{analytics.get('average_perclos', 0):.3f}",

        f"Average Reliability: "
        f"{analytics.get('average_reliability', 0):.3f}",

        f"Maximum Eye Closure: "
        f"{analytics.get('maximum_eye_closure', 0):.2f}s",

        f"Maximum Microsleep: "
        f"{analytics.get('maximum_microsleep', 0):.2f}s",

        "",

        "RISK DISTRIBUTION",

        "-" * 70,

        f"NORMAL: "
        f"{risk_distribution.get('NORMAL', 0)} samples",

        f"LOW: "
        f"{risk_distribution.get('LOW', 0)} samples",

        f"MODERATE: "
        f"{risk_distribution.get('MODERATE', 0)} samples",

        f"HIGH: "
        f"{risk_distribution.get('HIGH', 0)} samples",

        f"CRITICAL: "
        f"{risk_distribution.get('CRITICAL', 0)} samples",

        "",

        "EVENT SUMMARY",

        "-" * 70,

        f"Total Events: "
        f"{events.get('total_events', 0)}",

        f"Eye Closure: "
        f"{events.get('eye_closure', 0)}",

        f"Microsleep: "
        f"{events.get('microsleep', 0)}",

        f"Yawning: "
        f"{events.get('yawning', 0)}",

        f"Gaze Away: "
        f"{events.get('gaze_away', 0)}",

        f"Risk Transitions: "
        f"{events.get('risk_transition', 0)}",

        f"Interventions: "
        f"{events.get('intervention', 0)}",

        "",

        "=" * 70,

        "ADAPTIVE-DMS v1.7",

        "=" * 70,

    ]

    return "\n".join(
        lines
    )


# =============================================================
# FIND LATEST SESSION
# =============================================================

session_file = find_latest_session()


# =============================================================
# SIDEBAR
# =============================================================

with st.sidebar:

    st.header(
        "⚙️ Dashboard Controls"
    )

    # IMPORTANT:
    # Unique key prevents StreamlitDuplicateElementId.
    refresh_clicked = st.button(
        "🔄 Refresh Now",
        width="stretch",
        key="refresh_dashboard_v17",
    )

    if refresh_clicked:

        st.cache_data.clear()

        st.rerun()

    st.divider()

    st.subheader(
        "System"
    )

    st.write(
        f"Version: **{VERSION}**"
    )

    st.write(
        "Camera owner: **main.py**"
    )

    st.write(
        "Dashboard: **Read-only**"
    )

    st.divider()

    st.caption(
        "main.py controls camera processing."
    )

    st.caption(
        "This dashboard reads session data."
    )


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
# NO SESSION
# =============================================================

if session_file is None:

    st.warning(
        "No session CSV found."
    )

    st.info(
        "Start main.py first."
    )

    st.stop()


# =============================================================
# SESSION MODIFICATION TIME
# =============================================================

try:

    session_modified_time = os.path.getmtime(
        session_file
    )

except Exception:

    session_modified_time = time.time()


# =============================================================
# LOAD SESSION
# =============================================================

df = load_recent_session(
    session_file,
    session_modified_time,
)


# =============================================================
# EMPTY SESSION
# =============================================================

if df.empty:

    st.warning(
        "Waiting for readable session data..."
    )

    st.info(
        "Keep main.py running for a few seconds."
    )

    time.sleep(2)

    st.rerun()


# =============================================================
# PREPARE DATA
# =============================================================

df = prepare_dataframe(
    df
)


# =============================================================
# EVENT FILE
# =============================================================

try:

    event_modified_time = os.path.getmtime(
        EVENT_FILE_PATH
    )

except Exception:

    event_modified_time = time.time()


events_df = load_events(
    EVENT_FILE_PATH,
    event_modified_time,
)


# =============================================================
# ANALYTICS
# =============================================================

analytics = calculate_session_analytics(
    df,
    events_df,
)


# =============================================================
# LATEST SAMPLE
# =============================================================

latest = df.iloc[-1]


# =============================================================
# CURRENT VALUES
# =============================================================

fatigue_risk = safe_float(
    latest.get(
        "fatigue_risk",
        0,
    )
)

ear = safe_float(
    latest.get(
        "ear",
        0,
    )
)

mar = safe_float(
    latest.get(
        "mar",
        0,
    )
)

perclos = safe_float(
    latest.get(
        "perclos",
        0,
    )
)

reliability = safe_float(
    latest.get(
        "reliability",
        0,
    )
)

pitch = safe_float(
    latest.get(
        "pitch",
        0,
    )
)

yaw = safe_float(
    latest.get(
        "yaw",
        0,
    )
)

roll = safe_float(
    latest.get(
        "roll",
        0,
    )
)

eye_closure_duration = safe_float(
    latest.get(
        "eye_closure_duration",
        0,
    )
)

gaze_direction = safe_string(
    latest.get(
        "gaze_direction",
        "UNKNOWN",
    )
)

temporal_state = safe_string(
    latest.get(
        "temporal_state",
        "UNKNOWN",
    )
)

intervention_level = safe_string(
    latest.get(
        "intervention_level",
        "NORMAL",
    )
).upper()

intervention_action = safe_string(
    latest.get(
        "intervention_action",
        "NO_ACTION",
    )
)

risk_level = safe_string(
    latest.get(
        "risk_level",
        calculate_risk_level(
            fatigue_risk
        ),
    )
).upper()


# =============================================================
# SESSION DURATION
# =============================================================

session_duration = analytics.get(
    "session_duration_seconds",
    0,
)

session_minutes = int(
    session_duration // 60
)

session_seconds = int(
    session_duration % 60
)


# =============================================================
# SYSTEM STATUS
# =============================================================

camera_active = (
    is_live_camera_active()
)


# =============================================================
# SYSTEM STATUS SECTION
# =============================================================

st.subheader(
    "📡 System Status"
)

c1, c2, c3, c4 = st.columns(
    4
)


with c1:

    if camera_active:

        st.success(
            "🟢 SYSTEM RUNNING"
        )

    else:

        st.warning(
            "🟡 SESSION AVAILABLE"
        )


with c2:

    if camera_active:

        st.success(
            "🎥 CAMERA ACTIVE"
        )

    else:

        st.error(
            "🔴 CAMERA OFFLINE"
        )


with c3:

    st.success(
        "📝 SESSION DATA"
    )


with c4:

    st.metric(
        "Events",
        analytics[
            "event_summary"
        ][
            "total_events"
        ],
    )


# =============================================================
# RISK BANNER
# =============================================================

if risk_level == "CRITICAL":

    banner_class = (
        "alert-danger"
    )

    banner_text = (
        "🚨 CRITICAL RISK — "
        "IMMEDIATE DRIVER ATTENTION REQUIRED"
    )

elif risk_level == "HIGH":

    banner_class = (
        "alert-danger"
    )

    banner_text = (
        "🚨 HIGH RISK — "
        "DRIVER ATTENTION REQUIRED"
    )

elif risk_level == "MODERATE":

    banner_class = (
        "alert-warning"
    )

    banner_text = (
        "⚠️ MODERATE RISK — "
        "MONITOR DRIVER STATE"
    )

elif risk_level == "LOW":

    banner_class = (
        "alert-warning"
    )

    banner_text = (
        "🟡 LOW RISK — "
        "CONTINUE MONITORING"
    )

else:

    banner_class = (
        "alert-safe"
    )

    banner_text = (
        "🟢 DRIVER STATE STABLE"
    )


st.markdown(
    f'<div class="alert-banner '
    f'{banner_class}">'
    f'{banner_text}'
    f'</div>',
    unsafe_allow_html=True,
)


# =============================================================
# SESSION OVERVIEW
# =============================================================

st.subheader(
    "📊 Session Overview"
)

c1, c2, c3, c4 = st.columns(
    4
)


with c1:

    st.metric(
        "⏱️ Duration",
        (
            f"{session_minutes:02d}:"
            f"{session_seconds:02d}"
        ),
    )


with c2:

    st.metric(
        "🎯 Avg Risk",
        (
            f"{analytics['average_fatigue_risk']:.3f}"
        ),
    )


with c3:

    st.metric(
        "🚨 Max Risk",
        (
            f"{analytics['maximum_fatigue_risk']:.3f}"
        ),
    )


with c4:

    st.metric(
        "📋 Samples",
        analytics[
            "sample_count"
        ],
    )


# =============================================================
# LIVE CAMERA
# =============================================================

st.divider()

st.subheader(
    "🎥 Live Camera Monitoring"
)

if file_exists(
    LIVE_FRAME_PATH
):

    try:

        with open(
            LIVE_FRAME_PATH,
            "rb",
        ) as file:

            image_data = file.read()

        if image_data:

            st.image(
                image_data,
                caption="Processed live camera feed",
                width="stretch",
            )

            frame_age = get_file_age(
                LIVE_FRAME_PATH
            )

            if frame_age <= LIVE_FRAME_TIMEOUT:

                st.success(
                    f"🟢 Live feed active "
                    f"({frame_age:.1f}s ago)"
                )

            else:

                st.warning(
                    f"🟡 Frame is stale "
                    f"({frame_age:.1f}s old)"
                )

    except Exception as error:

        st.error(
            f"Unable to display camera frame: {error}"
        )

else:

    st.info(
        "Waiting for main.py to generate "
        "the live camera frame..."
    )


# =============================================================
# CURRENT DRIVER STATE
# =============================================================

st.divider()

st.subheader(
    "🧠 Current Driver State"
)

c1, c2, c3, c4, c5 = st.columns(
    5
)


with c1:

    st.metric(
        "👁️ EAR",
        f"{ear:.3f}",
    )


with c2:

    st.metric(
        "😮 MAR",
        f"{mar:.3f}",
    )


with c3:

    st.metric(
        "📊 PERCLOS",
        f"{perclos:.3f}",
    )


with c4:

    st.metric(
        "📡 Reliability",
        f"{reliability:.3f}",
    )


with c5:

    st.metric(
        "👀 Gaze",
        gaze_direction,
    )


# =============================================================
# CURRENT FATIGUE
# =============================================================

c1, c2, c3 = st.columns(
    3
)


with c1:

    st.markdown(
        f"""
        <div class="status-card">
        <b>Fatigue Risk</b>
        <br><br>
        {fatigue_risk:.3f}
        </div>
        """,
        unsafe_allow_html=True,
    )


with c2:

    st.markdown(
        f"""
        <div class="status-card">
        <b>Risk Level</b>
        <br><br>
        {risk_level}
        </div>
        """,
        unsafe_allow_html=True,
    )


with c3:

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


# =============================================================
# INTERVENTION
# =============================================================

st.subheader(
    "🚨 Safety Intervention"
)

c1, c2, c3 = st.columns(
    3
)


with c1:

    st.metric(
        "Level",
        intervention_level,
    )


with c2:

    st.metric(
        "Action",
        intervention_action,
    )


with c3:

    st.metric(
        "Alerts",
        analytics[
            "session_alert_count"
        ],
    )


# =============================================================
# EYE CLOSURE
# =============================================================

st.subheader(
    "👁️ Eye Closure Monitoring"
)

c1, c2, c3 = st.columns(
    3
)


with c1:

    st.metric(
        "Current Closure",
        f"{eye_closure_duration:.2f}s",
    )


with c2:

    st.metric(
        "Maximum Closure",
        (
            f"{analytics['maximum_eye_closure']:.2f}s"
        ),
    )


with c3:

    if eye_closure_duration >= 1.5:

        st.error(
            "🚨 1.5 SECOND THRESHOLD REACHED"
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
# EVENT STATISTICS
# =============================================================

st.divider()

st.subheader(
    "📊 Event Statistics"
)

events = analytics[
    "event_summary"
]

c1, c2, c3, c4, c5, c6 = st.columns(
    6
)


with c1:

    st.metric(
        "👁️ Eye Closure",
        events[
            "eye_closure"
        ],
    )


with c2:

    st.metric(
        "😴 Microsleep",
        events[
            "microsleep"
        ],
    )


with c3:

    st.metric(
        "😮 Yawning",
        events[
            "yawning"
        ],
    )


with c4:

    st.metric(
        "👀 Gaze Away",
        events[
            "gaze_away"
        ],
    )


with c5:

    st.metric(
        "⚠️ Risk Changes",
        events[
            "risk_transition"
        ],
    )


with c6:

    st.metric(
        "🔔 Interventions",
        events[
            "intervention"
        ],
    )


# =============================================================
# EVENT TIMELINE
# =============================================================

st.divider()

st.subheader(
    "🚨 Event & Alert Timeline"
)

if events_df.empty:

    st.info(
        "No events recorded yet."
    )

else:

    timeline = events_df.copy()

    event_y_map = {

        "EYE_CLOSURE": 1,

        "MICROSLEEP": 2,

        "YAWNING": 3,

        "GAZE_AWAY": 4,

        "RISK_TRANSITION": 5,

        "INTERVENTION": 6,

    }

    timeline[
        "event_y"
    ] = (
        timeline[
            "event_type"
        ]
        .astype(str)
        .str.upper()
        .map(
            event_y_map
        )
        .fillna(7)
    )

    fig_timeline = go.Figure()

    fig_timeline.add_trace(
        go.Scatter(
            x=timeline[
                "timestamp"
            ],
            y=timeline[
                "event_y"
            ],
            mode="markers",
            text=timeline[
                "event_type"
            ],
            customdata=timeline[
                [
                    "severity",
                    "message",
                ]
            ],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Time: %{x}<br>"
                "Severity: %{customdata[0]}<br>"
                "%{customdata[1]}"
                "<extra></extra>"
            ),
            marker=dict(
                size=13
            ),
        )
    )

    fig_timeline.update_yaxes(
        tickmode="array",
        tickvals=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
        ticktext=[
            "Eye Closure",
            "Microsleep",
            "Yawning",
            "Gaze Away",
            "Risk Transition",
            "Intervention",
            "Other",
        ],
    )

    fig_timeline.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Event",
        height=420,
        showlegend=False,
    )

    st.plotly_chart(
        fig_timeline,
        width="stretch",
    )

    st.subheader(
        "📋 Recent Events"
    )

    display_events = events_df[
        [
            "timestamp",
            "event_type",
            "severity",
            "message",
        ]
    ].copy()

    display_events[
        "timestamp"
    ] = display_events[
        "timestamp"
    ].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.dataframe(
        display_events,
        width="stretch",
        hide_index=True,
    )


# =============================================================
# ADVANCED ANALYTICS
# =============================================================

st.divider()

st.header(
    "📈 Advanced Session Analytics"
)

st.caption(
    "v1.7 analytics calculated directly from "
    "the current session."
)


# =============================================================
# FATIGUE ANALYTICS
# =============================================================

st.subheader(
    "🎯 Fatigue Analytics"
)

c1, c2, c3, c4 = st.columns(
    4
)


with c1:

    st.metric(
        "Average Risk",
        (
            f"{analytics['average_fatigue_risk']:.3f}"
        ),
    )


with c2:

    st.metric(
        "Maximum Risk",
        (
            f"{analytics['maximum_fatigue_risk']:.3f}"
        ),
    )


with c3:

    st.metric(
        "Minimum Risk",
        (
            f"{analytics['minimum_fatigue_risk']:.3f}"
        ),
    )


with c4:

    trend = analytics[
        "fatigue_trend"
    ]

    st.metric(
        "Fatigue Trend",
        f"{trend:+.3f}",
    )


# =============================================================
# TREND MESSAGE
# =============================================================

if trend > 0.02:

    st.warning(
        "📈 Fatigue trend is increasing."
    )

elif trend < -0.02:

    st.success(
        "📉 Fatigue trend is decreasing."
    )

else:

    st.info(
        "➡️ Fatigue trend is relatively stable."
    )


# =============================================================
# FATIGUE RISK GRAPH
# =============================================================

st.subheader(
    "📈 Fatigue Risk Trend"
)

fig_fatigue = go.Figure()

fig_fatigue.add_trace(
    go.Scatter(
        x=df[
            "elapsed_seconds"
        ],
        y=df[
            "fatigue_risk"
        ],
        mode="lines",
        name="Fatigue Risk",
    )
)

fig_fatigue.add_hline(
    y=0.20,
    line_dash="dash",
    annotation_text="LOW",
)

fig_fatigue.add_hline(
    y=0.35,
    line_dash="dash",
    annotation_text="MODERATE",
)

fig_fatigue.add_hline(
    y=0.55,
    line_dash="dash",
    annotation_text="HIGH",
)

fig_fatigue.add_hline(
    y=0.75,
    line_dash="dash",
    annotation_text="CRITICAL",
)

fig_fatigue.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Fatigue Risk",
    yaxis_range=[
        0,
        1,
    ],
    height=420,
)

st.plotly_chart(
    fig_fatigue,
    width="stretch",
)


# =============================================================
# RISK DISTRIBUTION
# =============================================================

st.subheader(
    "⚠️ Risk Distribution"
)

risk_distribution = analytics[
    "risk_distribution"
]

risk_df = pd.DataFrame(
    {
        "Risk Level": [
            "NORMAL",
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        ],
        "Samples": [
            risk_distribution[
                "NORMAL"
            ],
            risk_distribution[
                "LOW"
            ],
            risk_distribution[
                "MODERATE"
            ],
            risk_distribution[
                "HIGH"
            ],
            risk_distribution[
                "CRITICAL"
            ],
        ],
    }
)

fig_risk = go.Figure()

fig_risk.add_trace(
    go.Bar(
        x=risk_df[
            "Risk Level"
        ],
        y=risk_df[
            "Samples"
        ],
        text=risk_df[
            "Samples"
        ],
        textposition="auto",
    )
)

fig_risk.update_layout(
    xaxis_title="Risk Level",
    yaxis_title="Samples",
    height=400,
)

st.plotly_chart(
    fig_risk,
    width="stretch",
)


# =============================================================
# RISK DURATION
# =============================================================

st.subheader(
    "⏱️ Time Spent at Each Risk Level"
)

risk_duration = analytics[
    "risk_duration"
]

duration_df = pd.DataFrame(
    {
        "Risk Level": [
            "NORMAL",
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        ],
        "Duration (seconds)": [
            risk_duration[
                "NORMAL"
            ],
            risk_duration[
                "LOW"
            ],
            risk_duration[
                "MODERATE"
            ],
            risk_duration[
                "HIGH"
            ],
            risk_duration[
                "CRITICAL"
            ],
        ],
    }
)

fig_duration = go.Figure()

fig_duration.add_trace(
    go.Bar(
        x=duration_df[
            "Risk Level"
        ],
        y=duration_df[
            "Duration (seconds)"
        ],
        text=[
            f"{value:.1f}s"
            for value in duration_df[
                "Duration (seconds)"
            ]
        ],
        textposition="auto",
    )
)

fig_duration.update_layout(
    xaxis_title="Risk Level",
    yaxis_title="Duration (seconds)",
    height=400,
)

st.plotly_chart(
    fig_duration,
    width="stretch",
)


# =============================================================
# DRIVER METRIC ANALYTICS
# =============================================================

st.subheader(
    "🧠 Driver Metric Analytics"
)

c1, c2, c3, c4, c5 = st.columns(
    5
)


with c1:

    st.metric(
        "Average EAR",
        (
            f"{analytics['average_ear']:.3f}"
        ),
    )


with c2:

    st.metric(
        "Average MAR",
        (
            f"{analytics['average_mar']:.3f}"
        ),
    )


with c3:

    st.metric(
        "Average PERCLOS",
        (
            f"{analytics['average_perclos']:.3f}"
        ),
    )


with c4:

    st.metric(
        "Max Eye Closure",
        (
            f"{analytics['maximum_eye_closure']:.2f}s"
        ),
    )


with c5:

    st.metric(
        "Max Microsleep",
        (
            f"{analytics['maximum_microsleep']:.2f}s"
        ),
    )


# =============================================================
# EVENT FREQUENCY
# =============================================================

st.subheader(
    "🚨 Event Frequency"
)

event_chart_df = pd.DataFrame(
    {
        "Event": [
            "Eye Closure",
            "Microsleep",
            "Yawning",
            "Gaze Away",
            "Risk Transition",
            "Intervention",
        ],
        "Count": [
            events[
                "eye_closure"
            ],
            events[
                "microsleep"
            ],
            events[
                "yawning"
            ],
            events[
                "gaze_away"
            ],
            events[
                "risk_transition"
            ],
            events[
                "intervention"
            ],
        ],
    }
)

fig_event_frequency = go.Figure()

fig_event_frequency.add_trace(
    go.Bar(
        x=event_chart_df[
            "Event"
        ],
        y=event_chart_df[
            "Count"
        ],
        text=event_chart_df[
            "Count"
        ],
        textposition="auto",
    )
)

fig_event_frequency.update_layout(
    xaxis_title="Event Type",
    yaxis_title="Count",
    height=420,
)

st.plotly_chart(
    fig_event_frequency,
    width="stretch",
)


# =============================================================
# HEAD POSE
# =============================================================

st.divider()

st.subheader(
    "🧭 Head Pose"
)

fig_pose = go.Figure()

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
    width="stretch",
)


# =============================================================
# SIGNAL RELIABILITY
# =============================================================

st.subheader(
    "📡 Signal Reliability"
)

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
    width="stretch",
)


# =============================================================
# SESSION REPORT
# =============================================================

st.divider()

st.header(
    "📄 Session Report"
)

report_text = generate_report(
    analytics=analytics,
    current_risk=fatigue_risk,
    current_level=risk_level,
    current_intervention=intervention_level,
)


# =============================================================
# REPORT PREVIEW
# =============================================================

with st.expander(
    "👁️ Preview Session Report"
):

    st.code(
        report_text,
        language="text",
    )


# =============================================================
# ANALYSIS DIRECTORY
# =============================================================

os.makedirs(
    ANALYSIS_DIRECTORY,
    exist_ok=True,
)


# =============================================================
# CSV REPORT
# =============================================================

report_rows = [

    {
        "metric": "session_duration_seconds",
        "value": analytics[
            "session_duration_seconds"
        ],
    },

    {
        "metric": "sample_count",
        "value": analytics[
            "sample_count"
        ],
    },

    {
        "metric": "average_fatigue_risk",
        "value": analytics[
            "average_fatigue_risk"
        ],
    },

    {
        "metric": "maximum_fatigue_risk",
        "value": analytics[
            "maximum_fatigue_risk"
        ],
    },

    {
        "metric": "minimum_fatigue_risk",
        "value": analytics[
            "minimum_fatigue_risk"
        ],
    },

    {
        "metric": "fatigue_trend",
        "value": analytics[
            "fatigue_trend"
        ],
    },

    {
        "metric": "average_ear",
        "value": analytics[
            "average_ear"
        ],
    },

    {
        "metric": "average_mar",
        "value": analytics[
            "average_mar"
        ],
    },

    {
        "metric": "average_perclos",
        "value": analytics[
            "average_perclos"
        ],
    },

    {
        "metric": "average_reliability",
        "value": analytics[
            "average_reliability"
        ],
    },

    {
        "metric": "maximum_eye_closure",
        "value": analytics[
            "maximum_eye_closure"
        ],
    },

    {
        "metric": "maximum_microsleep",
        "value": analytics[
            "maximum_microsleep"
        ],
    },

]

csv_report = pd.DataFrame(
    report_rows
).to_csv(
    index=False
)


# =============================================================
# DOWNLOAD CSV
# =============================================================

st.download_button(
    label="⬇️ Download Analytics CSV",
    data=csv_report,
    file_name="session_analytics.csv",
    mime="text/csv",
    width="stretch",
    key="download_analytics_csv_v17",
)


# =============================================================
# DOWNLOAD TXT
# =============================================================

st.download_button(
    label="⬇️ Download Session Report",
    data=report_text,
    file_name="session_report.txt",
    mime="text/plain",
    width="stretch",
    key="download_session_report_v17",
)


# =============================================================
# SAVE REPORTS
# =============================================================

if st.button(
    "💾 Save Reports to analysis/",
    width="stretch",
    key="save_reports_v17",
):

    try:

        csv_path = os.path.join(
            ANALYSIS_DIRECTORY,
            "session_analytics.csv",
        )

        txt_path = os.path.join(
            ANALYSIS_DIRECTORY,
            "session_report.txt",
        )

        with open(
            csv_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            file.write(
                csv_report
            )

        with open(
            txt_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                report_text
            )

        st.success(
            "✅ Reports saved successfully."
        )

        st.write(
            f"CSV: `{csv_path}`"
        )

        st.write(
            f"TXT: `{txt_path}`"
        )

    except Exception as error:

        st.error(
            f"Unable to save reports: {error}"
        )


# =============================================================
# SESSION INFORMATION
# =============================================================

st.divider()

st.subheader(
    "📋 Session Information"
)

c1, c2, c3 = st.columns(
    3
)


with c1:

    st.write(
        "**Session file**"
    )

    st.code(
        os.path.basename(
            session_file
        )
    )


with c2:

    st.write(
        "**Session duration**"
    )

    st.write(
        (
            f"{session_minutes:02d}:"
            f"{session_seconds:02d}"
        )
    )


with c3:

    st.write(
        "**Samples displayed**"
    )

    st.write(
        len(df)
    )


# =============================================================
# DATA HEALTH
# =============================================================

st.subheader(
    "🔍 Data Health"
)

csv_age = get_file_age(
    session_file
)

frame_age = get_file_age(
    LIVE_FRAME_PATH
)

event_age = get_file_age(
    EVENT_FILE_PATH
)

c1, c2, c3 = st.columns(
    3
)


with c1:

    if csv_age <= 5:

        st.success(
            f"🟢 Session data fresh "
            f"({csv_age:.1f}s)"
        )

    else:

        st.warning(
            f"🟡 Session data stale "
            f"({csv_age:.1f}s)"
        )


with c2:

    if frame_age <= LIVE_FRAME_TIMEOUT:

        st.success(
            f"🟢 Camera frame fresh "
            f"({frame_age:.1f}s)"
        )

    else:

        st.warning(
            f"🟡 Camera frame stale "
            f"({frame_age:.1f}s)"
        )


with c3:

    if (
        file_exists(
            EVENT_FILE_PATH
        )
        and event_age <= 5
    ):

        st.success(
            f"🟢 Event log active "
            f"({event_age:.1f}s)"
        )

    elif file_exists(
        EVENT_FILE_PATH
    ):

        st.warning(
            f"🟡 Event log stale "
            f"({event_age:.1f}s)"
        )

    else:

        st.info(
            "ℹ️ No event log yet"
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
# CURRENT SESSION METRICS
# =============================================================

st.subheader(
    "📌 Current Session Metrics"
)

c1, c2, c3, c4 = st.columns(
    4
)


with c1:

    st.metric(
        "Current Risk",
        f"{fatigue_risk:.3f}",
    )


with c2:

    st.metric(
        "Risk Level",
        risk_level,
    )


with c3:

    st.metric(
        "Reliability",
        f"{reliability:.3f}",
    )


with c4:

    st.metric(
        "Gaze",
        gaze_direction,
    )


# =============================================================
# FOOTER
# =============================================================

st.divider()

st.caption(
    "ADAPTIVE-DMS v1.7 | "
    "Driver Monitoring + Event Timeline + Advanced Analytics"
)

st.caption(
    "Camera and processing are controlled by main.py."
)

st.caption(
    "Dashboard is read-only."
)


# =============================================================
# AUTO REFRESH
# =============================================================

time.sleep(2)

st.rerun()