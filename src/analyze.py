"""
ADAPTIVE-DMS
Data Analysis & Visualization

v1.0

Reads a session CSV from the logs/ directory
and generates analysis graphs.
"""

import os
import glob

import pandas as pd
import matplotlib.pyplot as plt


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

        raise FileNotFoundError(
            "No session CSV files found in logs/"
        )

    return max(
        files,
        key=os.path.getmtime,
    )


# =============================================================
# LOAD DATA
# =============================================================

def load_data(file_path):

    print("=" * 60)

    print("ADAPTIVE-DMS DATA ANALYSIS")

    print("=" * 60)

    print(
        f"Loading session:\n{file_path}"
    )

    df = pd.read_csv(
        file_path
    )

    print(
        f"\nTotal samples: {len(df)}"
    )

    return df


# =============================================================
# PREPARE DATA
# =============================================================

def prepare_data(df):

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Seconds from beginning
    df["elapsed_seconds"] = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    # Convert boolean columns
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
                .str.lower()
                .map(
                    {
                        "true": 1,
                        "false": 0,
                    }
                )
                .fillna(0)
            )

    return df


# =============================================================
# CREATE OUTPUT DIRECTORY
# =============================================================

def create_output_directory():

    output_directory = "analysis"

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    return output_directory


# =============================================================
# GRAPH 1 — EAR
# =============================================================

def plot_ear(
    df,
    output_directory,
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        df["elapsed_seconds"],
        df["ear"],
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Eye Aspect Ratio (EAR)"
    )

    plt.title(
        "EAR Over Time"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    path = os.path.join(
        output_directory,
        "ear_over_time.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Created: {path}"
    )


# =============================================================
# GRAPH 2 — PERCLOS
# =============================================================

def plot_perclos(
    df,
    output_directory,
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        df["elapsed_seconds"],
        df["perclos"],
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "PERCLOS"
    )

    plt.title(
        "PERCLOS Over Time"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    path = os.path.join(
        output_directory,
        "perclos_over_time.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Created: {path}"
    )


# =============================================================
# GRAPH 3 — FATIGUE RISK
# =============================================================

def plot_fatigue_risk(
    df,
    output_directory,
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        df["elapsed_seconds"],
        df["fatigue_risk"],
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Fatigue Risk"
    )

    plt.title(
        "Fatigue Risk Over Time"
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    path = os.path.join(
        output_directory,
        "fatigue_risk_over_time.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Created: {path}"
    )


# =============================================================
# GRAPH 4 — EYE CLOSURE
# =============================================================

def plot_eye_closure(
    df,
    output_directory,
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        df["elapsed_seconds"],
        df["eye_closure_duration"],
    )

    plt.axhline(
        1.5,
        linestyle="--",
        label="1.5 sec alert threshold",
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Eye Closure Duration (seconds)"
    )

    plt.title(
        "Eye Closure Duration Over Time"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    path = os.path.join(
        output_directory,
        "eye_closure_duration.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Created: {path}"
    )


# =============================================================
# GRAPH 5 — HEAD POSE
# =============================================================

def plot_head_pose(
    df,
    output_directory,
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        df["elapsed_seconds"],
        df["pitch"],
        label="Pitch",
    )

    plt.plot(
        df["elapsed_seconds"],
        df["yaw"],
        label="Yaw",
    )

    plt.plot(
        df["elapsed_seconds"],
        df["roll"],
        label="Roll",
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Angle (degrees)"
    )

    plt.title(
        "Head Pose Over Time"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    path = os.path.join(
        output_directory,
        "head_pose_over_time.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Created: {path}"
    )


# =============================================================
# GRAPH 6 — RISK LEVEL DISTRIBUTION
# =============================================================

def plot_risk_distribution(
    df,
    output_directory,
):

    counts = (
        df["risk_level"]
        .value_counts()
    )

    plt.figure(
        figsize=(8, 5)
    )

    counts.plot(
        kind="bar"
    )

    plt.xlabel(
        "Risk Level"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Fatigue Risk Level Distribution"
    )

    plt.xticks(
        rotation=0
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    path = os.path.join(
        output_directory,
        "risk_level_distribution.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Created: {path}"
    )


# =============================================================
# GRAPH 7 — ALERT EVENTS
# =============================================================

def plot_alerts(
    df,
    output_directory,
):

    alert_rows = df[
        df["alert_triggered"] == 1
    ]

    plt.figure(
        figsize=(12, 4)
    )

    plt.scatter(
        alert_rows[
            "elapsed_seconds"
        ],
        alert_rows[
            "fatigue_risk"
        ],
        marker="x",
        s=80,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Fatigue Risk"
    )

    plt.title(
        "Safety Alert Events"
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    path = os.path.join(
        output_directory,
        "alert_events.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Created: {path}"
    )


# =============================================================
# GRAPH 8 — RELIABILITY
# =============================================================

def plot_reliability(
    df,
    output_directory,
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        df["elapsed_seconds"],
        df["reliability"],
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Reliability"
    )

    plt.title(
        "Signal Reliability Over Time"
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    path = os.path.join(
        output_directory,
        "reliability_over_time.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Created: {path}"
    )


# =============================================================
# SUMMARY
# =============================================================

def print_summary(df):

    print()
    print("=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)

    duration = (
        df["elapsed_seconds"].iloc[-1]
    )

    print(
        f"Session duration: "
        f"{duration:.1f} seconds"
    )

    print(
        f"Samples recorded: "
        f"{len(df)}"
    )

    print(
        f"Average EAR: "
        f"{df['ear'].mean():.3f}"
    )

    print(
        f"Average PERCLOS: "
        f"{df['perclos'].mean():.3f}"
    )

    print(
        f"Average fatigue risk: "
        f"{df['fatigue_risk'].mean():.3f}"
    )

    print(
        f"Maximum fatigue risk: "
        f"{df['fatigue_risk'].max():.3f}"
    )

    print(
        f"Average reliability: "
        f"{df['reliability'].mean():.3f}"
    )

    print(
        f"Total alerts: "
        f"{int(df['alert_triggered'].sum())}"
    )

    print()
    print("Risk levels:")

    print(
        df["risk_level"]
        .value_counts()
    )

    print("=" * 60)


# =============================================================
# MAIN
# =============================================================

def main():

    try:

        # Find newest session
        file_path = (
            find_latest_session()
        )

        # Load CSV
        df = load_data(
            file_path
        )

        # Prepare
        df = prepare_data(
            df
        )

        # Create analysis folder
        output_directory = (
            create_output_directory()
        )

        print()
        print(
            "Generating graphs..."
        )
        print()

        # Generate graphs
        plot_ear(
            df,
            output_directory,
        )

        plot_perclos(
            df,
            output_directory,
        )

        plot_fatigue_risk(
            df,
            output_directory,
        )

        plot_eye_closure(
            df,
            output_directory,
        )

        plot_head_pose(
            df,
            output_directory,
        )

        plot_risk_distribution(
            df,
            output_directory,
        )

        plot_alerts(
            df,
            output_directory,
        )

        plot_reliability(
            df,
            output_directory,
        )

        # Print summary
        print_summary(
            df
        )

        print()
        print(
            "Analysis completed successfully."
        )

        print(
            f"Graphs saved in: "
            f"{output_directory}/"
        )

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(error)


if __name__ == "__main__":
    main()