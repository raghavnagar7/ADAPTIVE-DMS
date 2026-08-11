"""
ADAPTIVE-DMS
Performance Report Generator

v1.1

Reads the latest session CSV and creates a
text-based performance report.
"""

import os
import glob

import pandas as pd


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

    df = pd.read_csv(
        file_path
    )

    if df.empty:

        raise ValueError(
            "The session CSV is empty."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["elapsed_seconds"] = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


# =============================================================
# BOOLEAN CONVERSION
# =============================================================

def convert_boolean_columns(df):

    columns = [
        "eyes_closed",
        "microsleep",
        "alert_triggered",
    ]

    for column in columns:

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
# RISK DISTRIBUTION
# =============================================================

def calculate_risk_distribution(df):

    counts = (
        df["risk_level"]
        .value_counts()
    )

    percentages = (
        df["risk_level"]
        .value_counts(
            normalize=True
        )
        * 100
    )

    return counts, percentages


# =============================================================
# EYE CLOSURE EVENTS
# =============================================================

def calculate_eye_closure_events(df):

    if "eyes_closed" not in df.columns:

        return 0

    eyes_closed = (
        df["eyes_closed"]
        .astype(int)
        .values
    )

    events = 0

    previous = 0

    for value in eyes_closed:

        if value == 1 and previous == 0:

            events += 1

        previous = value

    return events


# =============================================================
# ALERT EVENTS
# =============================================================

def calculate_alert_events(df):

    if "alert_triggered" not in df.columns:

        return 0

    return int(
        df["alert_triggered"]
        .sum()
    )


# =============================================================
# HIGH RISK EVENTS
# =============================================================

def calculate_high_risk_samples(df):

    high_levels = [
        "HIGH",
        "CRITICAL",
    ]

    return int(
        df["risk_level"]
        .isin(high_levels)
        .sum()
    )


# =============================================================
# MAX EYE CLOSURE
# =============================================================

def calculate_max_eye_closure(df):

    if "eye_closure_duration" not in df.columns:

        return 0.0

    return float(
        df[
            "eye_closure_duration"
        ].max()
    )


# =============================================================
# ALERT RESPONSE ANALYSIS
# =============================================================

def calculate_alert_response(df):

    """
    Estimates response time between the beginning
    of a continuous eye-closure event and the first
    alert recorded during that event.

    This is based only on logged data.
    """

    if (
        "eyes_closed" not in df.columns
        or "alert_triggered" not in df.columns
    ):

        return []

    responses = []

    eye_closure_start = None

    for _, row in df.iterrows():

        eyes_closed = (
            int(row["eyes_closed"])
            == 1
        )

        alert = (
            int(row["alert_triggered"])
            == 1
        )

        timestamp = (
            row["elapsed_seconds"]
        )

        # -----------------------------------------------------
        # Eye closure starts
        # -----------------------------------------------------

        if (
            eyes_closed
            and eye_closure_start is None
        ):

            eye_closure_start = timestamp

        # -----------------------------------------------------
        # Alert occurs
        # -----------------------------------------------------

        if (
            alert
            and eye_closure_start is not None
        ):

            response_time = (
                timestamp
                - eye_closure_start
            )

            responses.append(
                response_time
            )

            eye_closure_start = None

        # -----------------------------------------------------
        # Eyes open
        # -----------------------------------------------------

        if not eyes_closed:

            eye_closure_start = None

    return responses


# =============================================================
# PRINT REPORT
# =============================================================

def print_report(
    df,
    risk_counts,
    risk_percentages,
    eye_closure_events,
    alert_events,
    high_risk_samples,
    max_eye_closure,
    response_times,
):

    duration = float(
        df[
            "elapsed_seconds"
        ].iloc[-1]
    )

    samples = len(df)

    average_ear = float(
        df["ear"].mean()
    )

    average_perclos = float(
        df["perclos"].mean()
    )

    average_fatigue_risk = float(
        df["fatigue_risk"].mean()
    )

    maximum_fatigue_risk = float(
        df["fatigue_risk"].max()
    )

    average_reliability = float(
        df["reliability"].mean()
    )

    print()
    print("=" * 70)
    print(
        "ADAPTIVE-DMS PERFORMANCE REPORT"
    )
    print("=" * 70)

    print()

    print(
        f"Session Duration:       "
        f"{duration:.2f} seconds"
    )

    print(
        f"Samples Recorded:       "
        f"{samples}"
    )

    print()

    print(
        "-------------------- DRIVER METRICS --------------------"
    )

    print(
        f"Average EAR:            "
        f"{average_ear:.3f}"
    )

    print(
        f"Average PERCLOS:        "
        f"{average_perclos:.3f}"
    )

    print(
        f"Average Fatigue Risk:   "
        f"{average_fatigue_risk:.3f}"
    )

    print(
        f"Maximum Fatigue Risk:   "
        f"{maximum_fatigue_risk:.3f}"
    )

    print(
        f"Average Reliability:    "
        f"{average_reliability:.3f}"
    )

    print()

    print(
        "-------------------- SAFETY EVENTS ---------------------"
    )

    print(
        f"Eye Closure Events:     "
        f"{eye_closure_events}"
    )

    print(
        f"Maximum Eye Closure:    "
        f"{max_eye_closure:.2f} seconds"
    )

    print(
        f"High/Critical Samples:  "
        f"{high_risk_samples}"
    )

    print(
        f"Safety Alerts:          "
        f"{alert_events}"
    )

    print()

    print(
        "-------------------- RISK DISTRIBUTION -----------------"
    )

    risk_order = [
        "NORMAL",
        "LOW",
        "MODERATE",
        "HIGH",
        "CRITICAL",
    ]

    for level in risk_order:

        count = int(
            risk_counts.get(
                level,
                0,
            )
        )

        percentage = float(
            risk_percentages.get(
                level,
                0.0,
            )
        )

        print(
            f"{level:<20}"
            f"{count:>6} samples  "
            f"({percentage:>6.2f}%)"
        )

    print()

    print(
        "-------------------- ALERT RESPONSE --------------------"
    )

    if response_times:

        average_response = sum(
            response_times
        ) / len(
            response_times
        )

        fastest_response = min(
            response_times
        )

        slowest_response = max(
            response_times
        )

        print(
            f"Measured Alert Events: "
            f"{len(response_times)}"
        )

        print(
            f"Average Response Time:  "
            f"{average_response:.3f} seconds"
        )

        print(
            f"Fastest Response:        "
            f"{fastest_response:.3f} seconds"
        )

        print(
            f"Slowest Response:        "
            f"{slowest_response:.3f} seconds"
        )

    else:

        print(
            "No eye-closure alert response "
            "events could be measured."
        )

    print()

    print("=" * 70)

    print(
        "Report generated successfully."
    )

    print("=" * 70)


# =============================================================
# SAVE REPORT TO TEXT FILE
# =============================================================

def save_report(
    df,
    risk_counts,
    risk_percentages,
    eye_closure_events,
    alert_events,
    high_risk_samples,
    max_eye_closure,
    response_times,
    source_file,
):

    os.makedirs(
        "analysis",
        exist_ok=True,
    )

    output_file = os.path.join(
        "analysis",
        "performance_report.txt",
    )

    duration = float(
        df[
            "elapsed_seconds"
        ].iloc[-1]
    )

    average_response = (
        sum(response_times)
        / len(response_times)
        if response_times
        else None
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ADAPTIVE-DMS PERFORMANCE REPORT\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Source Session: {source_file}\n"
        )

        file.write(
            f"Session Duration: "
            f"{duration:.2f} seconds\n"
        )

        file.write(
            f"Samples Recorded: "
            f"{len(df)}\n\n"
        )

        file.write(
            "DRIVER METRICS\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"Average EAR: "
            f"{df['ear'].mean():.3f}\n"
        )

        file.write(
            f"Average PERCLOS: "
            f"{df['perclos'].mean():.3f}\n"
        )

        file.write(
            f"Average Fatigue Risk: "
            f"{df['fatigue_risk'].mean():.3f}\n"
        )

        file.write(
            f"Maximum Fatigue Risk: "
            f"{df['fatigue_risk'].max():.3f}\n"
        )

        file.write(
            f"Average Reliability: "
            f"{df['reliability'].mean():.3f}\n\n"
        )

        file.write(
            "SAFETY EVENTS\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"Eye Closure Events: "
            f"{eye_closure_events}\n"
        )

        file.write(
            f"Maximum Eye Closure: "
            f"{max_eye_closure:.2f} seconds\n"
        )

        file.write(
            f"High/Critical Samples: "
            f"{high_risk_samples}\n"
        )

        file.write(
            f"Safety Alerts: "
            f"{alert_events}\n\n"
        )

        file.write(
            "RISK DISTRIBUTION\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        for level in [
            "NORMAL",
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        ]:

            count = int(
                risk_counts.get(
                    level,
                    0,
                )
            )

            percentage = float(
                risk_percentages.get(
                    level,
                    0.0,
                )
            )

            file.write(
                f"{level}: "
                f"{count} samples "
                f"({percentage:.2f}%)\n"
            )

        file.write(
            "\nALERT RESPONSE\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        if average_response is not None:

            file.write(
                f"Measured Events: "
                f"{len(response_times)}\n"
            )

            file.write(
                f"Average Response Time: "
                f"{average_response:.3f} seconds\n"
            )

            file.write(
                f"Fastest Response: "
                f"{min(response_times):.3f} seconds\n"
            )

            file.write(
                f"Slowest Response: "
                f"{max(response_times):.3f} seconds\n"
            )

        else:

            file.write(
                "No measurable eye-closure "
                "alert response events.\n"
            )

    return output_file


# =============================================================
# MAIN
# =============================================================

def main():

    try:

        print(
            "Searching for latest session..."
        )

        file_path = (
            find_latest_session()
        )

        print(
            f"Using: {file_path}"
        )

        df = load_data(
            file_path
        )

        df = convert_boolean_columns(
            df
        )

        # -----------------------------------------------------
        # Calculations
        # -----------------------------------------------------

        risk_counts, risk_percentages = (
            calculate_risk_distribution(
                df
            )
        )

        eye_closure_events = (
            calculate_eye_closure_events(
                df
            )
        )

        alert_events = (
            calculate_alert_events(
                df
            )
        )

        high_risk_samples = (
            calculate_high_risk_samples(
                df
            )
        )

        max_eye_closure = (
            calculate_max_eye_closure(
                df
            )
        )

        response_times = (
            calculate_alert_response(
                df
            )
        )

        # -----------------------------------------------------
        # Print
        # -----------------------------------------------------

        print_report(
            df,
            risk_counts,
            risk_percentages,
            eye_closure_events,
            alert_events,
            high_risk_samples,
            max_eye_closure,
            response_times,
        )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        report_file = save_report(
            df,
            risk_counts,
            risk_percentages,
            eye_closure_events,
            alert_events,
            high_risk_samples,
            max_eye_closure,
            response_times,
            file_path,
        )

        print()
        print(
            f"Saved report: {report_file}"
        )

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(error)


if __name__ == "__main__":

    main()