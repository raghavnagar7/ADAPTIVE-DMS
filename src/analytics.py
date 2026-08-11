"""
ADAPTIVE-DMS

Session Analytics Engine

Version:
    v1.7

Purpose:
    Convert session and event data into
    useful driver-monitoring analytics.

Analytics:
    - Session duration
    - Sample count
    - Average fatigue risk
    - Maximum fatigue risk
    - Risk distribution
    - Risk-level duration
    - Eye closure statistics
    - Microsleep statistics
    - Yawning statistics
    - Gaze-away statistics
    - Intervention statistics
    - Alert statistics
    - Fatigue trend
"""


import os
from datetime import datetime

import pandas as pd


class SessionAnalytics:

    def __init__(
        self,
        session_dataframe=None,
        events_dataframe=None,
    ):

        if session_dataframe is None:

            session_dataframe = pd.DataFrame()

        if events_dataframe is None:

            events_dataframe = pd.DataFrame()

        self.df = (
            session_dataframe.copy()
        )

        self.events = (
            events_dataframe.copy()
        )

        self._prepare_data()

    # =========================================================
    # DATA PREPARATION
    # =========================================================

    def _prepare_data(self):

        # -----------------------------------------------------
        # SESSION TIMESTAMP
        # -----------------------------------------------------

        if (
            not self.df.empty
            and "timestamp" in self.df.columns
        ):

            self.df["timestamp"] = (
                pd.to_datetime(
                    self.df["timestamp"],
                    errors="coerce",
                )
            )

            self.df = self.df.dropna(
                subset=["timestamp"]
            )

        # -----------------------------------------------------
        # EVENT TIMESTAMP
        # -----------------------------------------------------

        if (
            not self.events.empty
            and "timestamp" in self.events.columns
        ):

            self.events["timestamp"] = (
                pd.to_datetime(
                    self.events["timestamp"],
                    errors="coerce",
                )
            )

            self.events = (
                self.events.dropna(
                    subset=["timestamp"]
                )
            )

        # -----------------------------------------------------
        # NUMERIC COLUMNS
        # -----------------------------------------------------

        numeric_columns = [

            "fatigue_risk",

            "ear",

            "mar",

            "perclos",

            "blink_count",

            "blink_duration",

            "microsleep_duration",

            "gaze_away_duration",

            "eye_closure_duration",

            "reliability",

            "pitch",

            "yaw",

            "roll",

            "elapsed_seconds",
        ]

        for column in numeric_columns:

            if column in self.df.columns:

                self.df[column] = (
                    pd.to_numeric(
                        self.df[column],
                        errors="coerce",
                    )
                )

    # =========================================================
    # SESSION DURATION
    # =========================================================

    def session_duration_seconds(self):

        if self.df.empty:

            return 0.0

        if "elapsed_seconds" in self.df.columns:

            values = self.df[
                "elapsed_seconds"
            ].dropna()

            if not values.empty:

                return max(
                    0.0,
                    float(values.iloc[-1]),
                )

        if "timestamp" in self.df.columns:

            start = self.df[
                "timestamp"
            ].min()

            end = self.df[
                "timestamp"
            ].max()

            if pd.notna(start) and pd.notna(end):

                return max(
                    0.0,
                    (
                        end - start
                    ).total_seconds(),
                )

        return 0.0

    # =========================================================
    # SAMPLE COUNT
    # =========================================================

    def sample_count(self):

        return int(
            len(self.df)
        )

    # =========================================================
    # FATIGUE RISK
    # =========================================================

    def average_fatigue_risk(self):

        if (
            self.df.empty
            or
            "fatigue_risk"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "fatigue_risk"
            ]
            .dropna()
        )

        if values.empty:

            return 0.0

        return float(
            values.mean()
        )

    def maximum_fatigue_risk(self):

        if (
            self.df.empty
            or
            "fatigue_risk"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "fatigue_risk"
            ]
            .dropna()
        )

        if values.empty:

            return 0.0

        return float(
            values.max()
        )

    def minimum_fatigue_risk(self):

        if (
            self.df.empty
            or
            "fatigue_risk"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "fatigue_risk"
            ]
            .dropna()
        )

        if values.empty:

            return 0.0

        return float(
            values.min()
        )

    # =========================================================
    # FATIGUE TREND
    # =========================================================

    def fatigue_trend(self):

        if (
            self.df.empty
            or
            "fatigue_risk"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "fatigue_risk"
            ]
            .dropna()
        )

        if len(values) < 2:

            return 0.0

        midpoint = len(values) // 2

        if midpoint == 0:

            return 0.0

        first_half = (
            values.iloc[:midpoint]
            .mean()
        )

        second_half = (
            values.iloc[midpoint:]
            .mean()
        )

        return float(
            second_half
            - first_half
        )

    # =========================================================
    # RISK DISTRIBUTION
    # =========================================================

    def risk_distribution(self):

        result = {

            "NORMAL": 0,

            "LOW": 0,

            "MODERATE": 0,

            "HIGH": 0,

            "CRITICAL": 0,
        }

        if self.df.empty:

            return result

        # -----------------------------------------------------
        # Prefer recorded risk level
        # -----------------------------------------------------

        if "risk_level" in self.df.columns:

            levels = (
                self.df[
                    "risk_level"
                ]
                .astype(str)
                .str.upper()
            )

            for level in result:

                result[level] = int(
                    (
                        levels
                        == level
                    ).sum()
                )

            return result

        # -----------------------------------------------------
        # Fallback to fatigue risk
        # -----------------------------------------------------

        if "fatigue_risk" not in self.df.columns:

            return result

        risk = (
            pd.to_numeric(
                self.df[
                    "fatigue_risk"
                ],
                errors="coerce",
            )
            .fillna(0.0)
        )

        result["NORMAL"] = int(
            (
                risk < 0.20
            ).sum()
        )

        result["LOW"] = int(
            (
                (risk >= 0.20)
                & (risk < 0.35)
            ).sum()
        )

        result["MODERATE"] = int(
            (
                (risk >= 0.35)
                & (risk < 0.55)
            ).sum()
        )

        result["HIGH"] = int(
            (
                (risk >= 0.55)
                & (risk < 0.75)
            ).sum()
        )

        result["CRITICAL"] = int(
            (
                risk >= 0.75
            ).sum()
        )

        return result

    # =========================================================
    # RISK DURATIONS
    # =========================================================

    def risk_duration_seconds(self):

        result = {

            "NORMAL": 0.0,

            "LOW": 0.0,

            "MODERATE": 0.0,

            "HIGH": 0.0,

            "CRITICAL": 0.0,
        }

        if self.df.empty:

            return result

        if (
            "timestamp"
            not in self.df.columns
        ):

            return result

        if "risk_level" in self.df.columns:

            levels = (
                self.df[
                    "risk_level"
                ]
                .astype(str)
                .str.upper()
            )

        elif "fatigue_risk" in self.df.columns:

            risk = (
                pd.to_numeric(
                    self.df[
                        "fatigue_risk"
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
            )

            levels = pd.Series(
                "NORMAL",
                index=self.df.index,
            )

            levels[
                risk >= 0.20
            ] = "LOW"

            levels[
                risk >= 0.35
            ] = "MODERATE"

            levels[
                risk >= 0.55
            ] = "HIGH"

            levels[
                risk >= 0.75
            ] = "CRITICAL"

        else:

            return result

        timestamps = (
            self.df[
                "timestamp"
            ]
            .reset_index(
                drop=True
            )
        )

        levels = (
            levels
            .reset_index(
                drop=True
            )
        )

        if len(timestamps) < 2:

            return result

        for index in range(
            len(timestamps) - 1
        ):

            current_level = (
                levels.iloc[index]
            )

            if (
                current_level
                not in result
            ):

                continue

            delta = (
                timestamps.iloc[
                    index + 1
                ]
                - timestamps.iloc[
                    index
                ]
            ).total_seconds()

            if delta < 0:

                continue

            # Prevent one malformed
            # interval from dominating.

            delta = min(
                delta,
                5.0,
            )

            result[
                current_level
            ] += float(delta)

        return result

    # =========================================================
    # EVENT COUNT
    # =========================================================

    def event_count(
        self,
        event_type,
    ):

        if self.events.empty:

            return 0

        if (
            "event_type"
            not in self.events.columns
        ):

            return 0

        values = (
            self.events[
                "event_type"
            ]
            .astype(str)
            .str.upper()
        )

        return int(
            (
                values
                == event_type.upper()
            ).sum()
        )

    # =========================================================
    # TOTAL EVENTS
    # =========================================================

    def total_events(self):

        return int(
            len(self.events)
        )

    # =========================================================
    # EVENT SUMMARY
    # =========================================================

    def event_summary(self):

        return {

            "total_events":
                self.total_events(),

            "eye_closure":
                self.event_count(
                    "EYE_CLOSURE"
                ),

            "microsleep":
                self.event_count(
                    "MICROSLEEP"
                ),

            "yawning":
                self.event_count(
                    "YAWNING"
                ),

            "gaze_away":
                self.event_count(
                    "GAZE_AWAY"
                ),

            "risk_transition":
                self.event_count(
                    "RISK_TRANSITION"
                ),

            "intervention":
                self.event_count(
                    "INTERVENTION"
                ),
        }

    # =========================================================
    # SESSION ALERT COUNT
    # =========================================================

    def session_alert_count(self):

        if self.df.empty:

            return 0

        if (
            "alert_triggered"
            not in self.df.columns
        ):

            return 0

        values = (
            self.df[
                "alert_triggered"
            ]
            .astype(str)
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

        return int(
            values.sum()
        )

    # =========================================================
    # AVERAGE EAR
    # =========================================================

    def average_ear(self):

        if (
            self.df.empty
            or
            "ear"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "ear"
            ]
            .dropna()
        )

        if values.empty:

            return 0.0

        return float(
            values.mean()
        )

    # =========================================================
    # AVERAGE MAR
    # =========================================================

    def average_mar(self):

        if (
            self.df.empty
            or
            "mar"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "mar"
            ]
            .dropna()
        )

        if values.empty:

            return 0.0

        return float(
            values.mean()
        )

    # =========================================================
    # AVERAGE PERCLOS
    # =========================================================

    def average_perclos(self):

        if (
            self.df.empty
            or
            "perclos"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "perclos"
            ]
            .dropna()
        )

        if values.empty:

            return 0.0

        return float(
            values.mean()
        )

    # =========================================================
    # MAX EYE CLOSURE
    # =========================================================

    def maximum_eye_closure(self):

        if (
            self.df.empty
            or
            "eye_closure_duration"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "eye_closure_duration"
            ]
            .dropna()
        )

        if values.empty:

            return 0.0

        return float(
            values.max()
        )

    # =========================================================
    # MAX MICROSLEEP
    # =========================================================

    def maximum_microsleep(self):

        if (
            self.df.empty
            or
            "microsleep_duration"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "microsleep_duration"
            ]
            .dropna()
        )

        if values.empty:

            return 0.0

        return float(
            values.max()
        )

    # =========================================================
    # AVERAGE RELIABILITY
    # =========================================================

    def average_reliability(self):

        if (
            self.df.empty
            or
            "reliability"
            not in self.df.columns
        ):

            return 0.0

        values = (
            self.df[
                "reliability"
            ]
            .dropna()
        )

        if values.empty:

            return 0.0

        return float(
            values.mean()
        )

    # =========================================================
    # FULL SUMMARY
    # =========================================================

    def summary(self):

        risk_distribution = (
            self.risk_distribution()
        )

        risk_duration = (
            self.risk_duration_seconds()
        )

        event_summary = (
            self.event_summary()
        )

        return {

            "session_duration_seconds":
                self.session_duration_seconds(),

            "sample_count":
                self.sample_count(),

            "average_fatigue_risk":
                self.average_fatigue_risk(),

            "maximum_fatigue_risk":
                self.maximum_fatigue_risk(),

            "minimum_fatigue_risk":
                self.minimum_fatigue_risk(),

            "fatigue_trend":
                self.fatigue_trend(),

            "average_ear":
                self.average_ear(),

            "average_mar":
                self.average_mar(),

            "average_perclos":
                self.average_perclos(),

            "maximum_eye_closure":
                self.maximum_eye_closure(),

            "maximum_microsleep":
                self.maximum_microsleep(),

            "average_reliability":
                self.average_reliability(),

            "session_alert_count":
                self.session_alert_count(),

            "risk_distribution":
                risk_distribution,

            "risk_duration":
                risk_duration,

            "event_summary":
                event_summary,
        }

    # =========================================================
    # REPORT DATAFRAME
    # =========================================================

    def report_dataframe(self):

        summary = self.summary()

        risk_distribution = (
            summary[
                "risk_distribution"
            ]
        )

        risk_duration = (
            summary[
                "risk_duration"
            ]
        )

        events = (
            summary[
                "event_summary"
            ]
        )

        rows = [

            [
                "Session Duration (seconds)",
                summary[
                    "session_duration_seconds"
                ],
            ],

            [
                "Sample Count",
                summary[
                    "sample_count"
                ],
            ],

            [
                "Average Fatigue Risk",
                summary[
                    "average_fatigue_risk"
                ],
            ],

            [
                "Maximum Fatigue Risk",
                summary[
                    "maximum_fatigue_risk"
                ],
            ],

            [
                "Minimum Fatigue Risk",
                summary[
                    "minimum_fatigue_risk"
                ],
            ],

            [
                "Fatigue Trend",
                summary[
                    "fatigue_trend"
                ],
            ],

            [
                "Average EAR",
                summary[
                    "average_ear"
                ],
            ],

            [
                "Average MAR",
                summary[
                    "average_mar"
                ],
            ],

            [
                "Average PERCLOS",
                summary[
                    "average_perclos"
                ],
            ],

            [
                "Maximum Eye Closure (seconds)",
                summary[
                    "maximum_eye_closure"
                ],
            ],

            [
                "Maximum Microsleep (seconds)",
                summary[
                    "maximum_microsleep"
                ],
            ],

            [
                "Average Reliability",
                summary[
                    "average_reliability"
                ],
            ],

            [
                "Session Alerts",
                summary[
                    "session_alert_count"
                ],
            ],

            [
                "Eye Closure Events",
                events[
                    "eye_closure"
                ],
            ],

            [
                "Microsleep Events",
                events[
                    "microsleep"
                ],
            ],

            [
                "Yawning Events",
                events[
                    "yawning"
                ],
            ],

            [
                "Gaze Away Events",
                events[
                    "gaze_away"
                ],
            ],

            [
                "Risk Transitions",
                events[
                    "risk_transition"
                ],
            ],

            [
                "Interventions",
                events[
                    "intervention"
                ],
            ],

            [
                "Normal Samples",
                risk_distribution[
                    "NORMAL"
                ],
            ],

            [
                "Low Risk Samples",
                risk_distribution[
                    "LOW"
                ],
            ],

            [
                "Moderate Risk Samples",
                risk_distribution[
                    "MODERATE"
                ],
            ],

            [
                "High Risk Samples",
                risk_distribution[
                    "HIGH"
                ],
            ],

            [
                "Critical Risk Samples",
                risk_distribution[
                    "CRITICAL"
                ],

            ],

            [
                "Normal Duration (seconds)",
                risk_duration[
                    "NORMAL"
                ],
            ],

            [
                "Low Risk Duration (seconds)",
                risk_duration[
                    "LOW"
                ],
            ],

            [
                "Moderate Risk Duration (seconds)",
                risk_duration[
                    "MODERATE"
                ],
            ],

            [
                "High Risk Duration (seconds)",
                risk_duration[
                    "HIGH"
                ],
            ],

            [
                "Critical Risk Duration (seconds)",
                risk_duration[
                    "CRITICAL"
                ],
            ],
        ]

        return pd.DataFrame(
            rows,
            columns=[
                "Metric",
                "Value",
            ],
        )

    # =========================================================
    # CSV EXPORT
    # =========================================================

    def export_csv(
        self,
        output_path,
    ):

        report = (
            self.report_dataframe()
        )

        directory = os.path.dirname(
            output_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        report.to_csv(
            output_path,
            index=False,
        )

        return output_path

    # =========================================================
    # TEXT REPORT
    # =========================================================

    def generate_text_report(self):

        summary = self.summary()

        events = (
            summary[
                "event_summary"
            ]
        )

        risk = (
            summary[
                "risk_distribution"
            ]
        )

        duration = (
            summary[
                "risk_duration"
            ]
        )

        session_seconds = (
            summary[
                "session_duration_seconds"
            ]
        )

        minutes = int(
            session_seconds // 60
        )

        seconds = int(
            session_seconds % 60
        )

        trend = (
            summary[
                "fatigue_trend"
            ]
        )

        if trend > 0.02:

            trend_text = (
                "Increasing fatigue"
            )

        elif trend < -0.02:

            trend_text = (
                "Decreasing fatigue"
            )

        else:

            trend_text = (
                "Stable fatigue"
            )

        report = []

        report.append(
            "ADAPTIVE-DMS"
        )

        report.append(
            "Advanced Session Analytics Report"
        )

        report.append(
            "=" * 60
        )

        report.append(
            ""
        )

        report.append(
            "SESSION SUMMARY"
        )

        report.append(
            "-" * 60
        )

        report.append(
            f"Session Duration: "
            f"{minutes:02d}:{seconds:02d}"
        )

        report.append(
            f"Samples: "
            f"{summary['sample_count']}"
        )

        report.append(
            f"Average Fatigue Risk: "
            f"{summary['average_fatigue_risk']:.3f}"
        )

        report.append(
            f"Maximum Fatigue Risk: "
            f"{summary['maximum_fatigue_risk']:.3f}"
        )

        report.append(
            f"Minimum Fatigue Risk: "
            f"{summary['minimum_fatigue_risk']:.3f}"
        )

        report.append(
            f"Fatigue Trend: "
            f"{trend:.3f} "
            f"({trend_text})"
        )

        report.append(
            ""
        )

        report.append(
            "DRIVER METRICS"
        )

        report.append(
            "-" * 60
        )

        report.append(
            f"Average EAR: "
            f"{summary['average_ear']:.3f}"
        )

        report.append(
            f"Average MAR: "
            f"{summary['average_mar']:.3f}"
        )

        report.append(
            f"Average PERCLOS: "
            f"{summary['average_perclos']:.3f}"
        )

        report.append(
            f"Maximum Eye Closure: "
            f"{summary['maximum_eye_closure']:.2f}s"
        )

        report.append(
            f"Maximum Microsleep: "
            f"{summary['maximum_microsleep']:.2f}s"
        )

        report.append(
            f"Average Reliability: "
            f"{summary['average_reliability']:.3f}"
        )

        report.append(
            ""
        )

        report.append(
            "EVENT SUMMARY"
        )

        report.append(
            "-" * 60
        )

        report.append(
            f"Total Events: "
            f"{events['total_events']}"
        )

        report.append(
            f"Eye Closure: "
            f"{events['eye_closure']}"
        )

        report.append(
            f"Microsleep: "
            f"{events['microsleep']}"
        )

        report.append(
            f"Yawning: "
            f"{events['yawning']}"
        )

        report.append(
            f"Gaze Away: "
            f"{events['gaze_away']}"
        )

        report.append(
            f"Risk Transitions: "
            f"{events['risk_transition']}"
        )

        report.append(
            f"Interventions: "
            f"{events['intervention']}"
        )

        report.append(
            ""
        )

        report.append(
            "RISK DISTRIBUTION"
        )

        report.append(
            "-" * 60
        )

        for level in [
            "NORMAL",
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        ]:

            report.append(
                f"{level}: "
                f"{risk[level]} samples"
            )

        report.append(
            ""
        )

        report.append(
            "RISK DURATION"
        )

        report.append(
            "-" * 60
        )

        for level in [
            "NORMAL",
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        ]:

            report.append(
                f"{level}: "
                f"{duration[level]:.2f}s"
            )

        report.append(
            ""
        )

        report.append(
            "Generated by ADAPTIVE-DMS v1.7"
        )

        report.append(
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        return "\n".join(
            report
        )

    # =========================================================
    # TEXT REPORT EXPORT
    # =========================================================

    def export_text(
        self,
        output_path,
    ):

        directory = os.path.dirname(
            output_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        report = (
            self.generate_text_report()
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                report
            )

        return output_path