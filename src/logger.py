"""
ADAPTIVE-DMS
Session Data Logger

Stores driver monitoring data in CSV format
for later analysis and visualization.
"""

import csv
import os
from datetime import datetime


class SessionLogger:

    def __init__(self, log_directory="logs"):

        self.log_directory = log_directory

        os.makedirs(
            self.log_directory,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.file_path = os.path.join(
            self.log_directory,
            f"session_{timestamp}.csv",
        )

        self.file = open(
            self.file_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp",
            "ear",
            "mar",
            "perclos",
            "eyes_closed",
            "blink_count",
            "blink_duration",
            "microsleep",
            "microsleep_duration",
            "pitch",
            "yaw",
            "roll",
            "gaze_direction",
            "gaze_away_duration",
            "reliability",
            "fatigue_risk",
            "risk_level",
            "temporal_state",
            "intervention_level",
            "intervention_action",
            "alert_triggered",
            "eye_closure_duration",
        ])

        self.file.flush()

    def log(
        self,
        timestamp,
        driver_values,
        pose_values,
        gaze_values,
        reliability_values,
        fusion_values,
        temporal_values,
        intervention_values,
        eye_closure_duration,
    ):

        self.writer.writerow([

            datetime.fromtimestamp(
                timestamp
            ).isoformat(),

            driver_values.get(
                "ear",
                0.0,
            ),

            driver_values.get(
                "mar",
                0.0,
            ),

            driver_values.get(
                "perclos",
                0.0,
            ),

            driver_values.get(
                "eyes_closed",
                False,
            ),

            driver_values.get(
                "blink_count",
                0,
            ),

            driver_values.get(
                "blink_duration",
                0.0,
            ),

            driver_values.get(
                "microsleep",
                False,
            ),

            driver_values.get(
                "microsleep_duration",
                0.0,
            ),

            pose_values.get(
                "pitch",
                0.0,
            ),

            pose_values.get(
                "yaw",
                0.0,
            ),

            pose_values.get(
                "roll",
                0.0,
            ),

            gaze_values.get(
                "gaze_direction",
                "UNKNOWN",
            ),

            gaze_values.get(
                "gaze_away_duration",
                0.0,
            ),

            reliability_values.get(
                "overall_reliability",
                0.0,
            ),

            fusion_values.get(
                "fatigue_risk",
                0.0,
            ),

            fusion_values.get(
                "risk_level",
                "UNKNOWN",
            ),

            temporal_values.get(
                "state",
                "UNKNOWN",
            ),

            intervention_values.get(
                "level",
                "UNKNOWN",
            ),

            intervention_values.get(
                "action",
                "UNKNOWN",
            ),

            intervention_values.get(
                "alert_triggered",
                False,
            ),

            eye_closure_duration,
        ])

        self.file.flush()

    def close(self):

        if self.file:

            self.file.close()

            self.file = None