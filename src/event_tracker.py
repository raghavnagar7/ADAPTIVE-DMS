"""
ADAPTIVE-DMS
Event Tracker

Version:
    v1.6

Tracks:
    - Eye closure
    - Microsleep
    - Yawning
    - Gaze away
    - Risk transitions
    - Safety interventions
"""

import os
import csv
import time
from datetime import datetime


class EventTracker:

    def __init__(
        self,
        log_directory="logs",
        eye_closure_threshold=1.5,
        event_cooldown=2.0,
    ):

        self.log_directory = log_directory

        self.eye_closure_threshold = (
            eye_closure_threshold
        )

        self.event_cooldown = (
            event_cooldown
        )

        os.makedirs(
            self.log_directory,
            exist_ok=True,
        )

        self.file_path = os.path.join(
            self.log_directory,
            "events.csv",
        )

        self.previous_risk_level = None
        self.previous_intervention_level = None

        self.eye_closure_active = False
        self.microsleep_active = False
        self.yawning_active = False
        self.gaze_away_active = False

        self.last_event_times = {}

        self.event_count = 0

        self._create_file()

    # =========================================================
    # CREATE EVENT FILE
    # =========================================================

    def _create_file(self):

        if os.path.exists(
            self.file_path
        ):

            return

        with open(
            self.file_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "timestamp",
                    "event_type",
                    "severity",
                    "message",
                ]
            )

    # =========================================================
    # COOLDOWN
    # =========================================================

    def _can_log(
        self,
        event_type,
        timestamp,
    ):

        last_time = (
            self.last_event_times.get(
                event_type
            )
        )

        if last_time is None:

            return True

        return (
            timestamp - last_time
            >= self.event_cooldown
        )

    # =========================================================
    # LOG EVENT
    # =========================================================

    def _log_event(
        self,
        event_type,
        severity,
        message,
        timestamp,
    ):

        if not self._can_log(
            event_type,
            timestamp,
        ):

            return False

        self.last_event_times[
            event_type
        ] = timestamp

        timestamp_text = (
            datetime.fromtimestamp(
                timestamp
            ).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        with open(
            self.file_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    timestamp_text,
                    event_type,
                    severity,
                    message,
                ]
            )

        self.event_count += 1

        return True

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        timestamp=None,
        driver_values=None,
        gaze_values=None,
        fusion_values=None,
        intervention_values=None,
        eye_closure_duration=0.0,
    ):

        if timestamp is None:

            timestamp = time.time()

        if driver_values is None:

            driver_values = {}

        if gaze_values is None:

            gaze_values = {}

        if fusion_values is None:

            fusion_values = {}

        if intervention_values is None:

            intervention_values = {}

        events_created = []

        # =====================================================
        # EYE CLOSURE
        # =====================================================

        eyes_closed = bool(
            driver_values.get(
                "eyes_closed",
                False,
            )
        )

        if (
            eyes_closed
            and
            eye_closure_duration
            >= self.eye_closure_threshold
        ):

            if not self.eye_closure_active:

                created = self._log_event(
                    event_type="EYE_CLOSURE",
                    severity="HIGH",
                    message=(
                        "Eyes closed for "
                        f"{eye_closure_duration:.1f}s"
                    ),
                    timestamp=timestamp,
                )

                if created:

                    events_created.append(
                        "EYE_CLOSURE"
                    )

                self.eye_closure_active = True

        else:

            self.eye_closure_active = False

        # =====================================================
        # MICROSLEEP
        # =====================================================

        microsleep = bool(
            driver_values.get(
                "microsleep",
                False,
            )
        )

        if microsleep:

            if not self.microsleep_active:

                created = self._log_event(
                    event_type="MICROSLEEP",
                    severity="CRITICAL",
                    message="Microsleep detected",
                    timestamp=timestamp,
                )

                if created:

                    events_created.append(
                        "MICROSLEEP"
                    )

                self.microsleep_active = True

        else:

            self.microsleep_active = False

        # =====================================================
        # YAWNING
        # =====================================================

        yawning = bool(
            driver_values.get(
                "yawning",
                False,
            )
        )

        if yawning:

            if not self.yawning_active:

                created = self._log_event(
                    event_type="YAWNING",
                    severity="MODERATE",
                    message="Yawning detected",
                    timestamp=timestamp,
                )

                if created:

                    events_created.append(
                        "YAWNING"
                    )

                self.yawning_active = True

        else:

            self.yawning_active = False

        # =====================================================
        # GAZE AWAY
        # =====================================================

        prolonged_gaze_away = bool(
            gaze_values.get(
                "prolonged_gaze_away",
                False,
            )
        )

        gaze_away_duration = float(
            gaze_values.get(
                "gaze_away_duration",
                0.0,
            )
            or 0.0
        )

        if prolonged_gaze_away:

            if not self.gaze_away_active:

                created = self._log_event(
                    event_type="GAZE_AWAY",
                    severity="MODERATE",
                    message=(
                        "Prolonged gaze away "
                        f"for {gaze_away_duration:.1f}s"
                    ),
                    timestamp=timestamp,
                )

                if created:

                    events_created.append(
                        "GAZE_AWAY"
                    )

                self.gaze_away_active = True

        else:

            self.gaze_away_active = False

        # =====================================================
        # RISK LEVEL TRANSITION
        # =====================================================

        current_risk_level = str(
            fusion_values.get(
                "risk_level",
                "UNKNOWN",
            )
        )

        if (
            self.previous_risk_level
            is not None
            and
            current_risk_level
            != self.previous_risk_level
        ):

            created = self._log_event(
                event_type="RISK_TRANSITION",
                severity=current_risk_level,
                message=(
                    f"Risk changed from "
                    f"{self.previous_risk_level} "
                    f"to "
                    f"{current_risk_level}"
                ),
                timestamp=timestamp,
            )

            if created:

                events_created.append(
                    "RISK_TRANSITION"
                )

        self.previous_risk_level = (
            current_risk_level
        )

        # =====================================================
        # INTERVENTION LEVEL TRANSITION
        # =====================================================

        current_intervention_level = str(
            intervention_values.get(
                "level",
                "UNKNOWN",
            )
        )

        if (
            self.previous_intervention_level
            is not None
            and
            current_intervention_level
            != self.previous_intervention_level
        ):

            created = self._log_event(
                event_type="INTERVENTION",
                severity=current_intervention_level,
                message=(
                    f"Intervention changed from "
                    f"{self.previous_intervention_level} "
                    f"to "
                    f"{current_intervention_level}"
                ),
                timestamp=timestamp,
            )

            if created:

                events_created.append(
                    "INTERVENTION"
                )

        self.previous_intervention_level = (
            current_intervention_level
        )

        # =====================================================
        # RETURN
        # =====================================================

        return {
            "events_created": events_created,
            "event_count": self.event_count,
            "file_path": self.file_path,
        }

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        return True