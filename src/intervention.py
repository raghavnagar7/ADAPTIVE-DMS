"""
ADAPTIVE-DMS
Adaptive Safety Intervention

v0.8

Direct eye-closure safety trigger:
    Eyes continuously closed for >= 1.5 seconds
    -> HIGH warning + beep.

Other fatigue conditions use:
    - Fatigue risk
    - Temporal state
    - Reliability
    - Persistence
    - Cooldown
"""

import time

try:
    import winsound
    WINDOWS_AUDIO = True
except ImportError:
    WINDOWS_AUDIO = False


class AdaptiveSafetyIntervention:

    def __init__(
        self,
        low_threshold=0.20,
        moderate_threshold=0.35,
        high_threshold=0.55,
        critical_threshold=0.75,
        minimum_reliability=0.40,
        persistence_seconds=1.5,
        cooldown_seconds=5.0,
        eye_closure_trigger_seconds=1.5,
    ):

        self.low_threshold = low_threshold
        self.moderate_threshold = moderate_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold

        self.minimum_reliability = (
            minimum_reliability
        )

        self.persistence_seconds = (
            persistence_seconds
        )

        self.cooldown_seconds = (
            cooldown_seconds
        )

        self.eye_closure_trigger_seconds = (
            eye_closure_trigger_seconds
        )

        self.current_level = "NORMAL"
        self.pending_level = "NORMAL"
        self.pending_since = None

        self.last_alert_time = 0.0
        self.alert_count = 0

    # =========================================================
    # RISK CLASSIFICATION
    # =========================================================

    def classify_risk(self, risk):

        if risk >= self.critical_threshold:
            return "CRITICAL"

        if risk >= self.high_threshold:
            return "HIGH"

        if risk >= self.moderate_threshold:
            return "MODERATE"

        if risk >= self.low_threshold:
            return "LOW"

        return "NORMAL"

    # =========================================================
    # PRIORITY
    # =========================================================

    @staticmethod
    def level_priority(level):

        priorities = {
            "NORMAL": 0,
            "LOW": 1,
            "MODERATE": 2,
            "HIGH": 3,
            "CRITICAL": 4,
        }

        return priorities.get(level, 0)

    # =========================================================
    # MESSAGE
    # =========================================================

    @staticmethod
    def message_for_level(level):

        messages = {

            "NORMAL":
                "Driver state normal",

            "LOW":
                "Stay attentive",

            "MODERATE":
                "CAUTION: Signs of fatigue",

            "HIGH":
                "WARNING: Driver fatigue detected",

            "CRITICAL":
                "CRITICAL: Take a break immediately",
        }

        return messages.get(
            level,
            "Driver state normal",
        )

    # =========================================================
    # AUDIO ALERT
    # =========================================================

    def _play_alert(self, level):

        if not WINDOWS_AUDIO:
            return

        try:

            if level == "LOW":

                winsound.Beep(
                    700,
                    150,
                )

            elif level == "MODERATE":

                winsound.Beep(
                    900,
                    250,
                )

                time.sleep(0.05)

                winsound.Beep(
                    900,
                    250,
                )

            elif level == "HIGH":

                for _ in range(3):

                    winsound.Beep(
                        1200,
                        220,
                    )

                    time.sleep(0.06)

            elif level == "CRITICAL":

                for _ in range(4):

                    winsound.Beep(
                        1500,
                        250,
                    )

                    time.sleep(0.06)

        except Exception:
            pass

    # =========================================================
    # COOLDOWN
    # =========================================================

    def _can_alert(self, timestamp):

        return (
            timestamp - self.last_alert_time
            >= self.cooldown_seconds
        )

    # =========================================================
    # MAIN UPDATE
    # =========================================================

    def update(
        self,
        fatigue_risk,
        temporal_state,
        reliability,
        timestamp=None,
        eyes_closed=False,
        eye_closure_duration=0.0,
    ):

        if timestamp is None:
            timestamp = time.time()

        fatigue_risk = max(
            0.0,
            min(
                1.0,
                float(fatigue_risk),
            ),
        )

        reliability = max(
            0.0,
            min(
                1.0,
                float(reliability),
            ),
        )

        eye_closure_duration = max(
            0.0,
            float(eye_closure_duration),
        )

        # =====================================================
        # NORMAL RISK LEVEL
        # =====================================================

        raw_level = self.classify_risk(
            fatigue_risk
        )

        # =====================================================
        # RELIABILITY PROTECTION
        # =====================================================

        if (
            reliability < self.minimum_reliability
            and raw_level in (
                "HIGH",
                "CRITICAL",
            )
        ):

            raw_level = "MODERATE"

        # =====================================================
        # TEMPORAL ESCALATION
        # =====================================================

        if temporal_state == "CRITICAL":

            if reliability >= self.minimum_reliability:
                raw_level = "CRITICAL"

        elif temporal_state == "HIGH":

            if (
                reliability >= self.minimum_reliability
                and self.level_priority(
                    raw_level
                ) < 3
            ):

                raw_level = "HIGH"

        elif temporal_state == "INCREASING":

            if fatigue_risk >= self.moderate_threshold:
                raw_level = "MODERATE"

        # =====================================================
        # DIRECT EYE-CLOSURE TRIGGER
        # =====================================================

        eye_closure_triggered = (
            eyes_closed
            and
            eye_closure_duration
            >= self.eye_closure_trigger_seconds
        )

        # =====================================================
        # EYE CLOSURE HAS HIGHEST PRIORITY
        # =====================================================

        if eye_closure_triggered:

            self.current_level = "HIGH"
            self.pending_level = "HIGH"
            self.pending_since = timestamp

        else:

            # ---------------------------------------------
            # NORMAL PERSISTENCE LOGIC
            # ---------------------------------------------

            if raw_level != self.pending_level:

                self.pending_level = raw_level
                self.pending_since = timestamp

            if self.pending_since is None:

                self.pending_since = timestamp

            persistence = (
                timestamp
                - self.pending_since
            )

            if (
                persistence
                >= self.persistence_seconds
            ):

                self.current_level = (
                    self.pending_level
                )

        # =====================================================
        # AUDIO ALERT
        # =====================================================

        alert_triggered = False

        # Direct eye closure:
        # alert immediately after 1.5 seconds.
        if eye_closure_triggered:

            if self._can_alert(timestamp):

                alert_triggered = True

                self.last_alert_time = timestamp

                self.alert_count += 1

                self._play_alert("HIGH")

        # Other fatigue conditions:
        # use normal intervention logic.
        elif (
            self.level_priority(
                self.current_level
            )
            >= self.level_priority(
                "MODERATE"
            )
        ):

            if self._can_alert(timestamp):

                alert_triggered = True

                self.last_alert_time = timestamp

                self.alert_count += 1

                self._play_alert(
                    self.current_level
                )

        # =====================================================
        # ACTION
        # =====================================================

        action = self._action_for_level(
            self.current_level
        )

        return {

            "level":
                self.current_level,

            "raw_level":
                raw_level,

            "message":
                self.message_for_level(
                    self.current_level
                ),

            "action":
                action,

            "alert_triggered":
                alert_triggered,

            "alert_count":
                self.alert_count,

            "persistence":
                eye_closure_duration,

            "reliability":
                reliability,

            "eye_closure_triggered":
                eye_closure_triggered,
        }

    # =========================================================
    # ACTION
    # =========================================================

    @staticmethod
    def _action_for_level(level):

        actions = {

            "NORMAL":
                "NO_ACTION",

            "LOW":
                "VISUAL_REMINDER",

            "MODERATE":
                "AUDIO_VISUAL_WARNING",

            "HIGH":
                "STRONG_WARNING",

            "CRITICAL":
                "CRITICAL_WARNING",
        }

        return actions.get(
            level,
            "NO_ACTION",
        )