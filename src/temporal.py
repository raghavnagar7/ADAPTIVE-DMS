"""
Temporal fatigue analysis for ADAPTIVE-DMS.

v0.7

Analyzes fatigue-risk history over time.

This version uses:
    - Risk history
    - Moving average
    - Short-term trend
    - Risk persistence
    - Acceleration

It is a temporal analysis layer, not a trained ML model.
A learned GRU/LSTM model can be added later after collecting
properly labeled temporal driving data.
"""

from collections import deque


class TemporalFatiguePredictor:
    """
    Analyze fatigue risk across a sequence of observations.
    """

    def __init__(
        self,
        history_seconds=30.0,
        sample_interval=0.5,
        short_window_seconds=5.0,
        medium_window_seconds=15.0,
        increasing_threshold=0.015,
        high_risk_threshold=0.60,
    ):

        self.history_seconds = history_seconds

        self.sample_interval = sample_interval

        self.short_window_seconds = (
            short_window_seconds
        )

        self.medium_window_seconds = (
            medium_window_seconds
        )

        self.increasing_threshold = (
            increasing_threshold
        )

        self.high_risk_threshold = (
            high_risk_threshold
        )

        max_samples = max(
            10,
            int(
                history_seconds
                / sample_interval
            ),
        )

        self.history = deque(
            maxlen=max_samples
        )

        self.last_update_time = None

        self.previous_short_average = None

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    @staticmethod
    def clamp(
        value,
        minimum=0.0,
        maximum=1.0,
    ):
        """Clamp value to [0, 1]."""

        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    # ---------------------------------------------------------
    # Add observation
    # ---------------------------------------------------------

    def update(
        self,
        fatigue_risk,
        timestamp,
    ):
        """
        Add a new fatigue-risk observation.

        Sampling is controlled so that extremely high FPS
        does not fill the history with duplicate observations.
        """

        fatigue_risk = self.clamp(
            fatigue_risk
        )

        if self.last_update_time is not None:

            elapsed = (
                timestamp
                - self.last_update_time
            )

            if elapsed < self.sample_interval:

                return self.get_state()

        self.history.append(
            (
                timestamp,
                fatigue_risk,
            )
        )

        self.last_update_time = timestamp

        return self.get_state()

    # ---------------------------------------------------------
    # Window values
    # ---------------------------------------------------------

    def _values_from_window(
        self,
        seconds,
    ):
        """Return risk values from the recent time window."""

        if not self.history:

            return []

        newest_timestamp = (
            self.history[-1][0]
        )

        cutoff = (
            newest_timestamp
            - seconds
        )

        return [
            risk
            for timestamp, risk
            in self.history
            if timestamp >= cutoff
        ]

    # ---------------------------------------------------------
    # Average
    # ---------------------------------------------------------

    @staticmethod
    def _average(values):
        """Calculate average safely."""

        if not values:

            return 0.0

        return sum(values) / len(values)

    # ---------------------------------------------------------
    # Trend
    # ---------------------------------------------------------

    def _calculate_trend(self):
        """
        Estimate short-term fatigue trend.

        Positive:
            fatigue increasing

        Negative:
            fatigue decreasing
        """

        if len(self.history) < 2:

            return 0.0

        points = list(
            self.history
        )

        first_time, first_risk = (
            points[0]
        )

        last_time, last_risk = (
            points[-1]
        )

        duration = (
            last_time
            - first_time
        )

        if duration <= 0:

            return 0.0

        trend = (
            last_risk
            - first_risk
        ) / duration

        return trend

    # ---------------------------------------------------------
    # Short-term trend
    # ---------------------------------------------------------

    def _calculate_short_trend(self):
        """Calculate trend using the recent short window."""

        values = (
            self._values_from_window(
                self.short_window_seconds
            )
        )

        if len(values) < 2:

            return 0.0

        return (
            values[-1]
            - values[0]
        ) / max(
            self.short_window_seconds,
            1e-6,
        )

    # ---------------------------------------------------------
    # Acceleration
    # ---------------------------------------------------------

    def _calculate_acceleration(
        self,
        short_trend,
    ):
        """
        Compare current trend with previous trend.

        Positive acceleration means fatigue is increasing
        faster than before.
        """

        if (
            self.previous_short_average
            is None
        ):

            return 0.0

        acceleration = (
            short_trend
            - self.previous_short_average
        )

        return acceleration

    # ---------------------------------------------------------
    # State classification
    # ---------------------------------------------------------

    def _classify_state(
        self,
        current_risk,
        short_average,
        medium_average,
        trend,
        acceleration,
    ):

        # Critical/high current risk.
        if current_risk >= 0.80:

            return "CRITICAL"

        if current_risk >= self.high_risk_threshold:

            return "HIGH"

        # Strong increasing trend.
        if (
            trend
            >= self.increasing_threshold
            and short_average
            > medium_average
        ):

            return "INCREASING"

        # Moderate increasing trend.
        if (
            acceleration
            >= self.increasing_threshold
            and short_average
            > medium_average
        ):

            return "RISING"

        # Sustained moderate risk.
        if medium_average >= 0.40:

            return "PERSISTENT"

        return "STABLE"

    # ---------------------------------------------------------
    # Main state calculation
    # ---------------------------------------------------------

    def get_state(self):
        """Return the current temporal fatigue state."""

        if not self.history:

            return {
                "current_risk": 0.0,
                "short_average": 0.0,
                "medium_average": 0.0,
                "long_average": 0.0,
                "trend": 0.0,
                "acceleration": 0.0,
                "state": "INITIALIZING",
                "history_samples": 0,
            }

        current_risk = (
            self.history[-1][1]
        )

        short_values = (
            self._values_from_window(
                self.short_window_seconds
            )
        )

        medium_values = (
            self._values_from_window(
                self.medium_window_seconds
            )
        )

        long_values = (
            self._values_from_window(
                self.history_seconds
            )
        )

        short_average = self._average(
            short_values
        )

        medium_average = self._average(
            medium_values
        )

        long_average = self._average(
            long_values
        )

        trend = (
            self._calculate_trend()
        )

        short_trend = (
            self._calculate_short_trend()
        )

        acceleration = (
            self._calculate_acceleration(
                short_trend
            )
        )

        state = self._classify_state(
            current_risk,
            short_average,
            medium_average,
            trend,
            acceleration,
        )

        self.previous_short_average = (
            short_trend
        )

        return {
            "current_risk":
                current_risk,

            "short_average":
                short_average,

            "medium_average":
                medium_average,

            "long_average":
                long_average,

            "trend":
                trend,

            "acceleration":
                acceleration,

            "state":
                state,

            "history_samples":
                len(self.history),
        }
    