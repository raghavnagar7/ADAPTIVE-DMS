"""
=============================================================
ADAPTIVE-DMS
=============================================================

Steering Behaviour Analysis Module

Version:
    v1.0 - Step 7A

Purpose:
    Analyze steering behaviour and detect irregular steering
    patterns.

Important:
    This module does NOT claim to measure a real vehicle
    steering-wheel angle from the webcam.

    It accepts steering-angle input from a simulator,
    controller, keyboard interface, or vehicle interface.

Outputs:
    - steering_angle
    - steering_change
    - steering_rate
    - steering_variability
    - sudden_correction
    - irregularity_score
    - reliability
    - driving_state

=============================================================
"""

from collections import deque
import math
import statistics


class SteeringBehaviourAnalyzer:
    """
    Analyze temporal steering behaviour.

    Steering angle convention:

        negative = left
        zero     = center
        positive = right
    """

    def __init__(
        self,
        history_size=50,
        maximum_steering_angle=450.0,
        sudden_change_threshold=45.0,
        high_rate_threshold=180.0,
        irregularity_threshold=0.55,
    ):
        self.history_size = int(history_size)

        self.maximum_steering_angle = float(
            maximum_steering_angle
        )

        self.sudden_change_threshold = float(
            sudden_change_threshold
        )

        self.high_rate_threshold = float(
            high_rate_threshold
        )

        self.irregularity_threshold = float(
            irregularity_threshold
        )

        self.angle_history = deque(
            maxlen=self.history_size
        )

        self.time_history = deque(
            maxlen=self.history_size
        )

        self.previous_angle = 0.0
        self.previous_timestamp = None

        self.last_result = {
            "steering_angle": 0.0,
            "steering_change": 0.0,
            "steering_rate": 0.0,
            "steering_variability": 0.0,
            "sudden_correction": False,
            "irregularity_score": 0.0,
            "reliability": 0.0,
            "driving_state": "UNKNOWN",
            "sample_count": 0,
        }

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):
        try:

            result = float(value)

            if not math.isfinite(result):
                return default

            return result

        except (
            TypeError,
            ValueError,
        ):

            return default

    # =========================================================
    # CLAMP
    # =========================================================

    @staticmethod
    def _clamp(
        value,
        minimum=0.0,
        maximum=1.0,
    ):
        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    # =========================================================
    # NORMALIZE STEERING
    # =========================================================

    def _normalize_angle(
        self,
        angle,
    ):
        """
        Convert steering angle into [-1, 1].
        """

        return self._clamp(
            abs(angle)
            / max(
                1.0,
                self.maximum_steering_angle,
            ),
            0.0,
            1.0,
        )

    # =========================================================
    # VARIABILITY
    # =========================================================

    def _calculate_variability(self):
        """
        Calculate normalized steering variability.

        Uses standard deviation of recent steering angles.
        """

        if len(self.angle_history) < 3:
            return 0.0

        values = list(
            self.angle_history
        )

        try:

            standard_deviation = (
                statistics.stdev(values)
            )

        except statistics.StatisticsError:

            return 0.0

        variability = (
            standard_deviation
            / max(
                1.0,
                self.maximum_steering_angle,
            )
        )

        return self._clamp(
            variability,
            0.0,
            1.0,
        )

    # =========================================================
    # SUDDEN CORRECTION
    # =========================================================

    def _detect_sudden_correction(
        self,
        steering_change,
    ):
        """
        Detect a large steering change.
        """

        return (
            abs(steering_change)
            >= self.sudden_change_threshold
        )

    # =========================================================
    # IRREGULARITY
    # =========================================================

    def _calculate_irregularity(
        self,
        steering_change,
        steering_rate,
        variability,
        sudden_correction,
    ):
        """
        Calculate overall steering irregularity.

        Higher value means more irregular steering behaviour.
        """

        # -----------------------------------------------------
        # Change component
        # -----------------------------------------------------

        change_score = self._clamp(
            abs(steering_change)
            / max(
                1.0,
                self.sudden_change_threshold,
            )
        )

        # -----------------------------------------------------
        # Steering rate component
        # -----------------------------------------------------

        rate_score = self._clamp(
            abs(steering_rate)
            / max(
                1.0,
                self.high_rate_threshold,
            )
        )

        # -----------------------------------------------------
        # Sudden correction component
        # -----------------------------------------------------

        sudden_score = (
            1.0
            if sudden_correction
            else 0.0
        )

        # -----------------------------------------------------
        # Weighted score
        # -----------------------------------------------------

        irregularity = (
            variability * 0.40
            + change_score * 0.20
            + rate_score * 0.20
            + sudden_score * 0.20
        )

        return self._clamp(
            irregularity
        )

    # =========================================================
    # DRIVING STATE
    # =========================================================

    def _classify_state(
        self,
        irregularity_score,
    ):
        if irregularity_score < 0.20:
            return "STABLE"

        if irregularity_score < 0.40:
            return "MILD_VARIATION"

        if irregularity_score < 0.55:
            return "IRREGULAR"

        if irregularity_score < 0.75:
            return "HIGH_IRREGULARITY"

        return "CRITICAL_IRREGULARITY"

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        steering_angle=0.0,
        timestamp=None,
        input_available=True,
    ):
        """
        Process one steering sample.

        Parameters
        ----------
        steering_angle:
            Steering-wheel angle.

        timestamp:
            Current timestamp in seconds.

        input_available:
            Whether a valid steering source is currently
            available.
        """

        angle = self._safe_float(
            steering_angle
        )

        # -----------------------------------------------------
        # Clamp physically unreasonable input
        # -----------------------------------------------------

        angle = max(
            -self.maximum_steering_angle,
            min(
                self.maximum_steering_angle,
                angle,
            ),
        )

        # -----------------------------------------------------
        # Timestamp
        # -----------------------------------------------------

        if timestamp is None:

            import time

            timestamp = time.time()

        timestamp = self._safe_float(
            timestamp
        )

        # -----------------------------------------------------
        # Steering change
        # -----------------------------------------------------

        steering_change = (
            angle
            - self.previous_angle
        )

        # -----------------------------------------------------
        # Time difference
        # -----------------------------------------------------

        if self.previous_timestamp is None:

            delta_time = 0.0

        else:

            delta_time = (
                timestamp
                - self.previous_timestamp
            )

        # -----------------------------------------------------
        # Steering rate
        # -----------------------------------------------------

        if delta_time > 0:

            steering_rate = (
                steering_change
                / delta_time
            )

        else:

            steering_rate = 0.0

        # -----------------------------------------------------
        # Store history
        # -----------------------------------------------------

        self.angle_history.append(
            angle
        )

        self.time_history.append(
            timestamp
        )

        # -----------------------------------------------------
        # Variability
        # -----------------------------------------------------

        variability = (
            self._calculate_variability()
        )

        # -----------------------------------------------------
        # Sudden correction
        # -----------------------------------------------------

        sudden_correction = (
            self._detect_sudden_correction(
                steering_change
            )
        )

        # -----------------------------------------------------
        # Irregularity
        # -----------------------------------------------------

        irregularity_score = (
            self._calculate_irregularity(
                steering_change=steering_change,
                steering_rate=steering_rate,
                variability=variability,
                sudden_correction=sudden_correction,
            )
        )

        # -----------------------------------------------------
        # State
        # -----------------------------------------------------

        driving_state = (
            self._classify_state(
                irregularity_score
            )
        )

        # -----------------------------------------------------
        # Reliability
        # -----------------------------------------------------

        if not input_available:

            reliability = 0.0
            driving_state = "NO_INPUT"

        else:

            # More samples provide greater confidence.
            history_reliability = self._clamp(
                len(self.angle_history)
                / 10.0
            )

            reliability = (
                0.30
                + 0.70
                * history_reliability
            )

            reliability = self._clamp(
                reliability
            )

        # -----------------------------------------------------
        # Save state
        # -----------------------------------------------------

        self.previous_angle = angle
        self.previous_timestamp = timestamp

        self.last_result = {

            "steering_angle": float(
                angle
            ),

            "steering_change": float(
                steering_change
            ),

            "steering_rate": float(
                steering_rate
            ),

            "steering_variability": float(
                variability
            ),

            "sudden_correction": bool(
                sudden_correction
            ),

            "irregularity_score": float(
                irregularity_score
            ),

            "reliability": float(
                reliability
            ),

            "driving_state": (
                driving_state
            ),

            "sample_count": len(
                self.angle_history
            ),
        }

        return dict(
            self.last_result
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):
        """
        Reset steering history.
        """

        self.angle_history.clear()

        self.time_history.clear()

        self.previous_angle = 0.0

        self.previous_timestamp = None

        self.last_result = {
            "steering_angle": 0.0,
            "steering_change": 0.0,
            "steering_rate": 0.0,
            "steering_variability": 0.0,
            "sudden_correction": False,
            "irregularity_score": 0.0,
            "reliability": 0.0,
            "driving_state": "UNKNOWN",
            "sample_count": 0,
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):

        return {
            "history_size": self.history_size,
            "samples": len(
                self.angle_history
            ),
            "previous_angle": (
                self.previous_angle
            ),
            "last_result": dict(
                self.last_result
            ),
        }


# =============================================================
# SELF TEST
# =============================================================

def self_test():

    print("=" * 65)

    print("ADAPTIVE-DMS")

    print("STEERING BEHAVIOUR ANALYZER")

    print("v1.0 - STEP 7A SELF TEST")

    print("=" * 65)

    analyzer = (
        SteeringBehaviourAnalyzer(
            history_size=20,
            maximum_steering_angle=450.0,
            sudden_change_threshold=45.0,
            high_rate_threshold=180.0,
            irregularity_threshold=0.55,
        )
    )

    print()

    print("Simulating steering input...")

    print()

    # ---------------------------------------------------------
    # Stable steering
    # ---------------------------------------------------------

    stable_angles = [
        0,
        2,
        -1,
        1,
        0,
        2,
        -2,
        1,
        0,
        -1,
    ]

    timestamp = 0.0

    result = None

    for angle in stable_angles:

        timestamp += 0.5

        result = analyzer.update(
            steering_angle=angle,
            timestamp=timestamp,
            input_available=True,
        )

    print("Stable steering:")

    print(
        f"  Angle: "
        f"{result['steering_angle']:.2f}"
    )

    print(
        f"  Change: "
        f"{result['steering_change']:.2f}"
    )

    print(
        f"  Rate: "
        f"{result['steering_rate']:.2f}"
    )

    print(
        f"  Variability: "
        f"{result['steering_variability']:.3f}"
    )

    print(
        f"  Irregularity: "
        f"{result['irregularity_score']:.3f}"
    )

    print(
        f"  State: "
        f"{result['driving_state']}"
    )

    print(
        f"  Reliability: "
        f"{result['reliability']:.3f}"
    )

    print()

    # ---------------------------------------------------------
    # Irregular steering
    # ---------------------------------------------------------

    irregular_angles = [
        20,
        -30,
        40,
        -50,
        60,
        -70,
        80,
        -90,
        100,
        -110,
    ]

    for angle in irregular_angles:

        timestamp += 0.2

        result = analyzer.update(
            steering_angle=angle,
            timestamp=timestamp,
            input_available=True,
        )

    print("Irregular steering:")

    print(
        f"  Angle: "
        f"{result['steering_angle']:.2f}"
    )

    print(
        f"  Change: "
        f"{result['steering_change']:.2f}"
    )

    print(
        f"  Rate: "
        f"{result['steering_rate']:.2f}"
    )

    print(
        f"  Variability: "
        f"{result['steering_variability']:.3f}"
    )

    print(
        f"  Sudden correction: "
        f"{result['sudden_correction']}"
    )

    print(
        f"  Irregularity: "
        f"{result['irregularity_score']:.3f}"
    )

    print(
        f"  State: "
        f"{result['driving_state']}"
    )

    print(
        f"  Reliability: "
        f"{result['reliability']:.3f}"
    )

    print()

    # ---------------------------------------------------------
    # No input test
    # ---------------------------------------------------------

    result = analyzer.update(
        steering_angle=0.0,
        timestamp=timestamp + 0.5,
        input_available=False,
    )

    print("No steering input:")

    print(
        f"  State: "
        f"{result['driving_state']}"
    )

    print(
        f"  Reliability: "
        f"{result['reliability']:.3f}"
    )

    print()

    print("STEP 7A SELF TEST COMPLETE")

    print("=" * 65)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()