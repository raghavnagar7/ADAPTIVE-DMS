"""
Adaptive Multimodal Fusion for ADAPTIVE-DMS.

v0.6

Combines:
    - EAR
    - PERCLOS
    - Blink
    - Microsleep
    - MAR / Yawning
    - Head Pose
    - Gaze

using reliability-aware adaptive weighting.

Output:
    - Individual signal risk
    - Adaptive weights
    - Overall fatigue risk
    - Risk level
"""


class AdaptiveMultimodalFusion:
    """
    Reliability-aware adaptive multimodal fusion.

    Each signal has:
        1. A base importance weight.
        2. A current reliability value.
        3. A calculated risk value.

    Effective weight:

        effective_weight =
            base_weight * reliability

    The final risk is the weighted average of
    the individual signal risks.
    """

    def __init__(self):

        # -----------------------------------------------------
        # Base importance weights
        # -----------------------------------------------------

        self.base_weights = {
            "EAR": 0.20,
            "PERCLOS": 0.20,
            "Blink": 0.08,
            "Microsleep": 0.18,
            "MAR": 0.08,
            "HeadPose": 0.13,
            "Gaze": 0.13,
        }

        # Check that weights sum to 1.
        total = sum(
            self.base_weights.values()
        )

        if abs(total - 1.0) > 0.001:

            raise ValueError(
                "Fusion base weights must sum to 1.0"
            )

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    @staticmethod
    def clamp(
        value,
        minimum=0.0,
        maximum=1.0,
    ):
        """Limit value to [0, 1]."""

        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    # ---------------------------------------------------------
    # EAR risk
    # ---------------------------------------------------------

    def ear_risk(
        self,
        ear,
    ):
        """
        Convert EAR into drowsiness risk.

        Higher risk when EAR becomes low.
        """

        if ear >= 0.30:

            return 0.0

        if ear <= 0.15:

            return 1.0

        return self.clamp(
            (0.30 - ear)
            / (0.30 - 0.15)
        )

    # ---------------------------------------------------------
    # PERCLOS risk
    # ---------------------------------------------------------

    def perclos_risk(
        self,
        perclos,
    ):
        """
        Convert PERCLOS into fatigue risk.
        """

        if perclos <= 0.10:

            return 0.0

        if perclos >= 0.50:

            return 1.0

        return self.clamp(
            (perclos - 0.10)
            / (0.50 - 0.10)
        )

    # ---------------------------------------------------------
    # Blink risk
    # ---------------------------------------------------------

    def blink_risk(
        self,
        blink_duration,
    ):
        """
        Longer blink duration can indicate
        increasing fatigue.

        Normal short blink:
            low risk

        Prolonged blink:
            high risk
        """

        if blink_duration <= 0.20:

            return 0.0

        if blink_duration >= 0.80:

            return 1.0

        return self.clamp(
            (blink_duration - 0.20)
            / (0.80 - 0.20)
        )

    # ---------------------------------------------------------
    # Microsleep risk
    # ---------------------------------------------------------

    def microsleep_risk(
        self,
        microsleep,
        duration,
    ):
        """
        Microsleep is treated as a strong fatigue signal.
        """

        if microsleep:

            if duration >= 3.0:

                return 1.0

            return self.clamp(
                duration / 3.0
            )

        return 0.0

    # ---------------------------------------------------------
    # MAR / Yawning risk
    # ---------------------------------------------------------

    def mar_risk(
        self,
        mar,
        yawning,
    ):
        """
        Convert mouth opening/yawning into fatigue risk.
        """

        if yawning:

            return 1.0

        if mar <= 0.40:

            return 0.0

        if mar >= 0.80:

            return 1.0

        return self.clamp(
            (mar - 0.40)
            / (0.80 - 0.40)
        )

    # ---------------------------------------------------------
    # Head pose risk
    # ---------------------------------------------------------

    def head_pose_risk(
        self,
        pitch,
        yaw,
        roll,
    ):
        """
        Estimate attention risk from extreme
        head orientation.

        This is NOT a clinical drowsiness measure.
        It represents possible driver distraction/
        abnormal orientation.
        """

        pitch_risk = self.clamp(
            abs(pitch) / 45.0
        )

        yaw_risk = self.clamp(
            abs(yaw) / 60.0
        )

        roll_risk = self.clamp(
            abs(roll) / 45.0
        )

        return self.clamp(
            (
                pitch_risk
                + yaw_risk
                + roll_risk
            )
            / 3.0
        )

    # ---------------------------------------------------------
    # Gaze risk
    # ---------------------------------------------------------

    def gaze_risk(
        self,
        gaze_direction,
        gaze_away_duration,
    ):
        """
        Estimate attention risk from prolonged
        gaze away from the forward direction.
        """

        if gaze_direction == "CENTER":

            return 0.0

        if gaze_away_duration <= 0.5:

            return 0.20

        if gaze_away_duration >= 3.0:

            return 1.0

        return self.clamp(
            gaze_away_duration / 3.0
        )

    # ---------------------------------------------------------
    # Adaptive weights
    # ---------------------------------------------------------

    def calculate_adaptive_weights(
        self,
        reliability_vector,
    ):
        """
        Calculate effective weights.

        effective_weight =
            base_weight * reliability

        Then normalize all effective weights
        so that their sum is 1.
        """

        effective_weights = {}

        for signal, base_weight in (
            self.base_weights.items()
        ):

            reliability = self.clamp(
                reliability_vector.get(
                    signal,
                    0.0,
                )
            )

            effective_weights[signal] = (
                base_weight
                * reliability
            )

        total_weight = sum(
            effective_weights.values()
        )

        if total_weight <= 1e-8:

            # No reliable signals.
            return {
                signal: 0.0
                for signal
                in self.base_weights
            }

        normalized_weights = {}

        for signal, weight in (
            effective_weights.items()
        ):

            normalized_weights[signal] = (
                weight
                / total_weight
            )

        return normalized_weights

    # ---------------------------------------------------------
    # Risk level
    # ---------------------------------------------------------

    def risk_level(
        self,
        risk,
    ):
        """Convert numerical risk into a level."""

        if risk < 0.20:

            return "NORMAL"

        if risk < 0.40:

            return "LOW"

        if risk < 0.60:

            return "MODERATE"

        if risk < 0.80:

            return "HIGH"

        return "CRITICAL"

    # ---------------------------------------------------------
    # Main fusion
    # ---------------------------------------------------------

    def calculate(
        self,
        driver_values,
        pose_values,
        gaze_values,
        reliability_values,
    ):
        """
        Calculate adaptive multimodal fatigue risk.
        """

        # -----------------------------------------------------
        # Individual signal risks
        # -----------------------------------------------------

        risks = {

            "EAR":
                self.ear_risk(
                    driver_values["ear"]
                ),

            "PERCLOS":
                self.perclos_risk(
                    driver_values["perclos"]
                ),

            "Blink":
                self.blink_risk(
                    driver_values[
                        "blink_duration"
                    ]
                ),

            "Microsleep":
                self.microsleep_risk(
                    driver_values[
                        "microsleep"
                    ],
                    driver_values[
                        "microsleep_duration"
                    ],
                ),

            "MAR":
                self.mar_risk(
                    driver_values["mar"],
                    driver_values["yawning"],
                ),

            "HeadPose":
                self.head_pose_risk(
                    pose_values["pitch"],
                    pose_values["yaw"],
                    pose_values["roll"],
                ),

            "Gaze":
                self.gaze_risk(
                    gaze_values[
                        "gaze_direction"
                    ],
                    gaze_values[
                        "gaze_away_duration"
                    ],
                ),
        }

        # -----------------------------------------------------
        # Reliability vector
        # -----------------------------------------------------

        reliability_vector = (
            reliability_values[
                "reliability_vector"
            ]
        )

        # -----------------------------------------------------
        # Adaptive weights
        # -----------------------------------------------------

        adaptive_weights = (
            self.calculate_adaptive_weights(
                reliability_vector
            )
        )

        # -----------------------------------------------------
        # Weighted fusion
        # -----------------------------------------------------

        weighted_contributions = {}

        for signal in self.base_weights:

            weighted_contributions[signal] = (
                risks[signal]
                * adaptive_weights[signal]
            )

        fatigue_risk = sum(
            weighted_contributions.values()
        )

        fatigue_risk = self.clamp(
            fatigue_risk
        )

        # -----------------------------------------------------
        # Risk level
        # -----------------------------------------------------

        level = self.risk_level(
            fatigue_risk
        )

        return {
            "signal_risks": risks,

            "adaptive_weights":
                adaptive_weights,

            "weighted_contributions":
                weighted_contributions,

            "fatigue_risk":
                fatigue_risk,

            "risk_level":
                level,
        }