"""
Adaptive Multimodal Fusion for ADAPTIVE-DMS.

v0.8

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

    def __init__(self):

        # Base importance weights
        self.base_weights = {
            "EAR": 0.20,
            "PERCLOS": 0.20,
            "Blink": 0.08,
            "Microsleep": 0.18,
            "MAR": 0.08,
            "HeadPose": 0.13,
            "Gaze": 0.13,
        }

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

        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    # ---------------------------------------------------------
    # EAR RISK
    # ---------------------------------------------------------

    def ear_risk(self, ear):

        if ear >= 0.30:
            return 0.0

        if ear <= 0.15:
            return 1.0

        return self.clamp(
            (0.30 - ear)
            / (0.30 - 0.15)
        )

    # ---------------------------------------------------------
    # PERCLOS RISK
    # ---------------------------------------------------------

    def perclos_risk(self, perclos):

        if perclos <= 0.10:
            return 0.0

        if perclos >= 0.50:
            return 1.0

        return self.clamp(
            (perclos - 0.10)
            / (0.50 - 0.10)
        )

    # ---------------------------------------------------------
    # BLINK RISK
    # ---------------------------------------------------------

    def blink_risk(
        self,
        blink_duration,
    ):

        if blink_duration <= 0.20:
            return 0.0

        if blink_duration >= 0.80:
            return 1.0

        return self.clamp(
            (blink_duration - 0.20)
            / (0.80 - 0.20)
        )

    # ---------------------------------------------------------
    # MICROSLEEP RISK
    # ---------------------------------------------------------

    def microsleep_risk(
        self,
        microsleep,
        duration,
    ):

        if microsleep:

            if duration >= 3.0:
                return 1.0

            return self.clamp(
                duration / 3.0
            )

        return 0.0

    # ---------------------------------------------------------
    # MAR / YAWNING RISK
    # ---------------------------------------------------------

    def mar_risk(
        self,
        mar,
        yawning,
    ):

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
    # HEAD POSE RISK
    # ---------------------------------------------------------

    def head_pose_risk(
        self,
        pitch,
        yaw,
        roll,
    ):

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
            ) / 3.0
        )

    # ---------------------------------------------------------
    # GAZE RISK
    # ---------------------------------------------------------

    def gaze_risk(
        self,
        gaze_direction,
        gaze_away_duration,
    ):

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
    # ADAPTIVE WEIGHTS
    # ---------------------------------------------------------

    def calculate_adaptive_weights(
        self,
        reliability_vector,
    ):

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
    # CALIBRATED RISK LEVEL
    # ---------------------------------------------------------

    def risk_level(self, risk):

        # Calibrated thresholds
        #
        # 0.00 - 0.19 = NORMAL
        # 0.20 - 0.34 = LOW
        # 0.35 - 0.54 = MODERATE
        # 0.55 - 0.74 = HIGH
        # 0.75 - 1.00 = CRITICAL

        if risk < 0.20:
            return "NORMAL"

        if risk < 0.35:
            return "LOW"

        if risk < 0.55:
            return "MODERATE"

        if risk < 0.75:
            return "HIGH"

        return "CRITICAL"

    # ---------------------------------------------------------
    # MAIN FUSION
    # ---------------------------------------------------------

    def calculate(
        self,
        driver_values,
        pose_values,
        gaze_values,
        reliability_values,
    ):

        # -----------------------------------------------------
        # Individual risks
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
        # Reliability
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
        # Weighted contributions
        # -----------------------------------------------------

        weighted_contributions = {}

        for signal in self.base_weights:

            weighted_contributions[
                signal
            ] = (
                risks[signal]
                * adaptive_weights[signal]
            )

        # -----------------------------------------------------
        # Final fatigue risk
        # -----------------------------------------------------

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

            "signal_risks":
                risks,

            "adaptive_weights":
                adaptive_weights,

            "weighted_contributions":
                weighted_contributions,

            "fatigue_risk":
                fatigue_risk,

            "risk_level":
                level,
        }