"""
=============================================================
ADAPTIVE-DMS
=============================================================

Adaptive Multimodal Fusion

Version:
    v1.0 - STEP 8B

Combines:

    Facial:
        EAR
        PERCLOS
        Blink
        Microsleep
        MAR / Yawning
        Head Pose
        Gaze

    Behaviour:
        Steering

    Physiological:
        Heart Rate

Uses reliability-aware adaptive weighting.

Important:
    Heart rate is an experimental non-contact rPPG estimate.
    It is NOT a medical measurement.

Output:
    - Individual signal risks
    - Adaptive weights
    - Weighted contributions
    - Fatigue risk
    - Risk level

=============================================================
"""


class AdaptiveMultimodalFusion:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self):

        self.base_weights = {

            "EAR": 0.17,

            "PERCLOS": 0.17,

            "Blink": 0.07,

            "Microsleep": 0.15,

            "MAR": 0.07,

            "HeadPose": 0.11,

            "Gaze": 0.11,

            "Steering": 0.10,

            "HeartRate": 0.05,
        }

        total = sum(
            self.base_weights.values()
        )

        if abs(
            total - 1.0
        ) > 0.001:

            raise ValueError(
                "Fusion base weights must sum to 1.0"
            )

    # =========================================================
    # CLAMP
    # =========================================================

    @staticmethod
    def clamp(
        value,
        minimum=0.0,
        maximum=1.0,
    ):

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            value = minimum

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # =========================================================
    # EAR RISK
    # =========================================================

    def ear_risk(
        self,
        ear,
    ):

        if ear >= 0.30:

            return 0.0

        if ear <= 0.15:

            return 1.0

        return self.clamp(
            (
                0.30
                - ear
            )
            /
            (
                0.30
                - 0.15
            )
        )

    # =========================================================
    # PERCLOS
    # =========================================================

    def perclos_risk(
        self,
        perclos,
    ):

        if perclos <= 0.10:

            return 0.0

        if perclos >= 0.50:

            return 1.0

        return self.clamp(
            (
                perclos
                - 0.10
            )
            /
            (
                0.50
                - 0.10
            )
        )

    # =========================================================
    # BLINK
    # =========================================================

    def blink_risk(
        self,
        blink_duration,
    ):

        if blink_duration <= 0.20:

            return 0.0

        if blink_duration >= 0.80:

            return 1.0

        return self.clamp(
            (
                blink_duration
                - 0.20
            )
            /
            (
                0.80
                - 0.20
            )
        )

    # =========================================================
    # MICROSLEEP
    # =========================================================

    def microsleep_risk(
        self,
        microsleep,
        duration,
    ):

        if not microsleep:

            return 0.0

        if duration >= 3.0:

            return 1.0

        return self.clamp(
            duration
            /
            3.0
        )

    # =========================================================
    # MAR / YAWNING
    # =========================================================

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
            (
                mar
                - 0.40
            )
            /
            (
                0.80
                - 0.40
            )
        )

    # =========================================================
    # HEAD POSE
    # =========================================================

    def head_pose_risk(
        self,
        pitch,
        yaw,
        roll,
    ):

        pitch_risk = self.clamp(
            abs(
                pitch
            )
            /
            45.0
        )

        yaw_risk = self.clamp(
            abs(
                yaw
            )
            /
            60.0
        )

        roll_risk = self.clamp(
            abs(
                roll
            )
            /
            45.0
        )

        return self.clamp(
            (
                pitch_risk
                +
                yaw_risk
                +
                roll_risk
            )
            /
            3.0
        )

    # =========================================================
    # GAZE
    # =========================================================

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
            gaze_away_duration
            /
            3.0
        )

    # =========================================================
    # STEERING RISK
    # =========================================================

    def steering_risk(
        self,
        steering_values,
    ):

        if not steering_values:

            return 0.0

        irregularity = self.clamp(
            steering_values.get(
                "irregularity_score",
                0.0,
            )
        )

        sudden_correction = (
            steering_values.get(
                "sudden_correction",
                False,
            )
        )

        if sudden_correction:

            irregularity = max(
                irregularity,
                0.65,
            )

        return self.clamp(
            irregularity
        )

    # =========================================================
    # HEART RATE RISK
    # =========================================================

    def heart_rate_risk(
        self,
        heart_rate_values,
    ):

        if not heart_rate_values:

            return 0.0

        bpm = self.clamp(
            heart_rate_values.get(
                "heart_rate_bpm",
                0.0,
            ),
            0.0,
            250.0,
        )

        reliability = self.clamp(
            heart_rate_values.get(
                "reliability",
                0.0,
            )
        )

        state = heart_rate_values.get(
            "state",
            "NO_SIGNAL",
        )

        if reliability < 0.30:

            return 0.0

        if bpm <= 0:

            return 0.0

        # -----------------------------------------------------
        # Normal resting/driver range
        # -----------------------------------------------------

        if 50.0 <= bpm <= 100.0:

            base_risk = 0.0

        elif bpm < 50.0:

            base_risk = self.clamp(
                (
                    50.0 - bpm
                )
                /
                20.0
            )

        else:

            base_risk = self.clamp(
                (
                    bpm - 100.0
                )
                /
                50.0
            )

        # -----------------------------------------------------
        # Elevated/high state
        # -----------------------------------------------------

        if state == "HIGH":

            base_risk = max(
                base_risk,
                0.70,
            )

        elif state == "ELEVATED":

            base_risk = max(
                base_risk,
                0.30,
            )

        return self.clamp(
            base_risk
            * reliability
        )

    # =========================================================
    # ADAPTIVE WEIGHTS
    # =========================================================

    def calculate_adaptive_weights(
        self,
        reliability_vector,
    ):

        effective_weights = {}

        for (
            signal,
            base_weight,
        ) in self.base_weights.items():

            reliability = self.clamp(
                reliability_vector.get(
                    signal,
                    0.0,
                )
            )

            effective_weights[
                signal
            ] = (
                base_weight
                *
                reliability
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

        for (
            signal,
            weight,
        ) in effective_weights.items():

            normalized_weights[
                signal
            ] = (
                weight
                /
                total_weight
            )

        return normalized_weights

    # =========================================================
    # RISK LEVEL
    # =========================================================

    def risk_level(
        self,
        risk,
    ):

        if risk < 0.20:

            return "NORMAL"

        if risk < 0.35:

            return "LOW"

        if risk < 0.55:

            return "MODERATE"

        if risk < 0.75:

            return "HIGH"

        return "CRITICAL"

    # =========================================================
    # MAIN CALCULATION
    # =========================================================

    def calculate(
        self,
        driver_values,
        pose_values,
        gaze_values,
        reliability_values,
        steering_values=None,
        heart_rate_values=None,
    ):

        if steering_values is None:

            steering_values = {}

        if heart_rate_values is None:

            heart_rate_values = {}

        # =====================================================
        # SIGNAL RISKS
        # =====================================================

        risks = {

            "EAR":
                self.ear_risk(
                    driver_values.get(
                        "ear",
                        0.0,
                    )
                ),

            "PERCLOS":
                self.perclos_risk(
                    driver_values.get(
                        "perclos",
                        0.0,
                    )
                ),

            "Blink":
                self.blink_risk(
                    driver_values.get(
                        "blink_duration",
                        0.0,
                    )
                ),

            "Microsleep":
                self.microsleep_risk(
                    driver_values.get(
                        "microsleep",
                        False,
                    ),
                    driver_values.get(
                        "microsleep_duration",
                        0.0,
                    ),
                ),

            "MAR":
                self.mar_risk(
                    driver_values.get(
                        "mar",
                        0.0,
                    ),
                    driver_values.get(
                        "yawning",
                        False,
                    ),
                ),

            "HeadPose":
                self.head_pose_risk(
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
                ),

            "Gaze":
                self.gaze_risk(
                    gaze_values.get(
                        "gaze_direction",
                        "UNKNOWN",
                    ),
                    gaze_values.get(
                        "gaze_away_duration",
                        0.0,
                    ),
                ),

            "Steering":
                self.steering_risk(
                    steering_values
                ),

            "HeartRate":
                self.heart_rate_risk(
                    heart_rate_values
                ),
        }

        # =====================================================
        # RELIABILITY VECTOR
        # =====================================================

        reliability_vector = dict(
            reliability_values.get(
                "reliability_vector",
                {},
            )
        )

        # -----------------------------------------------------
        # Steering reliability
        # -----------------------------------------------------

        reliability_vector[
            "Steering"
        ] = self.clamp(
            steering_values.get(
                "reliability",
                0.0,
            )
        )

        # -----------------------------------------------------
        # Heart rate reliability
        # -----------------------------------------------------

        reliability_vector[
            "HeartRate"
        ] = self.clamp(
            heart_rate_values.get(
                "reliability",
                0.0,
            )
        )

        # =====================================================
        # ADAPTIVE WEIGHTS
        # =====================================================

        adaptive_weights = (
            self.calculate_adaptive_weights(
                reliability_vector
            )
        )

        # =====================================================
        # CONTRIBUTIONS
        # =====================================================

        weighted_contributions = {}

        for signal in (
            self.base_weights
        ):

            weighted_contributions[
                signal
            ] = (
                risks.get(
                    signal,
                    0.0,
                )
                *
                adaptive_weights.get(
                    signal,
                    0.0,
                )
            )

        # =====================================================
        # FINAL RISK
        # =====================================================

        fatigue_risk = self.clamp(
            sum(
                weighted_contributions.values()
            )
        )

        # =====================================================
        # RISK LEVEL
        # =====================================================

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

            "steering":
                dict(
                    steering_values
                ),

            "heart_rate":
                dict(
                    heart_rate_values
                ),
        }


# =============================================================
# SELF TEST
# =============================================================

def self_test():

    print("=" * 70)

    print(
        "ADAPTIVE-DMS"
    )

    print(
        "ADAPTIVE MULTIMODAL FUSION"
    )

    print(
        "v1.0 - STEP 8B"
    )

    print("=" * 70)

    fusion = (
        AdaptiveMultimodalFusion()
    )

    driver_values = {

        "ear": 0.28,

        "perclos": 0.12,

        "blink_duration": 0.20,

        "microsleep": False,

        "microsleep_duration": 0.0,

        "mar": 0.45,

        "yawning": False,
    }

    pose_values = {

        "pitch": 5.0,

        "yaw": 8.0,

        "roll": 3.0,
    }

    gaze_values = {

        "gaze_direction": "CENTER",

        "gaze_away_duration": 0.0,
    }

    reliability_values = {

        "reliability_vector": {

            "EAR": 1.0,

            "PERCLOS": 1.0,

            "Blink": 1.0,

            "Microsleep": 1.0,

            "MAR": 1.0,

            "HeadPose": 1.0,

            "Gaze": 1.0,
        }
    }

    steering_values = {

        "steering_angle": 40.0,

        "steering_change": 0.0,

        "steering_rate": 0.0,

        "steering_variability": 0.02,

        "sudden_correction": False,

        "irregularity_score": 0.10,

        "reliability": 1.0,

        "driving_state": "STABLE",
    }

    heart_rate_values = {

        "heart_rate_bpm": 78.0,

        "signal_quality": 0.85,

        "reliability": 0.90,

        "state": "NORMAL",

        "signal_ready": True,
    }

    result = fusion.calculate(

        driver_values=driver_values,

        pose_values=pose_values,

        gaze_values=gaze_values,

        reliability_values=reliability_values,

        steering_values=steering_values,

        heart_rate_values=heart_rate_values,
    )

    print()

    print(
        "SIGNAL RISKS"
    )

    for (
        signal,
        risk,
    ) in result[
        "signal_risks"
    ].items():

        print(
            f"  {signal:<12}: "
            f"{risk:.3f}"
        )

    print()

    print(
        "ADAPTIVE WEIGHTS"
    )

    for (
        signal,
        weight,
    ) in result[
        "adaptive_weights"
    ].items():

        print(
            f"  {signal:<12}: "
            f"{weight:.3f}"
        )

    print()

    print(
        "HEART RATE"
    )

    print(
        f"  BPM: "
        f"{heart_rate_values['heart_rate_bpm']:.1f}"
    )

    print(
        f"  Reliability: "
        f"{heart_rate_values['reliability']:.3f}"
    )

    print(
        f"  State: "
        f"{heart_rate_values['state']}"
    )

    print()

    print(
        "STEERING"
    )

    print(
        f"  Angle: "
        f"{steering_values['steering_angle']:.2f}"
    )

    print(
        f"  Irregularity: "
        f"{steering_values['irregularity_score']:.3f}"
    )

    print(
        f"  State: "
        f"{steering_values['driving_state']}"
    )

    print()

    print(
        f"FUSION FATIGUE RISK: "
        f"{result['fatigue_risk']:.3f}"
    )

    print(
        f"RISK LEVEL: "
        f"{result['risk_level']}"
    )

    print()

    print(
        "STEP 8B FUSION SELF TEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()