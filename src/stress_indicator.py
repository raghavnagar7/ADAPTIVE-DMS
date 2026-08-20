"""
=============================================================
ADAPTIVE-DMS
=============================================================

NON-MEDICAL DRIVER STRESS INDICATOR

Version:
    v1.0 - STEP 8F

Purpose:
    Estimate an experimental driver stress indicator using
    already available multimodal signals.

Signals:
    - Respiration
    - Respiration reliability
    - Heart rate
    - Heart-rate reliability
    - Head pose
    - Gaze
    - Steering irregularity

Important:
    This is NOT a medical diagnosis or clinical stress
    measurement.

Outputs:
    - stress_score
    - stress_level
    - reliability
    - signal_contributions

=============================================================
"""


class DriverStressIndicator:

    def __init__(self):

        self.base_weights = {

            "respiration": 0.30,

            "heart_rate": 0.20,

            "head_pose": 0.15,

            "gaze": 0.15,

            "steering": 0.20,
        }

        self.last_result = {

            "stress_score": 0.0,

            "stress_level": "NORMAL",

            "reliability": 0.0,

            "signal_contributions": {},
        }

    # =========================================================
    # CLAMP
    # =========================================================

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

    # =========================================================
    # RESPIRATION STRESS
    # =========================================================

    def respiration_stress(
        self,
        respiration_rate,
    ):

        rate = float(
            respiration_rate
        )

        if rate <= 12.0:

            return 0.0

        if rate >= 30.0:

            return 1.0

        return self.clamp(
            (
                rate
                - 12.0
            )
            /
            (
                30.0
                - 12.0
            )
        )

    # =========================================================
    # HEART RATE STRESS
    # =========================================================

    def heart_rate_stress(
        self,
        heart_rate,
    ):

        rate = float(
            heart_rate
        )

        if rate <= 60.0:

            return 0.0

        if rate >= 120.0:

            return 1.0

        return self.clamp(
            (
                rate
                - 60.0
            )
            /
            (
                120.0
                - 60.0
            )
        )

    # =========================================================
    # HEAD POSE STRESS
    # =========================================================

    def head_pose_stress(
        self,
        pitch,
        yaw,
        roll,
    ):

        pitch_score = self.clamp(
            abs(
                float(pitch)
            )
            /
            45.0
        )

        yaw_score = self.clamp(
            abs(
                float(yaw)
            )
            /
            60.0
        )

        roll_score = self.clamp(
            abs(
                float(roll)
            )
            /
            45.0
        )

        return self.clamp(
            (
                pitch_score
                +
                yaw_score
                +
                roll_score
            )
            /
            3.0
        )

    # =========================================================
    # GAZE STRESS
    # =========================================================

    def gaze_stress(
        self,
        gaze_direction,
        gaze_away_duration,
    ):

        if (
            str(
                gaze_direction
            ).upper()
            ==
            "CENTER"
        ):

            return 0.0

        duration = float(
            gaze_away_duration
        )

        if duration <= 0.5:

            return 0.20

        if duration >= 3.0:

            return 1.0

        return self.clamp(
            duration
            /
            3.0
        )

    # =========================================================
    # MAIN CALCULATION
    # =========================================================

    def calculate(
        self,
        respiration_values=None,
        heart_rate_values=None,
        pose_values=None,
        gaze_values=None,
        steering_values=None,
    ):

        respiration_values = (
            respiration_values
            or {}
        )

        heart_rate_values = (
            heart_rate_values
            or {}
        )

        pose_values = (
            pose_values
            or {}
        )

        gaze_values = (
            gaze_values
            or {}
        )

        steering_values = (
            steering_values
            or {}
        )

        # -----------------------------------------------------
        # Individual stress signals
        # -----------------------------------------------------

        respiration_rate = float(
            respiration_values.get(
                "respiration_rate_bpm",
                0.0,
            )
        )

        respiration_reliability = self.clamp(
            respiration_values.get(
                "reliability",
                0.0,
            )
        )

        heart_rate = float(
            heart_rate_values.get(
                "heart_rate_bpm",
                0.0,
            )
        )

        heart_rate_reliability = self.clamp(
            heart_rate_values.get(
                "reliability",
                0.0,
            )
        )

        head_pose_score = (
            self.head_pose_stress(
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
            )
        )

        gaze_score = (
            self.gaze_stress(
                gaze_values.get(
                    "gaze_direction",
                    "CENTER",
                ),
                gaze_values.get(
                    "gaze_away_duration",
                    0.0,
                ),
            )
        )

        steering_score = self.clamp(
            steering_values.get(
                "irregularity_score",
                0.0,
            )
        )

        respiration_score = (
            self.respiration_stress(
                respiration_rate
            )
        )

        heart_rate_score = (
            self.heart_rate_stress(
                heart_rate
            )
        )

        # -----------------------------------------------------
        # Reliability-aware weighting
        # -----------------------------------------------------

        effective_weights = {

            "respiration":
                self.base_weights[
                    "respiration"
                ]
                *
                respiration_reliability,

            "heart_rate":
                self.base_weights[
                    "heart_rate"
                ]
                *
                heart_rate_reliability,

            "head_pose":
                self.base_weights[
                    "head_pose"
                ],

            "gaze":
                self.base_weights[
                    "gaze"
                ],

            "steering":
                self.base_weights[
                    "steering"
                ],
        }

        total_weight = sum(
            effective_weights.values()
        )

        if total_weight <= 0:

            self.last_result = {

                "stress_score":
                    0.0,

                "stress_level":
                    "NO_SIGNAL",

                "reliability":
                    0.0,

                "signal_contributions":
                    {},
            }

            return dict(
                self.last_result
            )

        normalized_weights = {

            key:
                value
                /
                total_weight

            for key, value
            in effective_weights.items()
        }

        signal_scores = {

            "respiration":
                respiration_score,

            "heart_rate":
                heart_rate_score,

            "head_pose":
                head_pose_score,

            "gaze":
                gaze_score,

            "steering":
                steering_score,
        }

        # -----------------------------------------------------
        # Weighted stress score
        # -----------------------------------------------------

        contributions = {

            key:
                signal_scores[key]
                *
                normalized_weights[key]

            for key
            in normalized_weights
        }

        stress_score = self.clamp(
            sum(
                contributions.values()
            )
        )

        # -----------------------------------------------------
        # Reliability
        # -----------------------------------------------------

        reliability = self.clamp(
            (
                respiration_reliability
                +
                heart_rate_reliability
                +
                1.0
                +
                1.0
                +
                self.clamp(
                    steering_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )
            /
            5.0
        )

        # -----------------------------------------------------
        # Stress level
        # -----------------------------------------------------

        if stress_score < 0.20:

            stress_level = "NORMAL"

        elif stress_score < 0.40:

            stress_level = "LOW"

        elif stress_score < 0.60:

            stress_level = "MODERATE"

        elif stress_score < 0.80:

            stress_level = "HIGH"

        else:

            stress_level = "CRITICAL"

        self.last_result = {

            "stress_score":
                float(
                    stress_score
                ),

            "stress_level":
                stress_level,

            "reliability":
                float(
                    reliability
                ),

            "signal_contributions":
                contributions,

            "signal_scores":
                signal_scores,

            "adaptive_weights":
                normalized_weights,
        }

        return dict(
            self.last_result
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.last_result = {

            "stress_score":
                0.0,

            "stress_level":
                "NORMAL",

            "reliability":
                0.0,

            "signal_contributions":
                {},
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):

        return {

            "base_weights":
                dict(
                    self.base_weights
                ),

            "last_result":
                dict(
                    self.last_result
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
        "DRIVER STRESS INDICATOR"
    )

    print(
        "v1.0 - STEP 8F SELF TEST"
    )

    print("=" * 70)

    print()

    indicator = (
        DriverStressIndicator()
    )

    # ---------------------------------------------------------
    # Normal scenario
    # ---------------------------------------------------------

    normal = indicator.calculate(

        respiration_values={

            "respiration_rate_bpm":
                15.0,

            "reliability":
                0.90,
        },

        heart_rate_values={

            "heart_rate_bpm":
                70.0,

            "reliability":
                0.90,
        },

        pose_values={

            "pitch":
                2.0,

            "yaw":
                3.0,

            "roll":
                1.0,
        },

        gaze_values={

            "gaze_direction":
                "CENTER",

            "gaze_away_duration":
                0.0,
        },

        steering_values={

            "irregularity_score":
                0.05,

            "reliability":
                1.0,
        },
    )

    print(
        "NORMAL SCENARIO"
    )

    print(
        f"  Stress score: "
        f"{normal['stress_score']:.3f}"
    )

    print(
        f"  Stress level: "
        f"{normal['stress_level']}"
    )

    print(
        f"  Reliability: "
        f"{normal['reliability']:.3f}"
    )

    print()

    # ---------------------------------------------------------
    # Elevated scenario
    # ---------------------------------------------------------

    elevated = indicator.calculate(

        respiration_values={

            "respiration_rate_bpm":
                27.0,

            "reliability":
                0.85,
        },

        heart_rate_values={

            "heart_rate_bpm":
                105.0,

            "reliability":
                0.90,
        },

        pose_values={

            "pitch":
                15.0,

            "yaw":
                25.0,

            "roll":
                8.0,
        },

        gaze_values={

            "gaze_direction":
                "LEFT",

            "gaze_away_duration":
                2.0,
        },

        steering_values={

            "irregularity_score":
                0.60,

            "reliability":
                1.0,
        },
    )

    print(
        "ELEVATED SCENARIO"
    )

    print(
        f"  Stress score: "
        f"{elevated['stress_score']:.3f}"
    )

    print(
        f"  Stress level: "
        f"{elevated['stress_level']}"
    )

    print(
        f"  Reliability: "
        f"{elevated['reliability']:.3f}"
    )

    print()

    # ---------------------------------------------------------
    # No physiological signal
    # ---------------------------------------------------------

    no_signal = indicator.calculate(

        respiration_values={

            "respiration_rate_bpm":
                0.0,

            "reliability":
                0.0,
        },

        heart_rate_values={

            "heart_rate_bpm":
                0.0,

            "reliability":
                0.0,
        },

        pose_values={

            "pitch":
                0.0,

            "yaw":
                0.0,

            "roll":
                0.0,
        },

        gaze_values={

            "gaze_direction":
                "CENTER",

            "gaze_away_duration":
                0.0,
        },

        steering_values={

            "irregularity_score":
                0.0,

            "reliability":
                1.0,
        },
    )

    print(
        "NO PHYSIOLOGICAL SIGNAL"
    )

    print(
        f"  Stress score: "
        f"{no_signal['stress_score']:.3f}"
    )

    print(
        f"  Stress level: "
        f"{no_signal['stress_level']}"
    )

    print(
        f"  Reliability: "
        f"{no_signal['reliability']:.3f}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is an experimental driver-state"
    )

    print(
        "stress indicator, NOT a medical diagnosis."
    )

    print()

    print(
        "STEP 8F SELF TEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()