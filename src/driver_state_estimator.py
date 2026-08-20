"""
=============================================================
ADAPTIVE-DMS
=============================================================

DRIVER STATE ESTIMATOR

Version:
    v1.0 - STEP 9A

Purpose:
    Combine multimodal driver-state signals into one final
    driver state and confidence score.

Inputs:
    - Adaptive fusion fatigue risk
    - GRU temporal prediction
    - Stress indicator
    - Steering irregularity
    - Respiration
    - Signal reliability

Outputs:
    - driver_state
    - state_score
    - confidence
    - reliability
    - signal_contributions
    - state_reason

Important:
    This module is an experimental driver-state estimator.
    It is NOT a medical or safety-certified system.

=============================================================
"""


class DriverStateEstimator:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        normal_threshold=0.20,
        low_risk_threshold=0.35,
        moderate_risk_threshold=0.55,
        high_risk_threshold=0.75,
    ):

        self.normal_threshold = float(
            normal_threshold
        )

        self.low_risk_threshold = float(
            low_risk_threshold
        )

        self.moderate_risk_threshold = float(
            moderate_risk_threshold
        )

        self.high_risk_threshold = float(
            high_risk_threshold
        )

        # -----------------------------------------------------
        # Base weights
        # -----------------------------------------------------

        self.base_weights = {

            "fusion":
                0.30,

            "gru":
                0.20,

            "stress":
                0.15,

            "steering":
                0.15,

            "respiration":
                0.10,

            "reliability_adjustment":
                0.10,
        }

        self.last_result = {

            "driver_state":
                "UNKNOWN",

            "state_score":
                0.0,

            "confidence":
                0.0,

            "reliability":
                0.0,

            "signal_contributions":
                {},

            "state_reason":
                "NO_DATA",

            "risk_level":
                "UNKNOWN",
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

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            value = 0.0

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def safe_float(
        value,
        default=0.0,
    ):

        try:

            result = float(
                value
            )

            if result != result:

                return default

            return result

        except (
            TypeError,
            ValueError,
        ):

            return default

    # =========================================================
    # STATE CLASSIFICATION
    # =========================================================

    def classify_state(
        self,
        score,
    ):

        score = self.clamp(
            score
        )

        if score < self.normal_threshold:

            return "NORMAL"

        if score < self.low_risk_threshold:

            return "LOW_RISK"

        if score < self.moderate_risk_threshold:

            return "MODERATE_RISK"

        if score < self.high_risk_threshold:

            return "HIGH_RISK"

        return "CRITICAL"

    # =========================================================
    # RISK LEVEL
    # =========================================================

    def risk_level(
        self,
        score,
    ):

        score = self.clamp(
            score
        )

        if score < 0.20:

            return "NORMAL"

        if score < 0.35:

            return "LOW"

        if score < 0.55:

            return "MODERATE"

        if score < 0.75:

            return "HIGH"

        return "CRITICAL"

    # =========================================================
    # GRU RISK
    # =========================================================

    def calculate_gru_risk(
        self,
        gru_values,
    ):

        if not gru_values:

            return 0.0

        prediction = self.safe_float(
            gru_values.get(
                "prediction",
                gru_values.get(
                    "fatigue_risk",
                    0.0,
                ),
            )
        )

        fatigue_risk = self.safe_float(
            gru_values.get(
                "fatigue_risk",
                prediction,
            )
        )

        # Average prediction and fatigue risk
        # when both are available.

        if (
            prediction > 0.0
            and fatigue_risk > 0.0
        ):

            risk = (
                prediction
                +
                fatigue_risk
            ) / 2.0

        else:

            risk = max(
                prediction,
                fatigue_risk,
            )

        return self.clamp(
            risk
        )

    # =========================================================
    # STRESS RISK
    # =========================================================

    def calculate_stress_risk(
        self,
        stress_values,
    ):

        if not stress_values:

            return 0.0

        return self.clamp(
            stress_values.get(
                "stress_score",
                0.0,
            )
        )

    # =========================================================
    # STEERING RISK
    # =========================================================

    def calculate_steering_risk(
        self,
        steering_values,
    ):

        if not steering_values:

            return 0.0

        return self.clamp(
            steering_values.get(
                "irregularity_score",
                0.0,
            )
        )

    # =========================================================
    # RESPIRATION RISK
    # =========================================================

    def calculate_respiration_risk(
        self,
        respiration_values,
    ):

        if not respiration_values:

            return 0.0

        respiration_rate = (
            self.safe_float(
                respiration_values.get(
                    "respiration_rate_bpm",
                    0.0,
                )
            )
        )

        if respiration_rate <= 12.0:

            return 0.0

        if respiration_rate >= 30.0:

            return 1.0

        return self.clamp(
            (
                respiration_rate
                - 12.0
            )
            /
            18.0
        )

    # =========================================================
    # RELIABILITY
    # =========================================================

    def calculate_reliability(
        self,
        fusion_values,
        gru_values,
        stress_values,
        steering_values,
        respiration_values,
    ):

        values = []

        # -----------------------------------------------------
        # Fusion reliability
        # -----------------------------------------------------

        if fusion_values:

            fusion_reliability = (
                self.safe_float(
                    fusion_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

            # Some fusion implementations do not expose a
            # single reliability value. In that case use the
            # available adaptive weights.

            if fusion_reliability <= 0.0:

                weights = (
                    fusion_values.get(
                        "adaptive_weights",
                        {},
                    )
                )

                if weights:

                    fusion_reliability = (
                        sum(
                            weights.values()
                        )
                    )

            values.append(
                self.clamp(
                    fusion_reliability
                )
            )

        # -----------------------------------------------------
        # GRU confidence
        # -----------------------------------------------------

        if gru_values:

            gru_confidence = (
                self.clamp(
                    gru_values.get(
                        "confidence",
                        0.0,
                    )
                )
            )

            values.append(
                gru_confidence
            )

        # -----------------------------------------------------
        # Stress reliability
        # -----------------------------------------------------

        if stress_values:

            stress_reliability = (
                self.clamp(
                    stress_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

            values.append(
                stress_reliability
            )

        # -----------------------------------------------------
        # Steering reliability
        # -----------------------------------------------------

        if steering_values:

            steering_reliability = (
                self.clamp(
                    steering_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

            values.append(
                steering_reliability
            )

        # -----------------------------------------------------
        # Respiration reliability
        # -----------------------------------------------------

        if respiration_values:

            respiration_reliability = (
                self.clamp(
                    respiration_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

            values.append(
                respiration_reliability
            )

        if not values:

            return 0.0

        return self.clamp(
            sum(values)
            /
            len(values)
        )

    # =========================================================
    # STATE REASON
    # =========================================================

    def determine_reason(
        self,
        fusion_risk,
        gru_risk,
        stress_risk,
        steering_risk,
        respiration_risk,
        state,
    ):

        signals = {

            "fusion":
                fusion_risk,

            "gru":
                gru_risk,

            "stress":
                stress_risk,

            "steering":
                steering_risk,

            "respiration":
                respiration_risk,
        }

        strongest_signal = max(
            signals,
            key=signals.get,
        )

        strongest_value = (
            signals[
                strongest_signal
            ]
        )

        if state == "NORMAL":

            return (
                "Driver state stable"
            )

        if strongest_value < 0.20:

            return (
                "Multiple signals within normal range"
            )

        reason_names = {

            "fusion":
                "multimodal fatigue risk",

            "gru":
                "temporal fatigue prediction",

            "stress":
                "elevated stress indicator",

            "steering":
                "irregular steering behaviour",

            "respiration":
                "elevated respiration indicator",
        }

        return reason_names.get(
            strongest_signal,
            "multimodal risk",
        )

    # =========================================================
    # MAIN ESTIMATION
    # =========================================================

    def estimate(
        self,
        fusion_values=None,
        gru_values=None,
        stress_values=None,
        steering_values=None,
        respiration_values=None,
    ):

        fusion_values = (
            fusion_values
            or {}
        )

        gru_values = (
            gru_values
            or {}
        )

        stress_values = (
            stress_values
            or {}
        )

        steering_values = (
            steering_values
            or {}
        )

        respiration_values = (
            respiration_values
            or {}
        )

        # -----------------------------------------------------
        # Individual risk values
        # -----------------------------------------------------

        fusion_risk = self.clamp(
            fusion_values.get(
                "fatigue_risk",
                0.0,
            )
        )

        gru_risk = (
            self.calculate_gru_risk(
                gru_values
            )
        )

        stress_risk = (
            self.calculate_stress_risk(
                stress_values
            )
        )

        steering_risk = (
            self.calculate_steering_risk(
                steering_values
            )
        )

        respiration_risk = (
            self.calculate_respiration_risk(
                respiration_values
            )
        )

        # -----------------------------------------------------
        # Reliability
        # -----------------------------------------------------

        reliability = (
            self.calculate_reliability(
                fusion_values=fusion_values,
                gru_values=gru_values,
                stress_values=stress_values,
                steering_values=steering_values,
                respiration_values=respiration_values,
            )
        )

        # -----------------------------------------------------
        # Dynamic weights
        # -----------------------------------------------------

        weights = {

            "fusion":
                self.base_weights[
                    "fusion"
                ],

            "gru":
                self.base_weights[
                    "gru"
                ],

            "stress":
                self.base_weights[
                    "stress"
                ],

            "steering":
                self.base_weights[
                    "steering"
                ],

            "respiration":
                self.base_weights[
                    "respiration"
                ],
        }

        # -----------------------------------------------------
        # Reliability adjustment
        # -----------------------------------------------------

        reliability_bonus = (
            reliability
            *
            self.base_weights[
                "reliability_adjustment"
            ]
        )

        # -----------------------------------------------------
        # Weighted contributions
        # -----------------------------------------------------

        contributions = {

            "fusion":
                fusion_risk
                *
                weights["fusion"],

            "gru":
                gru_risk
                *
                weights["gru"],

            "stress":
                stress_risk
                *
                weights["stress"],

            "steering":
                steering_risk
                *
                weights["steering"],

            "respiration":
                respiration_risk
                *
                weights["respiration"],
        }

        # -----------------------------------------------------
        # Raw state score
        # -----------------------------------------------------

        raw_score = sum(
            contributions.values()
        )

        # Reliability adjustment should NOT artificially
        # increase risk. It controls confidence in the result.

        state_score = self.clamp(
            raw_score
        )

        # -----------------------------------------------------
        # State
        # -----------------------------------------------------

        driver_state = (
            self.classify_state(
                state_score
            )
        )

        risk_level = (
            self.risk_level(
                state_score
            )
        )

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        # Confidence represents how trustworthy the combined
        # estimate is, not how dangerous the driver is.

        signal_count = 5.0

        available_signals = 0.0

        if fusion_values:
            available_signals += 1.0

        if gru_values:
            available_signals += 1.0

        if stress_values:
            available_signals += 1.0

        if steering_values:
            available_signals += 1.0

        if respiration_values:
            available_signals += 1.0

        coverage = (
            available_signals
            /
            signal_count
        )

        confidence = self.clamp(
            (
                reliability
                *
                0.70
            )
            +
            (
                coverage
                *
                0.30
            )
        )

        # -----------------------------------------------------
        # Reason
        # -----------------------------------------------------

        reason = (
            self.determine_reason(
                fusion_risk=fusion_risk,
                gru_risk=gru_risk,
                stress_risk=stress_risk,
                steering_risk=steering_risk,
                respiration_risk=respiration_risk,
                state=driver_state,
            )
        )

        # -----------------------------------------------------
        # Result
        # -----------------------------------------------------

        self.last_result = {

            "driver_state":
                driver_state,

            "state_score":
                float(
                    state_score
                ),

            "confidence":
                float(
                    confidence
                ),

            "reliability":
                float(
                    reliability
                ),

            "signal_contributions":
                {
                    key:
                        float(value)
                    for key, value
                    in contributions.items()
                },

            "signal_risks":
                {

                    "fusion":
                        float(
                            fusion_risk
                        ),

                    "gru":
                        float(
                            gru_risk
                        ),

                    "stress":
                        float(
                            stress_risk
                        ),

                    "steering":
                        float(
                            steering_risk
                        ),

                    "respiration":
                        float(
                            respiration_risk
                        ),
                },

            "state_reason":
                reason,

            "risk_level":
                risk_level,

            "signal_coverage":
                float(
                    coverage
                ),
        }

        return dict(
            self.last_result
        )

    # =========================================================
    # UPDATE ALIAS
    # =========================================================

    def update(
        self,
        fusion_values=None,
        gru_values=None,
        stress_values=None,
        steering_values=None,
        respiration_values=None,
    ):

        return self.estimate(
            fusion_values=fusion_values,
            gru_values=gru_values,
            stress_values=stress_values,
            steering_values=steering_values,
            respiration_values=respiration_values,
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.last_result = {

            "driver_state":
                "UNKNOWN",

            "state_score":
                0.0,

            "confidence":
                0.0,

            "reliability":
                0.0,

            "signal_contributions":
                {},

            "state_reason":
                "NO_DATA",

            "risk_level":
                "UNKNOWN",
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):

        return {

            "thresholds": {

                "normal":
                    self.normal_threshold,

                "low_risk":
                    self.low_risk_threshold,

                "moderate_risk":
                    self.moderate_risk_threshold,

                "high_risk":
                    self.high_risk_threshold,
            },

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
        "DRIVER STATE ESTIMATOR"
    )

    print(
        "v1.0 - STEP 9A SELF TEST"
    )

    print("=" * 70)

    print()

    estimator = (
        DriverStateEstimator()
    )

    # =========================================================
    # NORMAL TEST
    # =========================================================

    normal_result = estimator.estimate(

        fusion_values={

            "fatigue_risk":
                0.08,

            "reliability":
                0.90,

            "adaptive_weights":
                {
                    "EAR":
                        0.20,

                    "PERCLOS":
                        0.20,

                    "Blink":
                        0.10,

                    "Microsleep":
                        0.15,

                    "MAR":
                        0.10,

                    "HeadPose":
                        0.15,

                    "Gaze":
                        0.10,
                },
        },

        gru_values={

            "prediction":
                0.07,

            "fatigue_risk":
                0.07,

            "confidence":
                0.90,
        },

        stress_values={

            "stress_score":
                0.10,

            "stress_level":
                "NORMAL",

            "reliability":
                0.85,
        },

        steering_values={

            "irregularity_score":
                0.05,

            "reliability":
                1.00,
        },

        respiration_values={

            "respiration_rate_bpm":
                15.0,

            "reliability":
                0.80,
        },
    )

    print(
        "NORMAL SCENARIO"
    )

    print(
        f"  Driver State: "
        f"{normal_result['driver_state']}"
    )

    print(
        f"  State Score: "
        f"{normal_result['state_score']:.3f}"
    )

    print(
        f"  Confidence: "
        f"{normal_result['confidence']:.3f}"
    )

    print(
        f"  Reliability: "
        f"{normal_result['reliability']:.3f}"
    )

    print(
        f"  Reason: "
        f"{normal_result['state_reason']}"
    )

    print()

    # =========================================================
    # MODERATE TEST
    # =========================================================

    moderate_result = estimator.estimate(

        fusion_values={

            "fatigue_risk":
                0.45,

            "reliability":
                0.90,
        },

        gru_values={

            "prediction":
                0.48,

            "fatigue_risk":
                0.45,

            "confidence":
                0.85,
        },

        stress_values={

            "stress_score":
                0.50,

            "stress_level":
                "MODERATE",

            "reliability":
                0.80,
        },

        steering_values={

            "irregularity_score":
                0.35,

            "reliability":
                1.00,
        },

        respiration_values={

            "respiration_rate_bpm":
                22.0,

            "reliability":
                0.80,
        },
    )

    print(
        "MODERATE SCENARIO"
    )

    print(
        f"  Driver State: "
        f"{moderate_result['driver_state']}"
    )

    print(
        f"  State Score: "
        f"{moderate_result['state_score']:.3f}"
    )

    print(
        f"  Confidence: "
        f"{moderate_result['confidence']:.3f}"
    )

    print(
        f"  Reliability: "
        f"{moderate_result['reliability']:.3f}"
    )

    print(
        f"  Reason: "
        f"{moderate_result['state_reason']}"
    )

    print()

    # =========================================================
    # CRITICAL TEST
    # =========================================================

    critical_result = estimator.estimate(

        fusion_values={

            "fatigue_risk":
                0.90,

            "reliability":
                0.95,
        },

        gru_values={

            "prediction":
                0.88,

            "fatigue_risk":
                0.90,

            "confidence":
                0.95,
        },

        stress_values={

            "stress_score":
                0.90,

            "stress_level":
                "CRITICAL",

            "reliability":
                0.90,
        },

        steering_values={

            "irregularity_score":
                0.90,

            "reliability":
                1.00,
        },

        respiration_values={

            "respiration_rate_bpm":
                29.0,

            "reliability":
                0.90,
        },
    )

    print(
        "CRITICAL SCENARIO"
    )

    print(
        f"  Driver State: "
        f"{critical_result['driver_state']}"
    )

    print(
        f"  State Score: "
        f"{critical_result['state_score']:.3f}"
    )

    print(
        f"  Confidence: "
        f"{critical_result['confidence']:.3f}"
    )

    print(
        f"  Reliability: "
        f"{critical_result['reliability']:.3f}"
    )

    print(
        f"  Reason: "
        f"{critical_result['state_reason']}"
    )

    print()

    # =========================================================
    # NO SIGNAL TEST
    # =========================================================

    estimator.reset()

    no_signal_result = (
        estimator.estimate()
    )

    print(
        "NO SIGNAL SCENARIO"
    )

    print(
        f"  Driver State: "
        f"{no_signal_result['driver_state']}"
    )

    print(
        f"  State Score: "
        f"{no_signal_result['state_score']:.3f}"
    )

    print(
        f"  Confidence: "
        f"{no_signal_result['confidence']:.3f}"
    )

    print(
        f"  Reliability: "
        f"{no_signal_result['reliability']:.3f}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This module is an experimental driver-state"
    )

    print(
        "estimator and is NOT safety-certified."
    )

    print()

    print(
        "STEP 9A SELF TEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()