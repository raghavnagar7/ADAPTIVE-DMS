"""
=============================================================
ADAPTIVE-DMS
ADAPTIVE SAFETY DECISION ENGINE
v1.0 - STEP 9B
=============================================================

Converts driver-state signals into an intervention decision.

Levels:
    NO_ACTION
    ADVISORY
    WARNING
    URGENT_WARNING
    CRITICAL_INTERVENTION

This is an experimental driver-monitoring component and is
NOT safety-certified or a medical diagnostic system.
=============================================================
"""


class AdaptiveSafetyDecisionEngine:

    LEVELS = {
        "NO_ACTION": 0,
        "ADVISORY": 1,
        "WARNING": 2,
        "URGENT_WARNING": 3,
        "CRITICAL_INTERVENTION": 4,
    }

    def __init__(
        self,
        advisory_threshold=0.20,
        warning_threshold=0.35,
        urgent_threshold=0.55,
        critical_threshold=0.75,
        minimum_confidence=0.35,
        minimum_reliability=0.30,
    ):

        self.advisory_threshold = float(
            advisory_threshold
        )

        self.warning_threshold = float(
            warning_threshold
        )

        self.urgent_threshold = float(
            urgent_threshold
        )

        self.critical_threshold = float(
            critical_threshold
        )

        self.minimum_confidence = float(
            minimum_confidence
        )

        self.minimum_reliability = float(
            minimum_reliability
        )

        self.last_decision = {
            "decision": "NO_ACTION",
            "decision_score": 0.0,
            "confidence": 0.0,
            "reliability": 0.0,
            "reason": "NO_DATA",
        }

    # =========================================================
    # UTILITIES
    # =========================================================

    @staticmethod
    def clamp(
        value,
        minimum=0.0,
        maximum=1.0,
    ):

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        return max(
            minimum,
            min(maximum, value),
        )

    # =========================================================
    # THRESHOLD DECISION
    # =========================================================

    def threshold_decision(
        self,
        score,
    ):

        score = self.clamp(score)

        if score < self.advisory_threshold:
            return "NO_ACTION"

        if score < self.warning_threshold:
            return "ADVISORY"

        if score < self.urgent_threshold:
            return "WARNING"

        if score < self.critical_threshold:
            return "URGENT_WARNING"

        return "CRITICAL_INTERVENTION"

    # =========================================================
    # DRIVER STATE SCORE
    # =========================================================

    def state_score(
        self,
        driver_state_values,
    ):

        if not driver_state_values:
            return 0.0

        return self.clamp(
            driver_state_values.get(
                "state_score",
                0.0,
            )
        )

    # =========================================================
    # FUSION SCORE
    # =========================================================

    def fusion_score(
        self,
        fusion_values,
    ):

        if not fusion_values:
            return 0.0

        return self.clamp(
            fusion_values.get(
                "fatigue_risk",
                0.0,
            )
        )

    # =========================================================
    # GRU SCORE
    # =========================================================

    def gru_score(
        self,
        gru_values,
    ):

        if not gru_values:
            return 0.0

        prediction = self.clamp(
            gru_values.get(
                "prediction",
                0.0,
            )
        )

        fatigue = self.clamp(
            gru_values.get(
                "fatigue_risk",
                prediction,
            )
        )

        return max(
            prediction,
            fatigue,
        )

    # =========================================================
    # STRESS SCORE
    # =========================================================

    def stress_score(
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
    # STEERING SCORE
    # =========================================================

    def steering_score(
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
    # ESCALATION SCORE
    # =========================================================

    def calculate_score(
        self,
        driver_state_values=None,
        fusion_values=None,
        gru_values=None,
        stress_values=None,
        steering_values=None,
    ):

        driver_state_values = (
            driver_state_values or {}
        )

        fusion_values = (
            fusion_values or {}
        )

        gru_values = (
            gru_values or {}
        )

        stress_values = (
            stress_values or {}
        )

        steering_values = (
            steering_values or {}
        )

        state = self.state_score(
            driver_state_values
        )

        fusion = self.fusion_score(
            fusion_values
        )

        gru = self.gru_score(
            gru_values
        )

        stress = self.stress_score(
            stress_values
        )

        steering = self.steering_score(
            steering_values
        )

        # Driver-state estimate gets the highest weight.
        score = (
            state * 0.40
            +
            fusion * 0.25
            +
            gru * 0.15
            +
            stress * 0.10
            +
            steering * 0.10
        )

        return self.clamp(score)

    # =========================================================
    # CONFIDENCE
    # =========================================================

    def calculate_confidence(
        self,
        driver_state_values=None,
        gru_values=None,
        stress_values=None,
        steering_values=None,
    ):

        driver_state_values = (
            driver_state_values or {}
        )

        gru_values = (
            gru_values or {}
        )

        stress_values = (
            stress_values or {}
        )

        steering_values = (
            steering_values or {}
        )

        values = []

        if driver_state_values:
            values.append(
                self.clamp(
                    driver_state_values.get(
                        "confidence",
                        0.0,
                    )
                )
            )

        if gru_values:
            values.append(
                self.clamp(
                    gru_values.get(
                        "confidence",
                        0.0,
                    )
                )
            )

        if stress_values:
            values.append(
                self.clamp(
                    stress_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

        if steering_values:
            values.append(
                self.clamp(
                    steering_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

        if not values:
            return 0.0

        return self.clamp(
            sum(values) / len(values)
        )

    # =========================================================
    # RELIABILITY
    # =========================================================

    def calculate_reliability(
        self,
        driver_state_values=None,
        fusion_values=None,
        stress_values=None,
        steering_values=None,
    ):

        driver_state_values = (
            driver_state_values or {}
        )

        fusion_values = (
            fusion_values or {}
        )

        stress_values = (
            stress_values or {}
        )

        steering_values = (
            steering_values or {}
        )

        values = []

        if driver_state_values:
            values.append(
                self.clamp(
                    driver_state_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

        if fusion_values:
            fusion_reliability = (
                self.clamp(
                    fusion_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

            if fusion_reliability <= 0.0:
                weights = fusion_values.get(
                    "adaptive_weights",
                    {},
                )

                if weights:
                    fusion_reliability = 1.0

            values.append(
                fusion_reliability
            )

        if stress_values:
            values.append(
                self.clamp(
                    stress_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

        if steering_values:
            values.append(
                self.clamp(
                    steering_values.get(
                        "reliability",
                        0.0,
                    )
                )
            )

        if not values:
            return 0.0

        return self.clamp(
            sum(values) / len(values)
        )

    # =========================================================
    # REASON
    # =========================================================

    def determine_reason(
        self,
        score,
        driver_state_values,
        fusion_values,
        gru_values,
        stress_values,
        steering_values,
    ):

        if score < self.advisory_threshold:
            return "Driver state stable"

        signals = {
            "driver_state":
                self.state_score(
                    driver_state_values
                ),

            "fatigue_fusion":
                self.fusion_score(
                    fusion_values
                ),

            "gru_prediction":
                self.gru_score(
                    gru_values
                ),

            "stress":
                self.stress_score(
                    stress_values
                ),

            "steering":
                self.steering_score(
                    steering_values
                ),
        }

        strongest = max(
            signals,
            key=signals.get,
        )

        reasons = {
            "driver_state":
                "elevated driver-state risk",

            "fatigue_fusion":
                "multimodal fatigue risk",

            "gru_prediction":
                "temporal fatigue prediction",

            "stress":
                "elevated stress indicator",

            "steering":
                "irregular steering behaviour",
        }

        return reasons.get(
            strongest,
            "multimodal risk",
        )

    # =========================================================
    # MAIN DECISION
    # =========================================================

    def decide(
        self,
        driver_state_values=None,
        fusion_values=None,
        gru_values=None,
        stress_values=None,
        steering_values=None,
    ):

        driver_state_values = (
            driver_state_values or {}
        )

        fusion_values = (
            fusion_values or {}
        )

        gru_values = (
            gru_values or {}
        )

        stress_values = (
            stress_values or {}
        )

        steering_values = (
            steering_values or {}
        )

        score = self.calculate_score(
            driver_state_values,
            fusion_values,
            gru_values,
            stress_values,
            steering_values,
        )

        confidence = self.calculate_confidence(
            driver_state_values,
            gru_values,
            stress_values,
            steering_values,
        )

        reliability = self.calculate_reliability(
            driver_state_values,
            fusion_values,
            stress_values,
            steering_values,
        )

        decision = self.threshold_decision(
            score
        )

        # -----------------------------------------------------
        # Reliability/confidence gate
        # -----------------------------------------------------

        if (
            confidence
            <
            self.minimum_confidence
            or
            reliability
            <
            self.minimum_reliability
        ):

            if score < self.urgent_threshold:

                decision = "ADVISORY"

        reason = self.determine_reason(
            score,
            driver_state_values,
            fusion_values,
            gru_values,
            stress_values,
            steering_values,
        )

        self.last_decision = {

            "decision":
                decision,

            "decision_score":
                float(score),

            "confidence":
                float(confidence),

            "reliability":
                float(reliability),

            "reason":
                reason,
        }

        return dict(
            self.last_decision
        )

    # =========================================================
    # UPDATE ALIAS
    # =========================================================

    def update(
        self,
        driver_state_values=None,
        fusion_values=None,
        gru_values=None,
        stress_values=None,
        steering_values=None,
    ):

        return self.decide(
            driver_state_values,
            fusion_values,
            gru_values,
            stress_values,
            steering_values,
        )

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):

        return {
            "thresholds": {
                "advisory":
                    self.advisory_threshold,

                "warning":
                    self.warning_threshold,

                "urgent":
                    self.urgent_threshold,

                "critical":
                    self.critical_threshold,
            },

            "minimum_confidence":
                self.minimum_confidence,

            "minimum_reliability":
                self.minimum_reliability,

            "last_decision":
                dict(
                    self.last_decision
                ),
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.last_decision = {
            "decision":
                "NO_ACTION",

            "decision_score":
                0.0,

            "confidence":
                0.0,

            "reliability":
                0.0,

            "reason":
                "NO_DATA",
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
        "ADAPTIVE SAFETY DECISION ENGINE"
    )

    print(
        "v1.0 - STEP 9B SELF TEST"
    )

    print("=" * 70)

    print()

    engine = (
        AdaptiveSafetyDecisionEngine()
    )

    # =========================================================
    # NORMAL
    # =========================================================

    normal = engine.decide(

        driver_state_values={
            "state_score":
                0.08,

            "confidence":
                0.92,

            "reliability":
                0.90,
        },

        fusion_values={
            "fatigue_risk":
                0.08,

            "reliability":
                0.90,
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

            "reliability":
                0.85,
        },

        steering_values={
            "irregularity_score":
                0.05,

            "reliability":
                1.00,
        },
    )

    print(
        "NORMAL SCENARIO"
    )

    print(
        f"  Decision: "
        f"{normal['decision']}"
    )

    print(
        f"  Score: "
        f"{normal['decision_score']:.3f}"
    )

    print(
        f"  Confidence: "
        f"{normal['confidence']:.3f}"
    )

    print(
        f"  Reliability: "
        f"{normal['reliability']:.3f}"
    )

    print()

    # =========================================================
    # ADVISORY
    # =========================================================

    advisory = engine.decide(

        driver_state_values={
            "state_score":
                0.25,

            "confidence":
                0.80,

            "reliability":
                0.80,
        },

        fusion_values={
            "fatigue_risk":
                0.22,

            "reliability":
                0.80,
        },

        gru_values={
            "prediction":
                0.20,

            "fatigue_risk":
                0.22,

            "confidence":
                0.80,
        },

        stress_values={
            "stress_score":
                0.20,

            "reliability":
                0.80,
        },

        steering_values={
            "irregularity_score":
                0.10,

            "reliability":
                1.00,
        },
    )

    print(
        "ADVISORY SCENARIO"
    )

    print(
        f"  Decision: "
        f"{advisory['decision']}"
    )

    print(
        f"  Score: "
        f"{advisory['decision_score']:.3f}"
    )

    print()

    # =========================================================
    # WARNING
    # =========================================================

    warning = engine.decide(

        driver_state_values={
            "state_score":
                0.50,

            "confidence":
                0.85,

            "reliability":
                0.85,
        },

        fusion_values={
            "fatigue_risk":
                0.45,

            "reliability":
                0.85,
        },

        gru_values={
            "prediction":
                0.50,

            "fatigue_risk":
                0.48,

            "confidence":
                0.85,
        },

        stress_values={
            "stress_score":
                0.40,

            "reliability":
                0.80,
        },

        steering_values={
            "irregularity_score":
                0.30,

            "reliability":
                1.00,
        },
    )

    print(
        "WARNING SCENARIO"
    )

    print(
        f"  Decision: "
        f"{warning['decision']}"
    )

    print(
        f"  Score: "
        f"{warning['decision_score']:.3f}"
    )

    print()

    # =========================================================
    # CRITICAL
    # =========================================================

    critical = engine.decide(

        driver_state_values={
            "state_score":
                0.90,

            "confidence":
                0.95,

            "reliability":
                0.95,
        },

        fusion_values={
            "fatigue_risk":
                0.90,

            "reliability":
                0.95,
        },

        gru_values={
            "prediction":
                0.90,

            "fatigue_risk":
                0.90,

            "confidence":
                0.95,
        },

        stress_values={
            "stress_score":
                0.90,

            "reliability":
                0.90,
        },

        steering_values={
            "irregularity_score":
                0.90,

            "reliability":
                1.00,
        },
    )

    print(
        "CRITICAL SCENARIO"
    )

    print(
        f"  Decision: "
        f"{critical['decision']}"
    )

    print(
        f"  Score: "
        f"{critical['decision_score']:.3f}"
    )

    print(
        f"  Confidence: "
        f"{critical['confidence']:.3f}"
    )

    print(
        f"  Reliability: "
        f"{critical['reliability']:.3f}"
    )

    print()

    print(
        "STEP 9B SELF TEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()