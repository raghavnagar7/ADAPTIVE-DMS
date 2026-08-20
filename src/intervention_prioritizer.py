"""
=============================================================
ADAPTIVE-DMS
INTERVENTION PRIORITIZER
v1.0 - STEP 9C
=============================================================

Converts the safety decision into a prioritized intervention.

Priority:
    0 = NONE
    1 = ADVISORY
    2 = WARNING
    3 = URGENT
    4 = CRITICAL

This is an experimental driver-monitoring component and is
NOT safety-certified or a medical diagnostic system.
=============================================================
"""


class InterventionPrioritizer:

    PRIORITY = {
        "NONE": 0,
        "ADVISORY": 1,
        "WARNING": 2,
        "URGENT": 3,
        "CRITICAL": 4,
    }

    def __init__(self):

        self.last_result = {
            "priority": "NONE",
            "priority_score": 0,
            "action": "NO_ACTION",
            "reason": "NO_DATA",
            "confidence": 0.0,
            "reliability": 0.0,
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
    # DECISION → PRIORITY
    # =========================================================

    def decision_priority(
        self,
        decision,
    ):

        mapping = {
            "NO_ACTION": "NONE",
            "ADVISORY": "ADVISORY",
            "WARNING": "WARNING",
            "URGENT_WARNING": "URGENT",
            "CRITICAL_INTERVENTION": "CRITICAL",
        }

        return mapping.get(
            str(decision).upper(),
            "NONE",
        )

    # =========================================================
    # ACTION
    # =========================================================

    def action_for_priority(
        self,
        priority,
    ):

        actions = {

            "NONE":
                "NO_ACTION",

            "ADVISORY":
                "VISUAL_ADVISORY",

            "WARNING":
                "AUDIO_VISUAL_WARNING",

            "URGENT":
                "URGENT_AUDIO_VISUAL_WARNING",

            "CRITICAL":
                "CRITICAL_INTERVENTION",
        }

        return actions.get(
            priority,
            "NO_ACTION",
        )

    # =========================================================
    # REASON
    # =========================================================

    def build_reason(
        self,
        decision_values,
        driver_state_values,
    ):

        reason = str(
            decision_values.get(
                "reason",
                "",
            )
        ).strip()

        if reason:
            return reason

        state = str(
            driver_state_values.get(
                "driver_state",
                "UNKNOWN",
            )
        )

        if state == "NORMAL":
            return "Driver state stable"

        if state == "LOW_RISK":
            return "Low driver-state risk"

        if state == "MODERATE_RISK":
            return "Moderate driver-state risk"

        if state == "HIGH_RISK":
            return "High driver-state risk"

        if state == "CRITICAL":
            return "Critical driver-state risk"

        return "Multimodal driver-state risk"

    # =========================================================
    # MAIN
    # =========================================================

    def prioritize(
        self,
        decision_values=None,
        driver_state_values=None,
    ):

        decision_values = (
            decision_values or {}
        )

        driver_state_values = (
            driver_state_values or {}
        )

        decision = str(
            decision_values.get(
                "decision",
                "NO_ACTION",
            )
        ).upper()

        priority = self.decision_priority(
            decision
        )

        priority_score = (
            self.PRIORITY.get(
                priority,
                0,
            )
        )

        confidence = self.clamp(
            decision_values.get(
                "confidence",
                driver_state_values.get(
                    "confidence",
                    0.0,
                ),
            )
        )

        reliability = self.clamp(
            decision_values.get(
                "reliability",
                driver_state_values.get(
                    "reliability",
                    0.0,
                ),
            )
        )

        action = (
            self.action_for_priority(
                priority
            )
        )

        reason = self.build_reason(
            decision_values,
            driver_state_values,
        )

        # -----------------------------------------------------
        # Confidence gate
        # -----------------------------------------------------

        if (
            confidence < 0.25
            and priority_score > 1
        ):

            priority = "ADVISORY"

            priority_score = 1

            action = (
                "VISUAL_ADVISORY"
            )

            reason = (
                "Elevated risk detected "
                "with limited confidence"
            )

        self.last_result = {

            "priority":
                priority,

            "priority_score":
                priority_score,

            "action":
                action,

            "reason":
                reason,

            "confidence":
                float(confidence),

            "reliability":
                float(reliability),

            "source_decision":
                decision,

            "driver_state":
                driver_state_values.get(
                    "driver_state",
                    "UNKNOWN",
                ),

            "state_score":
                self.clamp(
                    driver_state_values.get(
                        "state_score",
                        0.0,
                    )
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
        decision_values=None,
        driver_state_values=None,
    ):

        return self.prioritize(
            decision_values,
            driver_state_values,
        )

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):

        return {
            "priority_levels":
                dict(self.PRIORITY),

            "last_result":
                dict(self.last_result),
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.last_result = {
            "priority": "NONE",
            "priority_score": 0,
            "action": "NO_ACTION",
            "reason": "NO_DATA",
            "confidence": 0.0,
            "reliability": 0.0,
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
        "INTERVENTION PRIORITIZER"
    )

    print(
        "v1.0 - STEP 9C SELF TEST"
    )

    print("=" * 70)

    print()

    prioritizer = (
        InterventionPrioritizer()
    )

    scenarios = [

        (
            "NORMAL",
            {
                "decision":
                    "NO_ACTION",

                "decision_score":
                    0.08,

                "confidence":
                    0.92,

                "reliability":
                    0.90,

                "reason":
                    "Driver state stable",
            },
            {
                "driver_state":
                    "NORMAL",

                "state_score":
                    0.08,

                "confidence":
                    0.92,

                "reliability":
                    0.90,
            },
        ),

        (
            "ADVISORY",
            {
                "decision":
                    "ADVISORY",

                "decision_score":
                    0.25,

                "confidence":
                    0.85,

                "reliability":
                    0.85,

                "reason":
                    "Low driver-state risk",
            },
            {
                "driver_state":
                    "LOW_RISK",

                "state_score":
                    0.25,

                "confidence":
                    0.85,

                "reliability":
                    0.85,
            },
        ),

        (
            "WARNING",
            {
                "decision":
                    "WARNING",

                "decision_score":
                    0.48,

                "confidence":
                    0.88,

                "reliability":
                    0.86,

                "reason":
                    "Multimodal fatigue risk",
            },
            {
                "driver_state":
                    "MODERATE_RISK",

                "state_score":
                    0.48,

                "confidence":
                    0.88,

                "reliability":
                    0.86,
            },
        ),

        (
            "URGENT",
            {
                "decision":
                    "URGENT_WARNING",

                "decision_score":
                    0.65,

                "confidence":
                    0.90,

                "reliability":
                    0.90,

                "reason":
                    "High driver-state risk",
            },
            {
                "driver_state":
                    "HIGH_RISK",

                "state_score":
                    0.65,

                "confidence":
                    0.90,

                "reliability":
                    0.90,
            },
        ),

        (
            "CRITICAL",
            {
                "decision":
                    "CRITICAL_INTERVENTION",

                "decision_score":
                    0.90,

                "confidence":
                    0.96,

                "reliability":
                    0.95,

                "reason":
                    "Critical driver-state risk",
            },
            {
                "driver_state":
                    "CRITICAL",

                "state_score":
                    0.90,

                "confidence":
                    0.96,

                "reliability":
                    0.95,
            },
        ),
    ]

    for (
        name,
        decision_values,
        state_values,
    ) in scenarios:

        result = (
            prioritizer.prioritize(
                decision_values,
                state_values,
            )
        )

        print(
            f"{name} SCENARIO"
        )

        print(
            f"  Priority: "
            f"{result['priority']}"
        )

        print(
            f"  Priority Score: "
            f"{result['priority_score']}"
        )

        print(
            f"  Action: "
            f"{result['action']}"
        )

        print(
            f"  Confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            f"  Reliability: "
            f"{result['reliability']:.3f}"
        )

        print(
            f"  Reason: "
            f"{result['reason']}"
        )

        print()

    print(
        "STEP 9C SELF TEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()