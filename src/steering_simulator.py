"""
=============================================================
ADAPTIVE-DMS
=============================================================

Keyboard Steering Simulator

Version:
    v1.0 - STEP 7B

Purpose:
    Simulate steering-wheel input using the keyboard.

Controls:
    A / LEFT ARROW   = Steer Left
    D / RIGHT ARROW  = Steer Right
    W / UP ARROW     = Move Toward Center
    S / DOWN ARROW   = Faster Steering Movement
    SPACE            = Center Steering
    Q                = Quit

Output:
    - steering_angle
    - steering_change
    - steering_rate
    - steering_variability
    - sudden_correction
    - irregularity_score
    - reliability
    - driving_state

IMPORTANT:
    This is a steering INPUT SIMULATOR.

    It does NOT claim that the webcam can measure the
    physical steering-wheel angle.

=============================================================
"""

import os
import csv
import time
from datetime import datetime

from steering import SteeringBehaviourAnalyzer


# =============================================================
# CONFIGURATION
# =============================================================

MIN_ANGLE = -450.0
MAX_ANGLE = 450.0

NORMAL_STEP = 10.0
FAST_STEP = 25.0

UPDATE_INTERVAL = 0.10

LOG_DIRECTORY = "logs"

LOG_FILE = os.path.join(
    LOG_DIRECTORY,
    "steering_predictions.csv",
)


# =============================================================
# LOG INITIALIZATION
# =============================================================

def initialize_log():

    os.makedirs(
        LOG_DIRECTORY,
        exist_ok=True,
    )

    if not os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "timestamp",
                    "steering_angle",
                    "steering_change",
                    "steering_rate",
                    "steering_variability",
                    "sudden_correction",
                    "irregularity_score",
                    "reliability",
                    "driving_state",
                    "sample_count",
                ]
            )


# =============================================================
# LOG RESULT
# =============================================================

def log_result(result):

    try:

        with open(
            LOG_FILE,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    datetime.now().isoformat(),

                    f"{result['steering_angle']:.6f}",

                    f"{result['steering_change']:.6f}",

                    f"{result['steering_rate']:.6f}",

                    f"{result['steering_variability']:.6f}",

                    result["sudden_correction"],

                    f"{result['irregularity_score']:.6f}",

                    f"{result['reliability']:.6f}",

                    result["driving_state"],

                    result["sample_count"],
                ]
            )

    except Exception as error:

        print(
            f"Logging error: {error}"
        )


# =============================================================
# CLAMP ANGLE
# =============================================================

def clamp_angle(angle):

    return max(
        MIN_ANGLE,
        min(
            MAX_ANGLE,
            angle,
        ),
    )


# =============================================================
# KEYBOARD INPUT
# =============================================================

def get_key():

    """
    Read one keyboard key without requiring Enter.

    Windows implementation.
    """

    import msvcrt

    if not msvcrt.kbhit():

        return None

    key = msvcrt.getch()

    # ---------------------------------------------------------
    # Normal key
    # ---------------------------------------------------------

    if key in (
        b"a",
        b"A",
        b"d",
        b"D",
        b"w",
        b"W",
        b"s",
        b"S",
        b" ",
        b"q",
        b"Q",
    ):

        return key.decode(
            "utf-8"
        ).lower()

    # ---------------------------------------------------------
    # Arrow keys
    # ---------------------------------------------------------

    if key in (
        b"\x00",
        b"\xe0",
    ):

        second_key = msvcrt.getch()

        # LEFT ARROW
        if second_key == b"K":
            return "left"

        # RIGHT ARROW
        if second_key == b"M":
            return "right"

        # UP ARROW
        if second_key == b"H":
            return "up"

        # DOWN ARROW
        if second_key == b"P":
            return "down"

    return None


# =============================================================
# PRINT RESULT
# =============================================================

def print_result(result):

    print(
        "\033[H\033[J",
        end="",
    )

    print("=" * 70)

    print(
        "ADAPTIVE-DMS"
    )

    print(
        "STEERING BEHAVIOUR SIMULATOR"
    )

    print(
        "v1.0 - STEP 7B"
    )

    print("=" * 70)

    print()

    print(
        "CONTROLS"
    )

    print(
        "  A / LEFT ARROW  = Steer Left"
    )

    print(
        "  D / RIGHT ARROW = Steer Right"
    )

    print(
        "  W / UP ARROW    = Move Toward Center"
    )

    print(
        "  S / DOWN ARROW  = Fast Movement"
    )

    print(
        "  SPACE           = Center"
    )

    print(
        "  Q               = Quit"
    )

    print()

    print("-" * 70)

    print(
        f"Steering Angle       : "
        f"{result['steering_angle']:8.2f}°"
    )

    print(
        f"Steering Change      : "
        f"{result['steering_change']:8.2f}°"
    )

    print(
        f"Steering Rate        : "
        f"{result['steering_rate']:8.2f}°/s"
    )

    print(
        f"Variability          : "
        f"{result['steering_variability']:.3f}"
    )

    print(
        f"Sudden Correction    : "
        f"{result['sudden_correction']}"
    )

    print(
        f"Irregularity Score   : "
        f"{result['irregularity_score']:.3f}"
    )

    print(
        f"Reliability          : "
        f"{result['reliability']:.3f}"
    )

    print(
        f"Driving State        : "
        f"{result['driving_state']}"
    )

    print(
        f"Samples              : "
        f"{result['sample_count']}"
    )

    print("-" * 70)

    print()

    print(
        f"Log file: {LOG_FILE}"
    )

    print()

    print(
        "Press Q to quit."
    )


# =============================================================
# MAIN
# =============================================================

def main():

    print("=" * 70)

    print(
        "ADAPTIVE-DMS"
    )

    print(
        "STEERING BEHAVIOUR SIMULATOR"
    )

    print(
        "v1.0 - STEP 7B"
    )

    print("=" * 70)

    print()

    print(
        "Initializing steering analyzer..."
    )

    analyzer = SteeringBehaviourAnalyzer(
        history_size=50,
        maximum_steering_angle=450.0,
        sudden_change_threshold=45.0,
        high_rate_threshold=180.0,
        irregularity_threshold=0.55,
    )

    initialize_log()

    print(
        f"Log file: {LOG_FILE}"
    )

    print()

    print(
        "Controls:"
    )

    print(
        "A / LEFT  = Left"
    )

    print(
        "D / RIGHT = Right"
    )

    print(
        "W / UP    = Center"
    )

    print(
        "S / DOWN  = Fast movement"
    )

    print(
        "SPACE     = Center"
    )

    print(
        "Q         = Quit"
    )

    print()

    print(
        "Starting simulator..."
    )

    time.sleep(2)

    # ---------------------------------------------------------
    # Initial angle
    # ---------------------------------------------------------

    steering_angle = 0.0

    previous_update = time.time()

    running = True

    # =========================================================
    # MAIN LOOP
    # =========================================================

    while running:

        current_time = time.time()

        key = get_key()

        # -----------------------------------------------------
        # KEYBOARD CONTROL
        # -----------------------------------------------------

        if key in (
            "a",
            "left",
        ):

            steering_angle -= NORMAL_STEP

        elif key in (
            "d",
            "right",
        ):

            steering_angle += NORMAL_STEP

        elif key in (
            "s",
            "down",
        ):

            # Fast right movement
            steering_angle += FAST_STEP

        elif key in (
            "w",
            "up",
        ):

            # Move toward center
            if steering_angle > 0:

                steering_angle -= FAST_STEP

                if steering_angle < 0:
                    steering_angle = 0.0

            elif steering_angle < 0:

                steering_angle += FAST_STEP

                if steering_angle > 0:
                    steering_angle = 0.0

        elif key == " ":

            steering_angle = 0.0

        elif key == "q":

            running = False

            continue

        # -----------------------------------------------------
        # LIMIT STEERING
        # -----------------------------------------------------

        steering_angle = clamp_angle(
            steering_angle
        )

        # -----------------------------------------------------
        # UPDATE ANALYZER
        # -----------------------------------------------------

        if (
            current_time
            - previous_update
            >= UPDATE_INTERVAL
        ):

            result = analyzer.update(
                steering_angle=steering_angle,
                timestamp=current_time,
                input_available=True,
            )

            previous_update = current_time

            # -------------------------------------------------
            # LOG
            # -------------------------------------------------

            log_result(
                result
            )

            # -------------------------------------------------
            # DISPLAY
            # -------------------------------------------------

            print_result(
                result
            )

        time.sleep(0.01)

    # =========================================================
    # END
    # =========================================================

    print()

    print("=" * 70)

    print(
        "STEP 7B STEERING SIMULATOR STOPPED"
    )

    print(
        f"Steering log saved to: {LOG_FILE}"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()