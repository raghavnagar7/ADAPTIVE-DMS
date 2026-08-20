"""
=============================================================
ADAPTIVE-DMS
HEART RATE LOGGER
=============================================================

Version:
    v1.0 - STEP 8C

Purpose:
    Log camera-based non-contact heart-rate estimates.

Output:
    logs/heart_rate_predictions.csv

Important:
    Heart rate is an experimental rPPG estimate.
    It is NOT a medical-grade measurement.
=============================================================
"""

import os
import csv
from datetime import datetime


class HeartRateLogger:

    def __init__(
        self,
        log_directory="logs",
    ):

        self.log_directory = (
            log_directory
        )

        os.makedirs(
            self.log_directory,
            exist_ok=True,
        )

        self.file_path = os.path.join(
            self.log_directory,
            "heart_rate_predictions.csv",
        )

        self._initialize_file()

    # =========================================================
    # INITIALIZE
    # =========================================================

    def _initialize_file(self):

        if os.path.exists(
            self.file_path
        ):

            return

        with open(
            self.file_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "timestamp",
                    "heart_rate_bpm",
                    "raw_signal",
                    "filtered_signal",
                    "signal_quality",
                    "reliability",
                    "state",
                    "sample_count",
                    "signal_ready",
                    "roi_available",
                    "method",
                ]
            )

    # =========================================================
    # LOG
    # =========================================================

    def log(
        self,
        timestamp,
        heart_rate_values,
    ):

        try:

            with open(
                self.file_path,
                "a",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow(
                    [
                        datetime.fromtimestamp(
                            timestamp
                        ).isoformat(),

                        f"{float(heart_rate_values.get('heart_rate_bpm', 0.0)):.6f}",

                        f"{float(heart_rate_values.get('raw_signal', 0.0)):.6f}",

                        f"{float(heart_rate_values.get('filtered_signal', 0.0)):.6f}",

                        f"{float(heart_rate_values.get('signal_quality', 0.0)):.6f}",

                        f"{float(heart_rate_values.get('reliability', 0.0)):.6f}",

                        heart_rate_values.get(
                            "state",
                            "UNKNOWN",
                        ),

                        heart_rate_values.get(
                            "sample_count",
                            0,
                        ),

                        heart_rate_values.get(
                            "signal_ready",
                            False,
                        ),

                        heart_rate_values.get(
                            "roi_available",
                            False,
                        ),

                        heart_rate_values.get(
                            "method",
                            "GREEN_CHANNEL_RPPG",
                        ),
                    ]
                )

        except Exception as error:

            print(
                f"Heart-rate logging error: {error}"
            )

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):

        return {
            "file_path":
                self.file_path
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
        "HEART RATE LOGGER"
    )

    print(
        "v1.0 - STEP 8C SELF TEST"
    )

    print("=" * 70)

    logger = HeartRateLogger(
        log_directory="logs"
    )

    test_values = {

        "heart_rate_bpm":
            72.0,

        "raw_signal":
            145.0,

        "filtered_signal":
            0.42,

        "signal_quality":
            0.85,

        "reliability":
            0.90,

        "state":
            "NORMAL",

        "sample_count":
            360,

        "signal_ready":
            True,

        "roi_available":
            True,

        "method":
            "GREEN_CHANNEL_RPPG",
    }

    logger.log(
        timestamp=__import__(
            "time"
        ).time(),
        heart_rate_values=test_values,
    )

    print()

    print(
        "Test heart-rate entry logged."
    )

    print(
        f"Log file:"
    )

    print(
        logger.file_path
    )

    print()

    print(
        "STEP 8C HEART RATE LOGGER SELF TEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()