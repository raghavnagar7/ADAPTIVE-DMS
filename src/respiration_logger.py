"""
=============================================================
ADAPTIVE-DMS
=============================================================

RESPIRATION LOGGER

Version:
    v1.0 - STEP 8D LOGGING

Purpose:
    Save non-contact respiration estimates to CSV.

Output:
    logs/respiration_predictions.csv

=============================================================
"""

import csv
import os


class RespirationLogger:

    def __init__(
        self,
        file_path="logs/respiration_predictions.csv",
    ):

        self.file_path = file_path

        directory = os.path.dirname(
            self.file_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        self.fieldnames = [

            "timestamp",

            "respiration_rate_bpm",

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

        self._ensure_file()

    # =========================================================
    # CREATE CSV
    # =========================================================

    def _ensure_file(self):

        if (
            not os.path.exists(
                self.file_path
            )
            or
            os.path.getsize(
                self.file_path
            )
            == 0
        ):

            with open(
                self.file_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fieldnames,
                )

                writer.writeheader()

    # =========================================================
    # SAFE VALUE
    # =========================================================

    @staticmethod
    def _safe_value(
        value,
        default=0.0,
    ):

        if value is None:

            return default

        return value

    # =========================================================
    # LOG
    # =========================================================

    def log(
        self,
        timestamp,
        respiration_values,
    ):

        if respiration_values is None:

            respiration_values = {}

        row = {

            "timestamp":
                self._safe_value(
                    timestamp,
                    "",
                ),

            "respiration_rate_bpm":
                self._safe_value(
                    respiration_values.get(
                        "respiration_rate_bpm",
                        0.0,
                    )
                ),

            "raw_signal":
                self._safe_value(
                    respiration_values.get(
                        "raw_signal",
                        0.0,
                    )
                ),

            "filtered_signal":
                self._safe_value(
                    respiration_values.get(
                        "filtered_signal",
                        0.0,
                    )
                ),

            "signal_quality":
                self._safe_value(
                    respiration_values.get(
                        "signal_quality",
                        0.0,
                    )
                ),

            "reliability":
                self._safe_value(
                    respiration_values.get(
                        "reliability",
                        0.0,
                    )
                ),

            "state":
                self._safe_value(
                    respiration_values.get(
                        "state",
                        "UNKNOWN",
                    ),
                    "UNKNOWN",
                ),

            "sample_count":
                self._safe_value(
                    respiration_values.get(
                        "sample_count",
                        0,
                    ),
                    0,
                ),

            "signal_ready":
                self._safe_value(
                    respiration_values.get(
                        "signal_ready",
                        False,
                    ),
                    False,
                ),

            "roi_available":
                self._safe_value(
                    respiration_values.get(
                        "roi_available",
                        False,
                    ),
                    False,
                ),

            "method":
                self._safe_value(
                    respiration_values.get(
                        "method",
                        "VISUAL_INTENSITY_RESPIRATION",
                    ),
                    "VISUAL_INTENSITY_RESPIRATION",
                ),
        }

        with open(
            self.file_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames,
            )

            writer.writerow(
                row
            )

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):

        return {

            "file_path":
                self.file_path,

            "fields":
                list(
                    self.fieldnames
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
        "RESPIRATION LOGGER"
    )

    print(
        "v1.0 - STEP 8D LOGGING SELF TEST"
    )

    print("=" * 70)

    print()

    logger = RespirationLogger()

    print(
        "Logger status:"
    )

    print(
        f"  File: "
        f"{logger.file_path}"
    )

    print()

    test_values = {

        "respiration_rate_bpm":
            15.0,

        "raw_signal":
            0.42,

        "filtered_signal":
            0.18,

        "signal_quality":
            0.91,

        "reliability":
            0.88,

        "state":
            "NORMAL",

        "sample_count":
            600,

        "signal_ready":
            True,

        "roi_available":
            True,

        "method":
            "VISUAL_INTENSITY_RESPIRATION",
    }

    logger.log(
        timestamp="SELF_TEST",
        respiration_values=test_values,
    )

    print(
        "Test record written."
    )

    print()

    status = (
        logger.get_status()
    )

    print(
        "LOGGER STATUS"
    )

    print(
        f"  File: "
        f"{status['file_path']}"
    )

    print(
        f"  Fields: "
        f"{len(status['fields'])}"
    )

    print()

    print(
        "STEP 8D LOGGING SELF TEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()