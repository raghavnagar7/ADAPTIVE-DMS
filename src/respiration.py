"""
=============================================================
ADAPTIVE-DMS
=============================================================

NON-CONTACT RESPIRATION ESTIMATOR

Version:
    v1.0 - STEP 8D SELF TEST

Purpose:
    Estimate respiration rate from a camera-based
    non-contact visual signal.

Important:
    This is an experimental research estimate.

    It is NOT a medical-grade respiration measurement.

Method:
    Green-channel / visual intensity temporal analysis.

Outputs:
    - respiration_rate_bpm
    - raw_signal
    - filtered_signal
    - signal_quality
    - reliability
    - state
    - sample_count
    - signal_ready
    - roi_available
    - method

=============================================================
"""

from collections import deque
import math
import statistics
import time

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False


class NonContactRespirationEstimator:

    def __init__(
        self,
        history_seconds=20.0,
        sample_rate=30.0,
        min_breaths_per_minute=6.0,
        max_breaths_per_minute=40.0,
        minimum_samples=180,
        smoothing_window=7,
    ):

        self.history_seconds = float(
            history_seconds
        )

        self.sample_rate = float(
            sample_rate
        )

        self.min_breaths_per_minute = float(
            min_breaths_per_minute
        )

        self.max_breaths_per_minute = float(
            max_breaths_per_minute
        )

        self.minimum_samples = int(
            minimum_samples
        )

        self.maximum_samples = int(
            self.history_seconds
            * self.sample_rate
        )

        self.smoothing_window = int(
            smoothing_window
        )

        if self.smoothing_window < 3:

            self.smoothing_window = 3

        if self.smoothing_window % 2 == 0:

            self.smoothing_window += 1

        self.signal_history = deque(
            maxlen=self.maximum_samples
        )

        self.time_history = deque(
            maxlen=self.maximum_samples
        )

        self.filtered_history = deque(
            maxlen=self.maximum_samples
        )

        self.last_result = {

            "respiration_rate_bpm":
                0.0,

            "raw_signal":
                0.0,

            "filtered_signal":
                0.0,

            "signal_quality":
                0.0,

            "reliability":
                0.0,

            "state":
                "NO_SIGNAL",

            "sample_count":
                0,

            "signal_ready":
                False,

            "roi_available":
                False,

            "method":
                "VISUAL_INTENSITY_RESPIRATION",
        }

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):

        try:

            result = float(
                value
            )

            if not math.isfinite(
                result
            ):

                return default

            return result

        except (
            TypeError,
            ValueError,
        ):

            return default

    # =========================================================
    # CLAMP
    # =========================================================

    @staticmethod
    def _clamp(
        value,
        minimum=0.0,
        maximum=1.0,
    ):

        value = float(
            value
        )

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # =========================================================
    # ROI EXTRACTION
    # =========================================================

    def extract_roi_signal(
        self,
        frame,
    ):
        """
        Extract a central upper-body / face-region
        visual intensity signal.

        This is a non-contact experimental signal.
        """

        if frame is None:

            return (
                0.0,
                False,
            )

        if not OPENCV_AVAILABLE:

            return (
                0.0,
                False,
            )

        try:

            height, width = (
                frame.shape[:2]
            )

            if (
                height <= 0
                or
                width <= 0
            ):

                return (
                    0.0,
                    False,
                )

            # -------------------------------------------------
            # Central upper region.
            #
            # This deliberately avoids claiming that the
            # webcam can directly measure respiration.
            # -------------------------------------------------

            x1 = int(
                width * 0.25
            )

            x2 = int(
                width * 0.75
            )

            y1 = int(
                height * 0.15
            )

            y2 = int(
                height * 0.70
            )

            if (
                x2 <= x1
                or
                y2 <= y1
            ):

                return (
                    0.0,
                    False,
                )

            roi = frame[
                y1:y2,
                x1:x2,
            ]

            if roi.size == 0:

                return (
                    0.0,
                    False,
                )

            gray = cv2.cvtColor(
                roi,
                cv2.COLOR_BGR2GRAY,
            )

            signal = float(
                np.mean(gray)
                if np is not None
                else gray.mean()
            )

            if not math.isfinite(
                signal
            ):

                return (
                    0.0,
                    False,
                )

            return (
                signal,
                True,
            )

        except Exception:

            return (
                0.0,
                False,
            )

    # =========================================================
    # SIGNAL NORMALIZATION
    # =========================================================

    def _normalize_signal(
        self,
        values,
    ):

        if np is None:

            return list(
                values
            )

        array = np.asarray(
            values,
            dtype=float,
        )

        if len(array) == 0:

            return array

        mean_value = float(
            np.mean(array)
        )

        standard_deviation = float(
            np.std(array)
        )

        if (
            not math.isfinite(
                standard_deviation
            )
            or
            standard_deviation < 1e-8
        ):

            return np.zeros_like(
                array
            )

        return (
            array
            -
            mean_value
        ) / standard_deviation

    # =========================================================
    # BANDPASS FILTER
    # =========================================================

    def _bandpass_filter(
        self,
        values,
    ):
        """
        Respiration band:

            approximately 0.10 - 0.67 Hz

        Equivalent to:

            6 - 40 breaths/minute
        """

        if np is None:

            return list(
                values
            )

        array = np.asarray(
            values,
            dtype=float,
        )

        if len(array) < 30:

            return array

        if not SCIPY_AVAILABLE:

            return array

        try:

            nyquist = (
                self.sample_rate
                /
                2.0
            )

            low = (
                0.10
                /
                nyquist
            )

            high = (
                0.67
                /
                nyquist
            )

            low = max(
                0.001,
                min(
                    0.99,
                    low,
                ),
            )

            high = max(
                low + 0.001,
                min(
                    0.99,
                    high,
                ),
            )

            b, a = butter(
                2,
                [
                    low,
                    high,
                ],
                btype="band",
            )

            if len(array) < (
                3
                *
                max(
                    len(a),
                    len(b),
                )
            ):

                return array

            filtered = filtfilt(
                b,
                a,
                array,
            )

            return filtered

        except Exception:

            return array

    # =========================================================
    # SMOOTHING
    # =========================================================

    def _smooth(
        self,
        values,
    ):

        if np is None:

            return list(
                values
            )

        array = np.asarray(
            values,
            dtype=float,
        )

        if len(array) < 3:

            return array

        window = min(
            self.smoothing_window,
            len(array),
        )

        if window % 2 == 0:

            window -= 1

        if window < 3:

            return array

        kernel = (
            np.ones(
                window
            )
            /
            window
        )

        return np.convolve(
            array,
            kernel,
            mode="same",
        )

    # =========================================================
    # PEAK DETECTION
    # =========================================================

    def _estimate_breathing_rate(
        self,
        filtered_signal,
    ):

        if np is None:

            return 0.0

        values = np.asarray(
            filtered_signal,
            dtype=float,
        )

        if len(values) < (
            self.minimum_samples
        ):

            return 0.0

        if SCIPY_AVAILABLE:

            try:

                from scipy.signal import find_peaks

                minimum_distance = int(
                    self.sample_rate
                    *
                    60.0
                    /
                    self.max_breaths_per_minute
                )

                minimum_distance = max(
                    1,
                    minimum_distance,
                )

                standard_deviation = float(
                    np.std(values)
                )

                if standard_deviation < 1e-8:

                    return 0.0

                prominence = (
                    standard_deviation
                    *
                    0.25
                )

                peaks, properties = (
                    find_peaks(
                        values,
                        distance=(
                            minimum_distance
                        ),
                        prominence=prominence,
                    )
                )

                if len(peaks) < 2:

                    return 0.0

                intervals = np.diff(
                    peaks
                )

                if len(intervals) == 0:

                    return 0.0

                median_interval = float(
                    np.median(
                        intervals
                    )
                )

                if median_interval <= 0:

                    return 0.0

                breathing_rate = (
                    self.sample_rate
                    *
                    60.0
                    /
                    median_interval
                )

                if (
                    breathing_rate
                    <
                    self.min_breaths_per_minute
                    or
                    breathing_rate
                    >
                    self.max_breaths_per_minute
                ):

                    return 0.0

                return float(
                    breathing_rate
                )

            except Exception:

                pass

        # -----------------------------------------------------
        # FFT fallback
        # -----------------------------------------------------

        try:

            values = (
                values
                -
                np.mean(values)
            )

            window = np.hanning(
                len(values)
            )

            spectrum = np.abs(
                np.fft.rfft(
                    values
                    *
                    window
                )
            )

            frequencies = (
                np.fft.rfftfreq(
                    len(values),
                    d=(
                        1.0
                        /
                        self.sample_rate
                    ),
                )
            )

            minimum_frequency = (
                self.min_breaths_per_minute
                /
                60.0
            )

            maximum_frequency = (
                self.max_breaths_per_minute
                /
                60.0
            )

            mask = (
                (frequencies >= minimum_frequency)
                &
                (frequencies <= maximum_frequency)
            )

            if not np.any(mask):

                return 0.0

            valid_spectrum = (
                spectrum[mask]
            )

            valid_frequencies = (
                frequencies[mask]
            )

            if len(
                valid_spectrum
            ) == 0:

                return 0.0

            index = int(
                np.argmax(
                    valid_spectrum
                )
            )

            frequency = float(
                valid_frequencies[index]
            )

            return float(
                frequency
                *
                60.0
            )

        except Exception:

            return 0.0

    # =========================================================
    # SIGNAL QUALITY
    # =========================================================

    def _calculate_quality(
        self,
        filtered_signal,
    ):

        if np is None:

            return 0.0

        values = np.asarray(
            filtered_signal,
            dtype=float,
        )

        if len(values) < 20:

            return 0.0

        try:

            standard_deviation = float(
                np.std(values)
            )

            if not math.isfinite(
                standard_deviation
            ):

                return 0.0

            # -------------------------------------------------
            # Signal quality is based on usable temporal
            # variation, not medical validity.
            # -------------------------------------------------

            if standard_deviation < 0.02:

                return 0.0

            if standard_deviation >= 1.0:

                return 1.0

            return self._clamp(
                standard_deviation
            )

        except Exception:

            return 0.0

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        frame=None,
        timestamp=None,
        signal_value=None,
        input_available=True,
    ):

        if timestamp is None:

            timestamp = time.time()

        timestamp = self._safe_float(
            timestamp
        )

        roi_available = False

        # -----------------------------------------------------
        # CAMERA SIGNAL
        # -----------------------------------------------------

        if signal_value is None:

            if frame is not None:

                signal_value, roi_available = (
                    self.extract_roi_signal(
                        frame
                    )
                )

            else:

                signal_value = 0.0

        else:

            signal_value = self._safe_float(
                signal_value
            )

            roi_available = True

        # -----------------------------------------------------
        # NO INPUT
        # -----------------------------------------------------

        if not input_available:

            self.last_result = {

                "respiration_rate_bpm":
                    0.0,

                "raw_signal":
                    signal_value,

                "filtered_signal":
                    0.0,

                "signal_quality":
                    0.0,

                "reliability":
                    0.0,

                "state":
                    "NO_INPUT",

                "sample_count":
                    len(
                        self.signal_history
                    ),

                "signal_ready":
                    False,

                "roi_available":
                    roi_available,

                "method":
                    "VISUAL_INTENSITY_RESPIRATION",
            }

            return dict(
                self.last_result
            )

        # -----------------------------------------------------
        # STORE
        # -----------------------------------------------------

        self.signal_history.append(
            signal_value
        )

        self.time_history.append(
            timestamp
        )

        sample_count = len(
            self.signal_history
        )

        # -----------------------------------------------------
        # PREPARE SIGNAL
        # -----------------------------------------------------

        values = list(
            self.signal_history
        )

        filtered_signal = 0.0

        breathing_rate = 0.0

        quality = 0.0

        if sample_count >= 30:

            normalized = (
                self._normalize_signal(
                    values
                )
            )

            filtered = (
                self._bandpass_filter(
                    normalized
                )
            )

            smoothed = (
                self._smooth(
                    filtered
                )
            )

            if len(smoothed) > 0:

                filtered_signal = float(
                    smoothed[-1]
                )

            self.filtered_history.append(
                filtered_signal
            )

            quality = (
                self._calculate_quality(
                    smoothed
                )
            )

            if sample_count >= (
                self.minimum_samples
            ):

                breathing_rate = (
                    self._estimate_breathing_rate(
                        smoothed
                    )
                )

        # -----------------------------------------------------
        # READY
        # -----------------------------------------------------

        signal_ready = (
            sample_count
            >=
            self.minimum_samples
        )

        # -----------------------------------------------------
        # RELIABILITY
        # -----------------------------------------------------

        history_reliability = (
            self._clamp(
                sample_count
                /
                float(
                    self.minimum_samples
                )
            )
        )

        reliability = (
            history_reliability
            *
            0.50
            +
            quality
            *
            0.50
        )

        reliability = self._clamp(
            reliability
        )

        # -----------------------------------------------------
        # STATE
        # -----------------------------------------------------

        if not roi_available:

            state = "NO_SIGNAL"

        elif not signal_ready:

            state = "COLLECTING"

        elif breathing_rate <= 0:

            state = "LOW_SIGNAL"

        elif reliability < 0.30:

            state = "LOW_RELIABILITY"

        elif breathing_rate < 8:

            state = "LOW_RATE"

        elif breathing_rate <= 20:

            state = "NORMAL"

        elif breathing_rate <= 30:

            state = "ELEVATED"

        else:

            state = "HIGH_RATE"

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        self.last_result = {

            "respiration_rate_bpm":
                float(
                    breathing_rate
                ),

            "raw_signal":
                float(
                    signal_value
                ),

            "filtered_signal":
                float(
                    filtered_signal
                ),

            "signal_quality":
                float(
                    quality
                ),

            "reliability":
                float(
                    reliability
                ),

            "state":
                state,

            "sample_count":
                sample_count,

            "signal_ready":
                bool(
                    signal_ready
                ),

            "roi_available":
                bool(
                    roi_available
                ),

            "method":
                "VISUAL_INTENSITY_RESPIRATION",
        }

        return dict(
            self.last_result
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.signal_history.clear()

        self.time_history.clear()

        self.filtered_history.clear()

        self.last_result = {

            "respiration_rate_bpm":
                0.0,

            "raw_signal":
                0.0,

            "filtered_signal":
                0.0,

            "signal_quality":
                0.0,

            "reliability":
                0.0,

            "state":
                "NO_SIGNAL",

            "sample_count":
                0,

            "signal_ready":
                False,

            "roi_available":
                False,

            "method":
                "VISUAL_INTENSITY_RESPIRATION",
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):

        return {

            "history_seconds":
                self.history_seconds,

            "sample_rate":
                self.sample_rate,

            "min_breaths_per_minute":
                self.min_breaths_per_minute,

            "max_breaths_per_minute":
                self.max_breaths_per_minute,

            "samples":
                len(
                    self.signal_history
                ),

            "maximum_samples":
                self.maximum_samples,

            "minimum_samples":
                self.minimum_samples,

            "scipy_available":
                SCIPY_AVAILABLE,

            "opencv_available":
                OPENCV_AVAILABLE,

            "last_result":
                dict(
                    self.last_result
                ),
        }


# =============================================================
# SYNTHETIC SIGNAL
# =============================================================

def generate_synthetic_respiration(
    target_bpm=15.0,
    duration_seconds=20.0,
    sample_rate=30.0,
):

    if np is None:

        return []

    sample_count = int(
        duration_seconds
        *
        sample_rate
    )

    time_values = (
        np.arange(
            sample_count
        )
        /
        sample_rate
    )

    frequency = (
        target_bpm
        /
        60.0
    )

    signal = (
        np.sin(
            2.0
            *
            np.pi
            *
            frequency
            *
            time_values
        )
    )

    # Small harmonics/noise to make the
    # synthetic signal more realistic.

    signal += (
        0.10
        *
        np.sin(
            2.0
            *
            np.pi
            *
            frequency
            *
            2.0
            *
            time_values
        )
    )

    rng = np.random.default_rng(
        42
    )

    signal += (
        0.03
        *
        rng.normal(
            size=sample_count
        )
    )

    return signal


# =============================================================
# SELF TEST
# =============================================================

def self_test():

    print("=" * 70)

    print(
        "ADAPTIVE-DMS"
    )

    print(
        "NON-CONTACT RESPIRATION ESTIMATOR"
    )

    print(
        "v1.0 - STEP 8D SELF TEST"
    )

    print("=" * 70)

    print()

    print(
        f"OpenCV available: "
        f"{OPENCV_AVAILABLE}"
    )

    print(
        f"SciPy available: "
        f"{SCIPY_AVAILABLE}"
    )

    print()

    if np is None:

        print(
            "NumPy is not available."
        )

        print(
            "SELF TEST FAILED"
        )

        return

    # =========================================================
    # ESTIMATOR
    # =========================================================

    estimator = (
        NonContactRespirationEstimator(
            history_seconds=20.0,
            sample_rate=30.0,
            min_breaths_per_minute=6.0,
            max_breaths_per_minute=40.0,
            minimum_samples=180,
            smoothing_window=7,
        )
    )

    # =========================================================
    # SYNTHETIC TEST
    # =========================================================

    target_bpm = 15.0

    print(
        "Generating synthetic respiration signal..."
    )

    synthetic_signal = (
        generate_synthetic_respiration(
            target_bpm=target_bpm,
            duration_seconds=20.0,
            sample_rate=30.0,
        )
    )

    timestamp = 0.0

    result = None

    for value in synthetic_signal:

        timestamp += (
            1.0
            /
            30.0
        )

        result = estimator.update(
            signal_value=float(
                value
            ),
            timestamp=timestamp,
            input_available=True,
        )

    print()

    print(
        "Synthetic respiration:"
    )

    print(
        f"  Target BPM: "
        f"{target_bpm:.2f}"
    )

    print(
        f"  Estimated BPM: "
        f"{result['respiration_rate_bpm']:.2f}"
    )

    print(
        f"  Signal quality: "
        f"{result['signal_quality']:.3f}"
    )

    print(
        f"  Reliability: "
        f"{result['reliability']:.3f}"
    )

    print(
        f"  Samples: "
        f"{result['sample_count']}"
    )

    print(
        f"  State: "
        f"{result['state']}"
    )

    if result[
        "respiration_rate_bpm"
    ] > 0:

        absolute_error = abs(
            result[
                "respiration_rate_bpm"
            ]
            -
            target_bpm
        )

        print(
            f"  Absolute error: "
            f"{absolute_error:.2f} BPM"
        )

        if absolute_error <= 2.0:

            print(
                "  Result: PASS"
            )

        else:

            print(
                "  Result: REVIEW"
            )

    else:

        print(
            "  Result: FAIL"
        )

    print()

    # =========================================================
    # NO INPUT
    # =========================================================

    no_input = estimator.update(
        signal_value=0.0,
        timestamp=timestamp + 1.0,
        input_available=False,
    )

    print(
        "No input test:"
    )

    print(
        f"  State: "
        f"{no_input['state']}"
    )

    print(
        f"  Reliability: "
        f"{no_input['reliability']:.3f}"
    )

    if (
        no_input["state"]
        ==
        "NO_INPUT"
        and
        no_input["reliability"]
        ==
        0.0
    ):

        print(
            "  Result: PASS"
        )

    else:

        print(
            "  Result: FAIL"
        )

    print()

    # =========================================================
    # CAMERA ROI TEST
    # =========================================================

    print(
        "Testing camera-style ROI processing..."
    )

    if OPENCV_AVAILABLE:

        test_frame = np.zeros(
            (
                540,
                960,
                3,
            ),
            dtype=np.uint8,
        )

        test_frame[:, :, 1] = 120

        roi_signal, roi_available = (
            estimator.extract_roi_signal(
                test_frame
            )
        )

        print(
            f"  ROI: "
            f"{'PASS' if roi_available else 'FAIL'}"
        )

        print(
            f"  Visual signal: "
            f"{roi_signal:.2f}"
        )

    else:

        print(
            "  ROI: SKIPPED"
        )

    print()

    # =========================================================
    # STATUS
    # =========================================================

    status = (
        estimator.get_status()
    )

    print(
        "ESTIMATOR STATUS"
    )

    print(
        f"  History: "
        f"{status['history_seconds']:.1f} seconds"
    )

    print(
        f"  Sample rate: "
        f"{status['sample_rate']:.1f} Hz"
    )

    print(
        f"  Breathing range: "
        f"{status['min_breaths_per_minute']:.1f}"
        f" - "
        f"{status['max_breaths_per_minute']:.1f}"
        f" breaths/min"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This module provides an experimental"
    )

    print(
        "camera-based respiration estimate."
    )

    print(
        "It is NOT a medical-grade measurement."
    )

    print()

    print(
        "STEP 8D SELF TEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()