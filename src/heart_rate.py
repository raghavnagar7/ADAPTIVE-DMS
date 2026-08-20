"""
=============================================================
ADAPTIVE-DMS
=============================================================

Non-Contact Heart Rate Estimation Module

Version:
    v1.0 - STEP 8A

Purpose:
    Estimate heart rate from webcam video using a simple
    remote photoplethysmography (rPPG) approach.

Method:
    - Extract skin-region ROI
    - Use green-channel intensity
    - Maintain temporal signal history
    - Detrend signal
    - Apply band-pass filtering
    - Estimate dominant frequency
    - Convert frequency to BPM
    - Calculate signal quality / reliability

Important:
    This is an experimental non-contact estimate.
    It is NOT a medical-grade heart-rate measurement.

Inputs:
    - Video frame
    - Optional face bounding box

Outputs:
    - heart_rate_bpm
    - raw_signal
    - filtered_signal
    - signal_quality
    - reliability
    - state
    - sample_count
    - signal_ready

=============================================================
"""

from collections import deque
import math
import statistics
import time

import numpy as np


# =============================================================
# OPTIONAL SCIPY
# =============================================================

try:

    from scipy.signal import (
        butter,
        filtfilt,
        detrend,
    )

    SCIPY_AVAILABLE = True

except Exception:

    SCIPY_AVAILABLE = False


# =============================================================
# OPENCV
# =============================================================

try:

    import cv2

    OPENCV_AVAILABLE = True

except Exception:

    OPENCV_AVAILABLE = False


# =============================================================
# HEART RATE ESTIMATOR
# =============================================================

class NonContactHeartRateEstimator:
    """
    Estimate heart rate from webcam frames.

    Typical physiological range used:

        45 BPM -> lower bound
        180 BPM -> upper bound

    The estimator requires a sufficient temporal history
    before producing a reliable result.
    """

    def __init__(
        self,
        history_seconds=12.0,
        sample_rate=30.0,
        min_bpm=45.0,
        max_bpm=180.0,
        roi_scale=0.35,
        minimum_samples=120,
        smoothing_window=5,
    ):

        self.history_seconds = float(
            history_seconds
        )

        self.sample_rate = float(
            sample_rate
        )

        self.min_bpm = float(
            min_bpm
        )

        self.max_bpm = float(
            max_bpm
        )

        self.roi_scale = float(
            roi_scale
        )

        self.minimum_samples = int(
            minimum_samples
        )

        self.smoothing_window = int(
            smoothing_window
        )

        self.max_samples = max(
            self.minimum_samples,
            int(
                self.history_seconds
                * self.sample_rate
            ),
        )

        # -----------------------------------------------------
        # Signal history
        # -----------------------------------------------------

        self.signal_history = deque(
            maxlen=self.max_samples
        )

        self.time_history = deque(
            maxlen=self.max_samples
        )

        self.raw_signal_history = deque(
            maxlen=self.max_samples
        )

        # -----------------------------------------------------
        # Last state
        # -----------------------------------------------------

        self.last_result = {

            "heart_rate_bpm": 0.0,

            "raw_signal": 0.0,

            "filtered_signal": 0.0,

            "signal_quality": 0.0,

            "reliability": 0.0,

            "state": "NO_SIGNAL",

            "sample_count": 0,

            "signal_ready": False,

            "roi_available": False,

            "method": "GREEN_CHANNEL_RPPG",
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

            result = float(value)

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

        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    # =========================================================
    # FACE ROI
    # =========================================================

    def _extract_roi(
        self,
        frame,
        face_bbox=None,
    ):
        """
        Extract an approximate forehead/upper-face ROI.

        face_bbox:
            (x, y, width, height)

        If no face bounding box is supplied, use a central
        region as a fallback.
        """

        if frame is None:

            return None

        if not hasattr(
            frame,
            "shape",
        ):

            return None

        height, width = (
            frame.shape[:2]
        )

        # -----------------------------------------------------
        # Face detected
        # -----------------------------------------------------

        if face_bbox is not None:

            try:

                x, y, w, h = (
                    face_bbox
                )

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                if w <= 0 or h <= 0:

                    return None

                # ---------------------------------------------
                # Upper-center face region
                # ---------------------------------------------

                roi_x1 = int(
                    x
                    + 0.30 * w
                )

                roi_x2 = int(
                    x
                    + 0.70 * w
                )

                roi_y1 = int(
                    y
                    + 0.15 * h
                )

                roi_y2 = int(
                    y
                    + 0.35 * h
                )

                roi_x1 = max(
                    0,
                    min(
                        width - 1,
                        roi_x1,
                    ),
                )

                roi_x2 = max(
                    0,
                    min(
                        width,
                        roi_x2,
                    ),
                )

                roi_y1 = max(
                    0,
                    min(
                        height - 1,
                        roi_y1,
                    ),
                )

                roi_y2 = max(
                    0,
                    min(
                        height,
                        roi_y2,
                    ),
                )

                if (
                    roi_x2 <= roi_x1
                    or roi_y2 <= roi_y1
                ):

                    return None

                roi = frame[
                    roi_y1:roi_y2,
                    roi_x1:roi_x2,
                ]

                if roi.size == 0:

                    return None

                return roi

            except Exception:

                return None

        # -----------------------------------------------------
        # Center fallback ROI
        # -----------------------------------------------------

        x1 = int(
            width * 0.35
        )

        x2 = int(
            width * 0.65
        )

        y1 = int(
            height * 0.20
        )

        y2 = int(
            height * 0.45
        )

        roi = frame[
            y1:y2,
            x1:x2,
        ]

        if roi.size == 0:

            return None

        return roi

    # =========================================================
    # EXTRACT GREEN SIGNAL
    # =========================================================

    def _extract_green_signal(
        self,
        roi,
    ):
        """
        Calculate mean green-channel intensity.

        OpenCV frame convention:

            BGR

        Therefore:

            channel 1 = Green
        """

        if roi is None:

            return None

        try:

            if len(
                roi.shape
            ) < 3:

                return None

            green_channel = roi[
                :,
                :,
                1,
            ]

            value = float(
                np.mean(
                    green_channel
                )
            )

            if not math.isfinite(
                value
            ):

                return None

            return value

        except Exception:

            return None

    # =========================================================
    # RESAMPLE SIGNAL
    # =========================================================

    def _resample_signal(
        self,
        values,
        timestamps,
    ):
        """
        Interpolate signal to a uniform sampling rate.
        """

        values = np.asarray(
            values,
            dtype=float,
        )

        timestamps = np.asarray(
            timestamps,
            dtype=float,
        )

        if len(values) < 3:

            return None

        if len(timestamps) != len(
            values
        ):

            return None

        # -----------------------------------------------------
        # Remove invalid values
        # -----------------------------------------------------

        valid = (
            np.isfinite(values)
            & np.isfinite(timestamps)
        )

        values = values[
            valid
        ]

        timestamps = timestamps[
            valid
        ]

        if len(values) < 3:

            return None

        # -----------------------------------------------------
        # Remove duplicate timestamps
        # -----------------------------------------------------

        unique_times, indices = (
            np.unique(
                timestamps,
                return_index=True,
            )
        )

        values = values[
            indices
        ]

        timestamps = unique_times

        if len(values) < 3:

            return None

        duration = (
            timestamps[-1]
            - timestamps[0]
        )

        if duration <= 0:

            return None

        # -----------------------------------------------------
        # Uniform timeline
        # -----------------------------------------------------

        target_count = max(
            3,
            int(
                duration
                * self.sample_rate
            ),
        )

        uniform_times = np.linspace(
            timestamps[0],
            timestamps[-1],
            target_count,
        )

        interpolated = np.interp(
            uniform_times,
            timestamps,
            values,
        )

        return interpolated

    # =========================================================
    # FILTER
    # =========================================================

    def _bandpass_filter(
        self,
        signal,
    ):
        """
        Keep approximately 45-180 BPM.

        Frequencies:

            45 BPM  = 0.75 Hz
            180 BPM = 3.00 Hz
        """

        signal = np.asarray(
            signal,
            dtype=float,
        )

        if len(signal) < 30:

            return signal

        # -----------------------------------------------------
        # Detrend
        # -----------------------------------------------------

        if SCIPY_AVAILABLE:

            try:

                signal = detrend(
                    signal
                )

            except Exception:

                signal = (
                    signal
                    - np.mean(signal)
                )

        else:

            signal = (
                signal
                - np.mean(signal)
            )

        # -----------------------------------------------------
        # Butterworth bandpass
        # -----------------------------------------------------

        if SCIPY_AVAILABLE:

            try:

                nyquist = (
                    self.sample_rate
                    / 2.0
                )

                low = (
                    self.min_bpm
                    / 60.0
                    / nyquist
                )

                high = (
                    self.max_bpm
                    / 60.0
                    / nyquist
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
                        0.999,
                        high,
                    ),
                )

                b, a = butter(
                    3,
                    [
                        low,
                        high,
                    ],
                    btype="band",
                )

                filtered = filtfilt(
                    b,
                    a,
                    signal,
                )

                return filtered

            except Exception:

                pass

        # -----------------------------------------------------
        # Fallback
        # -----------------------------------------------------

        return signal

    # =========================================================
    # HEART RATE FROM FFT
    # =========================================================

    def _estimate_from_fft(
        self,
        filtered_signal,
    ):
        """
        Estimate dominant frequency using FFT.
        """

        signal = np.asarray(
            filtered_signal,
            dtype=float,
        )

        if len(signal) < 30:

            return 0.0, 0.0

        # -----------------------------------------------------
        # Remove DC
        # -----------------------------------------------------

        signal = (
            signal
            - np.mean(signal)
        )

        # -----------------------------------------------------
        # Window
        # -----------------------------------------------------

        window = np.hanning(
            len(signal)
        )

        windowed = (
            signal
            * window
        )

        # -----------------------------------------------------
        # FFT
        # -----------------------------------------------------

        spectrum = np.fft.rfft(
            windowed
        )

        frequencies = (
            np.fft.rfftfreq(
                len(signal),
                d=1.0
                / self.sample_rate,
            )
        )

        magnitudes = np.abs(
            spectrum
        )

        # -----------------------------------------------------
        # Physiological frequency range
        # -----------------------------------------------------

        minimum_frequency = (
            self.min_bpm
            / 60.0
        )

        maximum_frequency = (
            self.max_bpm
            / 60.0
        )

        mask = (
            (frequencies >= minimum_frequency)
            &
            (frequencies <= maximum_frequency)
        )

        if not np.any(mask):

            return 0.0, 0.0

        valid_frequencies = (
            frequencies[mask]
        )

        valid_magnitudes = (
            magnitudes[mask]
        )

        if len(
            valid_magnitudes
        ) == 0:

            return 0.0, 0.0

        # -----------------------------------------------------
        # Dominant frequency
        # -----------------------------------------------------

        peak_index = int(
            np.argmax(
                valid_magnitudes
            )
        )

        dominant_frequency = (
            valid_frequencies[
                peak_index
            ]
        )

        heart_rate = (
            dominant_frequency
            * 60.0
        )

        # -----------------------------------------------------
        # Spectral quality
        # -----------------------------------------------------

        total_power = float(
            np.sum(
                valid_magnitudes
            )
        )

        peak_power = float(
            valid_magnitudes[
                peak_index
            ]
        )

        if total_power <= 1e-9:

            quality = 0.0

        else:

            quality = (
                peak_power
                / total_power
            )

        # Normalize quality.

        quality = self._clamp(
            quality * 5.0
        )

        return (
            float(heart_rate),
            float(quality),
        )

    # =========================================================
    # SIGNAL QUALITY
    # =========================================================

    def _calculate_signal_quality(
        self,
        signal,
    ):
        """
        Calculate basic signal quality based on variation.
        """

        signal = np.asarray(
            signal,
            dtype=float,
        )

        if len(signal) < 5:

            return 0.0

        if not np.all(
            np.isfinite(signal)
        ):

            return 0.0

        mean_value = float(
            np.mean(signal)
        )

        standard_deviation = float(
            np.std(signal)
        )

        if abs(mean_value) < 1e-9:

            return 0.0

        coefficient = (
            standard_deviation
            / abs(mean_value)
        )

        # Very tiny variations often indicate a flat
        # or unusable signal.
        if coefficient < 0.00005:

            return 0.0

        # Avoid treating extreme noise as excellent quality.
        if coefficient > 0.20:

            return 0.20

        quality = (
            coefficient
            / 0.02
        )

        return self._clamp(
            quality
        )

    # =========================================================
    # STATE
    # =========================================================

    def _classify_state(
        self,
        heart_rate,
        reliability,
    ):

        if reliability < 0.30:

            return "LOW_SIGNAL"

        if heart_rate <= 0:

            return "NO_SIGNAL"

        if heart_rate < 50:

            return "LOW"

        if heart_rate <= 100:

            return "NORMAL"

        if heart_rate <= 120:

            return "ELEVATED"

        return "HIGH"

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        frame,
        face_bbox=None,
        timestamp=None,
    ):
        """
        Process one camera frame.

        Parameters
        ----------
        frame:
            OpenCV BGR frame.

        face_bbox:
            Optional tuple:
                (x, y, width, height)

        timestamp:
            Timestamp in seconds.
        """

        if timestamp is None:

            timestamp = time.time()

        timestamp = self._safe_float(
            timestamp
        )

        # -----------------------------------------------------
        # ROI
        # -----------------------------------------------------

        roi = self._extract_roi(
            frame,
            face_bbox,
        )

        if roi is None:

            self.last_result = {

                "heart_rate_bpm": 0.0,

                "raw_signal": 0.0,

                "filtered_signal": 0.0,

                "signal_quality": 0.0,

                "reliability": 0.0,

                "state": "NO_SIGNAL",

                "sample_count": len(
                    self.signal_history
                ),

                "signal_ready": False,

                "roi_available": False,

                "method": (
                    "GREEN_CHANNEL_RPPG"
                ),
            }

            return dict(
                self.last_result
            )

        # -----------------------------------------------------
        # Green signal
        # -----------------------------------------------------

        green_value = (
            self._extract_green_signal(
                roi
            )
        )

        if green_value is None:

            self.last_result = {

                "heart_rate_bpm": 0.0,

                "raw_signal": 0.0,

                "filtered_signal": 0.0,

                "signal_quality": 0.0,

                "reliability": 0.0,

                "state": "NO_SIGNAL",

                "sample_count": len(
                    self.signal_history
                ),

                "signal_ready": False,

                "roi_available": False,

                "method": (
                    "GREEN_CHANNEL_RPPG"
                ),
            }

            return dict(
                self.last_result
            )

        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------

        self.signal_history.append(
            green_value
        )

        self.raw_signal_history.append(
            green_value
        )

        self.time_history.append(
            timestamp
        )

        sample_count = len(
            self.signal_history
        )

        # -----------------------------------------------------
        # Not enough samples
        # -----------------------------------------------------

        if sample_count < self.minimum_samples:

            progress = (
                sample_count
                / max(
                    1,
                    self.minimum_samples,
                )
            )

            reliability = self._clamp(
                progress * 0.30
            )

            self.last_result = {

                "heart_rate_bpm": 0.0,

                "raw_signal": float(
                    green_value
                ),

                "filtered_signal": 0.0,

                "signal_quality": 0.0,

                "reliability": float(
                    reliability
                ),

                "state": "COLLECTING",

                "sample_count": sample_count,

                "signal_ready": False,

                "roi_available": True,

                "method": (
                    "GREEN_CHANNEL_RPPG"
                ),
            }

            return dict(
                self.last_result
            )

        # =====================================================
        # SIGNAL PROCESSING
        # =====================================================

        values = list(
            self.signal_history
        )

        timestamps = list(
            self.time_history
        )

        resampled = (
            self._resample_signal(
                values,
                timestamps,
            )
        )

        if resampled is None:

            self.last_result = {

                "heart_rate_bpm": 0.0,

                "raw_signal": float(
                    green_value
                ),

                "filtered_signal": 0.0,

                "signal_quality": 0.0,

                "reliability": 0.0,

                "state": "LOW_SIGNAL",

                "sample_count": sample_count,

                "signal_ready": False,

                "roi_available": True,

                "method": (
                    "GREEN_CHANNEL_RPPG"
                ),
            }

            return dict(
                self.last_result
            )

        # -----------------------------------------------------
        # Filter
        # -----------------------------------------------------

        filtered = (
            self._bandpass_filter(
                resampled
            )
        )

        # -----------------------------------------------------
        # FFT
        # -----------------------------------------------------

        heart_rate, spectral_quality = (
            self._estimate_from_fft(
                filtered
            )
        )

        # -----------------------------------------------------
        # Signal quality
        # -----------------------------------------------------

        signal_quality = (
            self._calculate_signal_quality(
                resampled
            )
        )

        # -----------------------------------------------------
        # Combine quality measures
        # -----------------------------------------------------

        combined_quality = self._clamp(
            signal_quality * 0.40
            + spectral_quality * 0.60
        )

        # -----------------------------------------------------
        # Reliability
        # -----------------------------------------------------

        history_reliability = self._clamp(
            sample_count
            / max(
                1.0,
                self.max_samples,
            )
        )

        reliability = self._clamp(
            history_reliability * 0.30
            + combined_quality * 0.70
        )

        # -----------------------------------------------------
        # Low-quality result
        # -----------------------------------------------------

        if combined_quality < 0.10:

            displayed_heart_rate = 0.0

        else:

            displayed_heart_rate = (
                heart_rate
            )

        # -----------------------------------------------------
        # State
        # -----------------------------------------------------

        state = (
            self._classify_state(
                displayed_heart_rate,
                reliability,
            )
        )

        # -----------------------------------------------------
        # Save result
        # -----------------------------------------------------

        self.last_result = {

            "heart_rate_bpm": float(
                displayed_heart_rate
            ),

            "raw_signal": float(
                green_value
            ),

            "filtered_signal": float(
                filtered[-1]
                if len(filtered) > 0
                else 0.0
            ),

            "signal_quality": float(
                combined_quality
            ),

            "reliability": float(
                reliability
            ),

            "state": state,

            "sample_count": sample_count,

            "signal_ready": True,

            "roi_available": True,

            "method": (
                "GREEN_CHANNEL_RPPG"
            ),
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

        self.raw_signal_history.clear()

        self.last_result = {

            "heart_rate_bpm": 0.0,

            "raw_signal": 0.0,

            "filtered_signal": 0.0,

            "signal_quality": 0.0,

            "reliability": 0.0,

            "state": "NO_SIGNAL",

            "sample_count": 0,

            "signal_ready": False,

            "roi_available": False,

            "method": (
                "GREEN_CHANNEL_RPPG"
            ),
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

            "min_bpm":
                self.min_bpm,

            "max_bpm":
                self.max_bpm,

            "samples":
                len(
                    self.signal_history
                ),

            "maximum_samples":
                self.max_samples,

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
# SYNTHETIC SELF TEST
# =============================================================

def self_test():

    print("=" * 70)

    print(
        "ADAPTIVE-DMS"
    )

    print(
        "NON-CONTACT HEART RATE ESTIMATOR"
    )

    print(
        "v1.0 - STEP 8A SELF TEST"
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

    # =========================================================
    # CREATE ESTIMATOR
    # =========================================================

    estimator = (
        NonContactHeartRateEstimator(
            history_seconds=12.0,
            sample_rate=30.0,
            min_bpm=45.0,
            max_bpm=180.0,
            minimum_samples=120,
        )
    )

    # =========================================================
    # SYNTHETIC PPG
    # =========================================================

    print(
        "Generating synthetic rPPG signal..."
    )

    target_bpm = 72.0

    target_frequency = (
        target_bpm
        / 60.0
    )

    duration = 12.0

    sample_rate = 30.0

    sample_count = int(
        duration
        * sample_rate
    )

    timestamps = np.arange(
        sample_count
    ) / sample_rate

    # ---------------------------------------------------------
    # Synthetic green-channel signal
    # ---------------------------------------------------------

    synthetic_signal = (
        100.0
        + 2.0
        * np.sin(
            2.0
            * np.pi
            * target_frequency
            * timestamps
        )
        + 0.25
        * np.random.default_rng(
            42
        ).normal(
            0.0,
            1.0,
            sample_count,
        )
    )

    # =========================================================
    # DIRECT FFT TEST
    # =========================================================

    filtered = (
        estimator._bandpass_filter(
            synthetic_signal
        )
    )

    estimated_bpm, quality = (
        estimator._estimate_from_fft(
            filtered
        )
    )

    print()

    print(
        "Synthetic heart rate:"
    )

    print(
        f"  Target BPM: "
        f"{target_bpm:.2f}"
    )

    print(
        f"  Estimated BPM: "
        f"{estimated_bpm:.2f}"
    )

    print(
        f"  Spectral quality: "
        f"{quality:.3f}"
    )

    # =========================================================
    # ACCURACY CHECK
    # =========================================================

    error = abs(
        estimated_bpm
        - target_bpm
    )

    print(
        f"  Absolute error: "
        f"{error:.2f} BPM"
    )

    if error <= 8.0:

        print(
            "  Result: PASS"
        )

    else:

        print(
            "  Result: CHECK"
        )

    # =========================================================
    # IMAGE TEST
    # =========================================================

    if OPENCV_AVAILABLE:

        print()

        print(
            "Testing camera-style ROI processing..."
        )

        # -----------------------------------------------------
        # Create synthetic BGR frame
        # -----------------------------------------------------

        frame = np.zeros(
            (
                480,
                640,
                3,
            ),
            dtype=np.uint8,
        )

        # -----------------------------------------------------
        # Add synthetic skin-like region
        # -----------------------------------------------------

        frame[
            100:250,
            200:440,
            0
        ] = 120

        frame[
            100:250,
            200:440,
            1
        ] = 150

        frame[
            100:250,
            200:440,
            2
        ] = 170

        roi = (
            estimator._extract_roi(
                frame
            )
        )

        if roi is not None:

            green_value = (
                estimator._extract_green_signal(
                    roi
                )
            )

            print(
                f"  ROI: PASS"
            )

            print(
                f"  Green signal: "
                f"{green_value:.2f}"
            )

        else:

            print(
                "  ROI: FAIL"
            )

    # =========================================================
    # STATUS
    # =========================================================

    print()

    status = (
        estimator.get_status()
    )

    print(
        "ESTIMATOR STATUS"
    )

    print(
        f"  History: "
        f"{status['history_seconds']} seconds"
    )

    print(
        f"  Sample rate: "
        f"{status['sample_rate']} Hz"
    )

    print(
        f"  BPM range: "
        f"{status['min_bpm']} - "
        f"{status['max_bpm']}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This module provides an experimental"
    )

    print(
        "camera-based heart-rate estimate."
    )

    print(
        "It is NOT a medical-grade measurement."
    )

    print()

    print(
        "STEP 8A SELF TEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()