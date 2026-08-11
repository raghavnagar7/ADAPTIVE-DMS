"""
Signal reliability estimation for ADAPTIVE-DMS.

The purpose of this module is to estimate how trustworthy
each driver-state signal is at the current moment.

Signals:
    - EAR
    - MAR
    - PERCLOS
    - Blink
    - Microsleep
    - Head Pose
    - Gaze

Reliability values are normalized to [0, 1].
"""

import cv2
import numpy as np


class SignalReliabilityEstimator:
    """
    Estimate reliability of individual driver-state signals.

    This is the initial reliability layer.

    Later versions can be improved using:
        - learned reliability models
        - uncertainty estimation
        - user-specific calibration
        - temporal reliability
        - sensor disagreement
    """

    def __init__(
        self,
        min_brightness=40.0,
        max_brightness=220.0,
        min_blur_variance=20.0,
        perclos_min_samples=10,
    ):

        self.min_brightness = (
            min_brightness
        )

        self.max_brightness = (
            max_brightness
        )

        self.min_blur_variance = (
            min_blur_variance
        )

        self.perclos_min_samples = (
            perclos_min_samples
        )

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    @staticmethod
    def clamp(value, minimum=0.0, maximum=1.0):
        """Limit value to [0, 1]."""

        return max(
            minimum,
            min(maximum, float(value)),
        )

    # ---------------------------------------------------------
    # Image quality
    # ---------------------------------------------------------

    def image_quality(self, frame):
        """
        Estimate basic image quality.

        Uses:
            - brightness
            - sharpness

        Returns:
            brightness
            sharpness
            overall quality
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        brightness = float(
            np.mean(gray)
        )

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        # -----------------------------------------------------
        # Brightness reliability
        # -----------------------------------------------------

        if (
            brightness < self.min_brightness
            or brightness > self.max_brightness
        ):

            brightness_score = 0.3

        else:

            # Maximum reliability near the middle
            # of the acceptable brightness range.
            center = (
                self.min_brightness
                + self.max_brightness
            ) / 2.0

            half_range = (
                self.max_brightness
                - self.min_brightness
            ) / 2.0

            brightness_score = (
                1.0
                - abs(
                    brightness - center
                )
                / half_range
            )

        brightness_score = self.clamp(
            brightness_score
        )

        # -----------------------------------------------------
        # Sharpness reliability
        # -----------------------------------------------------

        if sharpness <= self.min_blur_variance:

            sharpness_score = 0.2

        else:

            sharpness_score = min(
                1.0,
                sharpness
                / (
                    self.min_blur_variance
                    * 10.0
                ),
            )

        # -----------------------------------------------------
        # Overall image quality
        # -----------------------------------------------------

        quality = (
            0.5 * brightness_score
            + 0.5 * sharpness_score
        )

        return {
            "brightness": brightness,
            "sharpness": sharpness,
            "brightness_reliability":
                brightness_score,
            "sharpness_reliability":
                sharpness_score,
            "image_quality":
                self.clamp(quality),
        }

    # ---------------------------------------------------------
    # EAR reliability
    # ---------------------------------------------------------

    def ear_reliability(
        self,
        ear,
        image_quality,
        face_detected=True,
    ):
        """
        Estimate EAR reliability.
        """

        if not face_detected:
            return 0.0

        if ear <= 0.05 or ear >= 0.80:
            signal_score = 0.2

        else:
            signal_score = 1.0

        return self.clamp(
            signal_score
            * image_quality
        )

    # ---------------------------------------------------------
    # MAR reliability
    # ---------------------------------------------------------

    def mar_reliability(
        self,
        mar,
        image_quality,
        face_detected=True,
    ):
        """Estimate MAR reliability."""

        if not face_detected:
            return 0.0

        if mar < 0.0 or mar > 2.0:
            signal_score = 0.2

        else:
            signal_score = 1.0

        return self.clamp(
            signal_score
            * image_quality
        )

    # ---------------------------------------------------------
    # PERCLOS reliability
    # ---------------------------------------------------------

    def perclos_reliability(
        self,
        sample_count,
        image_quality,
    ):
        """
        PERCLOS reliability increases as more observations
        become available in the temporal window.
        """

        if sample_count <= 0:
            temporal_score = 0.0

        else:

            temporal_score = min(
                1.0,
                sample_count
                / self.perclos_min_samples,
            )

        return self.clamp(
            temporal_score
            * image_quality
        )

    # ---------------------------------------------------------
    # Blink reliability
    # ---------------------------------------------------------

    def blink_reliability(
        self,
        blink_duration,
        image_quality,
        face_detected=True,
    ):
        """Estimate blink signal reliability."""

        if not face_detected:
            return 0.0

        if blink_duration <= 0:
            signal_score = 0.8

        elif blink_duration <= 1.0:
            signal_score = 1.0

        else:
            signal_score = 0.5

        return self.clamp(
            signal_score
            * image_quality
        )

    # ---------------------------------------------------------
    # Microsleep reliability
    # ---------------------------------------------------------

    def microsleep_reliability(
        self,
        eye_closure_duration,
        image_quality,
        face_detected=True,
    ):
        """Estimate microsleep signal reliability."""

        if not face_detected:
            return 0.0

        if eye_closure_duration < 0:
            signal_score = 0.2

        else:
            signal_score = 1.0

        return self.clamp(
            signal_score
            * image_quality
        )

    # ---------------------------------------------------------
    # Head pose reliability
    # ---------------------------------------------------------

    def head_pose_reliability(
        self,
        pitch,
        yaw,
        roll,
        image_quality,
        valid=True,
    ):
        """
        Estimate head-pose reliability.

        Extreme angles are treated as less reliable because
        landmark geometry can become unstable.
        """

        if not valid:
            return 0.0

        pitch_score = (
            1.0
            if abs(pitch) <= 60
            else 0.3
        )

        yaw_score = (
            1.0
            if abs(yaw) <= 70
            else 0.3
        )

        roll_score = (
            1.0
            if abs(roll) <= 70
            else 0.3
        )

        pose_score = (
            pitch_score
            + yaw_score
            + roll_score
        ) / 3.0

        return self.clamp(
            pose_score
            * image_quality
        )

    # ---------------------------------------------------------
    # Gaze reliability
    # ---------------------------------------------------------

    def gaze_reliability(
        self,
        horizontal_ratio,
        vertical_ratio,
        image_quality,
        face_detected=True,
    ):
        """Estimate gaze reliability."""

        if not face_detected:
            return 0.0

        if not (
            0.0 <= horizontal_ratio <= 1.0
            and 0.0 <= vertical_ratio <= 1.0
        ):
            signal_score = 0.2

        else:
            signal_score = 1.0

        return self.clamp(
            signal_score
            * image_quality
        )

    # ---------------------------------------------------------
    # Complete reliability vector
    # ---------------------------------------------------------

    def calculate(
        self,
        frame,
        driver_values,
        pose_values,
        gaze_values,
        perclos_sample_count,
        face_detected=True,
    ):
        """
        Calculate reliability for all available signals.

        Returns a dictionary containing individual
        reliability scores and image-quality information.
        """

        quality_values = self.image_quality(
            frame
        )

        image_quality = (
            quality_values["image_quality"]
        )

        ear = driver_values["ear"]

        mar = driver_values["mar"]

        blink_duration = (
            driver_values["blink_duration"]
        )

        eye_closure_duration = (
            driver_values["microsleep_duration"]
        )

        ear_reliability = self.ear_reliability(
            ear,
            image_quality,
            face_detected,
        )

        mar_reliability = self.mar_reliability(
            mar,
            image_quality,
            face_detected,
        )

        perclos_reliability = (
            self.perclos_reliability(
                perclos_sample_count,
                image_quality,
            )
        )

        blink_reliability = (
            self.blink_reliability(
                blink_duration,
                image_quality,
                face_detected,
            )
        )

        microsleep_reliability = (
            self.microsleep_reliability(
                eye_closure_duration,
                image_quality,
                face_detected,
            )
        )

        head_pose_reliability = (
            self.head_pose_reliability(
                pose_values["pitch"],
                pose_values["yaw"],
                pose_values["roll"],
                image_quality,
                pose_values["valid"],
            )
        )

        gaze_reliability = (
            self.gaze_reliability(
                gaze_values["horizontal_ratio"],
                gaze_values["vertical_ratio"],
                image_quality,
                face_detected,
            )
        )

        reliability_vector = {
            "EAR":
                ear_reliability,

            "MAR":
                mar_reliability,

            "PERCLOS":
                perclos_reliability,

            "Blink":
                blink_reliability,

            "Microsleep":
                microsleep_reliability,

            "HeadPose":
                head_pose_reliability,

            "Gaze":
                gaze_reliability,
        }

        # Average reliability of available signals.
        overall_reliability = (
            sum(
                reliability_vector.values()
            )
            / len(reliability_vector)
        )

        return {
            "image_quality":
                image_quality,

            "brightness":
                quality_values["brightness"],

            "sharpness":
                quality_values["sharpness"],

            "reliability_vector":
                reliability_vector,

            "overall_reliability":
                self.clamp(
                    overall_reliability
                ),
        }