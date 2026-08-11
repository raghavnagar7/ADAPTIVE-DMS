"""
Driver-state metrics for ADAPTIVE-DMS.

Features:
    - Eye Aspect Ratio (EAR)
    - Mouth Aspect Ratio (MAR)
    - PERCLOS
    - Blink detection
    - Microsleep detection
"""

import math
import time
from collections import deque


class DriverMetrics:
    """Calculates and stores driver-state metrics."""

    # MediaPipe Face Mesh landmarks
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Mouth landmarks
    MOUTH = [61, 291, 13, 14, 78, 308]

    def __init__(
        self,
        ear_threshold=0.21,
        mar_threshold=0.60,
        perclos_window_seconds=30,
        microsleep_threshold=1.5,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.perclos_window_seconds = perclos_window_seconds
        self.microsleep_threshold = microsleep_threshold

        # PERCLOS history
        self.eye_history = deque()

        # Blink tracking
        self.blink_count = 0
        self.previous_eyes_closed = False
        self.current_blink_start = None
        self.last_blink_duration = 0.0

        # Microsleep tracking
        self.eye_closed_start = None
        self.microsleep_duration = 0.0
        self.microsleep = False

    @staticmethod
    def distance(point_a, point_b):
        """Calculate Euclidean distance between two landmarks."""

        return math.sqrt(
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
        )

    # ---------------------------------------------------------
    # EAR
    # ---------------------------------------------------------

    def eye_aspect_ratio(self, landmarks, eye_indices):
        """
        Calculate Eye Aspect Ratio (EAR).

        EAR = average vertical eye opening / horizontal eye width
        """

        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        vertical_1 = self.distance(p2, p6)
        vertical_2 = self.distance(p3, p5)

        horizontal = self.distance(p1, p4)

        if horizontal == 0:
            return 0.0

        ear = (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

        return ear

    def calculate_ear(self, landmarks):
        """Calculate average EAR from both eyes."""

        left_ear = self.eye_aspect_ratio(
            landmarks,
            self.LEFT_EYE,
        )

        right_ear = self.eye_aspect_ratio(
            landmarks,
            self.RIGHT_EYE,
        )

        return (
            left_ear + right_ear
        ) / 2.0

    # ---------------------------------------------------------
    # MAR
    # ---------------------------------------------------------

    def calculate_mar(self, landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR).

        MAR = mouth height / mouth width
        """

        left = landmarks[self.MOUTH[0]]
        right = landmarks[self.MOUTH[1]]

        top = landmarks[self.MOUTH[2]]
        bottom = landmarks[self.MOUTH[3]]

        mouth_width = self.distance(
            left,
            right,
        )

        mouth_height = self.distance(
            top,
            bottom,
        )

        if mouth_width == 0:
            return 0.0

        return mouth_height / mouth_width

    # ---------------------------------------------------------
    # PERCLOS
    # ---------------------------------------------------------

    def update_perclos(
        self,
        timestamp,
        eyes_closed,
    ):
        """
        Calculate PERCLOS over a sliding time window.

        PERCLOS =
            proportion of observed frames where
            the eyes are considered closed.
        """

        self.eye_history.append(
            (
                timestamp,
                bool(eyes_closed),
            )
        )

        cutoff = (
            timestamp
            - self.perclos_window_seconds
        )

        # Remove old samples.
        while self.eye_history:

            oldest_timestamp = (
                self.eye_history[0][0]
            )

            if oldest_timestamp >= cutoff:
                break

            self.eye_history.popleft()

        if not self.eye_history:
            return 0.0

        closed_count = sum(
            1
            for _, closed
            in self.eye_history
            if closed
        )

        return (
            closed_count
            / len(self.eye_history)
        )

    # ---------------------------------------------------------
    # BLINK
    # ---------------------------------------------------------

    def update_blink(
        self,
        timestamp,
        eyes_closed,
    ):
        """
        Detect a blink using:

            Open → Closed → Open
        """

        eyes_closed = bool(
            eyes_closed
        )

        # Eyes just closed
        if (
            eyes_closed
            and not self.previous_eyes_closed
        ):

            self.current_blink_start = (
                timestamp
            )

        # Eyes just opened
        elif (
            not eyes_closed
            and self.previous_eyes_closed
        ):

            if (
                self.current_blink_start
                is not None
            ):

                duration = (
                    timestamp
                    - self.current_blink_start
                )

                self.last_blink_duration = (
                    duration
                )

                # Normal blink threshold
                if duration <= 0.8:
                    self.blink_count += 1

            self.current_blink_start = None

        self.previous_eyes_closed = (
            eyes_closed
        )

    # ---------------------------------------------------------
    # MICROSLEEP
    # ---------------------------------------------------------

    def update_microsleep(
        self,
        timestamp,
        eyes_closed,
    ):
        """
        Detect prolonged eye closure.

        Microsleep is triggered when the eyes
        remain closed for >= microsleep_threshold.
        """

        eyes_closed = bool(
            eyes_closed
        )

        if eyes_closed:

            # Start timing eye closure
            if self.eye_closed_start is None:

                self.eye_closed_start = (
                    timestamp
                )

            # Calculate duration
            self.microsleep_duration = (
                timestamp
                - self.eye_closed_start
            )

        else:

            # Reset when eyes open
            self.eye_closed_start = None

            self.microsleep_duration = 0.0

        self.microsleep = (
            self.microsleep_duration
            >= self.microsleep_threshold
        )

        return (
            self.microsleep_duration,
            self.microsleep,
        )

    # ---------------------------------------------------------
    # ALL METRICS
    # ---------------------------------------------------------

    def calculate(
        self,
        landmarks,
        timestamp=None,
    ):
        """Calculate all driver-state metrics."""

        if timestamp is None:
            timestamp = time.time()

        # EAR
        ear = self.calculate_ear(
            landmarks
        )

        # MAR
        mar = self.calculate_mar(
            landmarks
        )

        # Eye state
        eyes_closed = (
            ear < self.ear_threshold
        )

        # Yawning
        yawning = (
            mar > self.mar_threshold
        )

        # PERCLOS
        perclos = self.update_perclos(
            timestamp,
            eyes_closed,
        )

        # Blink
        self.update_blink(
            timestamp,
            eyes_closed,
        )

        # Microsleep
        (
            microsleep_duration,
            microsleep,
        ) = self.update_microsleep(
            timestamp,
            eyes_closed,
        )

        return {
            "ear": ear,
            "mar": mar,
            "perclos": perclos,
            "eyes_closed": eyes_closed,
            "yawning": yawning,
            "blink_count": self.blink_count,
            "blink_duration": self.last_blink_duration,
            "microsleep_duration": microsleep_duration,
            "microsleep": microsleep,
        }