"""
Driver-state metrics for ADAPTIVE-DMS.

Current metrics:
    - Eye Aspect Ratio (EAR)
    - Mouth Aspect Ratio (MAR)
    - PERCLOS
    - Blink detection
"""

import math
import time
from collections import deque


class DriverMetrics:
    """Calculates and stores driver-state metrics."""

    # MediaPipe Face Mesh landmark indices.
    # These are the standard landmarks used for the first version.
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    MOUTH = [61, 291, 13, 14, 78, 308]

    def __init__(
        self,
        ear_threshold=0.21,
        mar_threshold=0.60,
        perclos_window_seconds=30,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold

        self.perclos_window_seconds = perclos_window_seconds

        # Stores:
        # (timestamp, eyes_closed)
        self.eye_history = deque()

        self.blink_count = 0
        self.previous_eyes_closed = False

        self.current_blink_start = None
        self.last_blink_duration = 0.0

    @staticmethod
    def distance(point_a, point_b):
        """Calculate Euclidean distance between two landmarks."""

        return math.sqrt(
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
        )

    def eye_aspect_ratio(self, landmarks, eye_indices):
        """
        Calculate EAR.

        EAR = vertical eye opening / horizontal eye width
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

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

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

        return (left_ear + right_ear) / 2.0

    def calculate_mar(self, landmarks):
        """
        Calculate Mouth Aspect Ratio.

        The current implementation uses vertical mouth opening
        relative to mouth width.
        """

        left = landmarks[self.MOUTH[0]]
        right = landmarks[self.MOUTH[1]]
        top = landmarks[self.MOUTH[2]]
        bottom = landmarks[self.MOUTH[3]]

        mouth_width = self.distance(left, right)
        mouth_height = self.distance(top, bottom)

        if mouth_width == 0:
            return 0.0

        return mouth_height / mouth_width

    def update_perclos(self, timestamp, eyes_closed):
        """Update the PERCLOS sliding window."""

        self.eye_history.append(
            (timestamp, bool(eyes_closed))
        )

        cutoff = timestamp - self.perclos_window_seconds

        while self.eye_history:
            oldest_timestamp = self.eye_history[0][0]

            if oldest_timestamp >= cutoff:
                break

            self.eye_history.popleft()

        if not self.eye_history:
            return 0.0

        closed_count = sum(
            1
            for _, closed in self.eye_history
            if closed
        )

        return closed_count / len(self.eye_history)

    def update_blink(self, timestamp, eyes_closed):
        """
        Detect a blink by observing the transition:

        open → closed → open
        """

        eyes_closed = bool(eyes_closed)

        # Open -> Closed
        if eyes_closed and not self.previous_eyes_closed:
            self.current_blink_start = timestamp

        # Closed -> Open
        elif not eyes_closed and self.previous_eyes_closed:

            if self.current_blink_start is not None:

                duration = (
                    timestamp - self.current_blink_start
                )

                self.last_blink_duration = duration

                # Treat short eye closure as a blink.
                if duration <= 0.8:
                    self.blink_count += 1

            self.current_blink_start = None

        self.previous_eyes_closed = eyes_closed

    def calculate(
        self,
        landmarks,
        timestamp=None,
    ):
        """Calculate all currently supported metrics."""

        if timestamp is None:
            timestamp = time.time()

        ear = self.calculate_ear(landmarks)
        mar = self.calculate_mar(landmarks)

        eyes_closed = ear < self.ear_threshold
        yawning = mar > self.mar_threshold

        perclos = self.update_perclos(
            timestamp,
            eyes_closed,
        )

        self.update_blink(
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
        }