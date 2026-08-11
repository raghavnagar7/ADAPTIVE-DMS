"""
Gaze estimation for ADAPTIVE-DMS.

Current functionality:
    - Left/right gaze estimation
    - Center gaze estimation
    - Vertical gaze estimation
    - Gaze-away duration
"""

import time
from collections import deque


class GazeEstimator:
    """
    Estimate gaze direction using MediaPipe iris landmarks.

    This is the initial gaze module.
    Later versions will include:
        - Personal calibration
        - Head-pose compensation
        - Temporal smoothing
        - Reliability estimation
    """

    # ---------------------------------------------------------
    # MediaPipe landmarks
    # ---------------------------------------------------------

    # Left eye corners
    LEFT_EYE_LEFT_CORNER = 33
    LEFT_EYE_RIGHT_CORNER = 133

    # Right eye corners
    RIGHT_EYE_LEFT_CORNER = 362
    RIGHT_EYE_RIGHT_CORNER = 263

    # Upper/lower eyelids
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145

    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Iris landmarks
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    def __init__(
        self,
        horizontal_left_threshold=0.35,
        horizontal_right_threshold=0.65,
        vertical_up_threshold=0.35,
        vertical_down_threshold=0.65,
        gaze_away_threshold=1.0,
        smoothing_window=5,
    ):

        self.horizontal_left_threshold = (
            horizontal_left_threshold
        )

        self.horizontal_right_threshold = (
            horizontal_right_threshold
        )

        self.vertical_up_threshold = (
            vertical_up_threshold
        )

        self.vertical_down_threshold = (
            vertical_down_threshold
        )

        self.gaze_away_threshold = (
            gaze_away_threshold
        )

        self.history = deque(
            maxlen=smoothing_window
        )

        self.away_start_time = None

        self.gaze_away_duration = 0.0

    # ---------------------------------------------------------
    # Basic point utilities
    # ---------------------------------------------------------

    @staticmethod
    def _average_point(
        landmarks,
        indices,
    ):
        """Calculate average normalized landmark position."""

        x = sum(
            landmarks[index].x
            for index in indices
        ) / len(indices)

        y = sum(
            landmarks[index].y
            for index in indices
        ) / len(indices)

        return x, y

    @staticmethod
    def _single_point(
        landmarks,
        index,
    ):
        """Get a single normalized landmark."""

        return (
            landmarks[index].x,
            landmarks[index].y,
        )

    # ---------------------------------------------------------
    # Eye gaze ratio
    # ---------------------------------------------------------

    @staticmethod
    def _horizontal_ratio(
        iris_x,
        left_corner_x,
        right_corner_x,
    ):
        """
        Calculate horizontal iris position.

        Approximately:
            0.0 = one side of eye
            0.5 = center
            1.0 = opposite side
        """

        eye_width = (
            right_corner_x
            - left_corner_x
        )

        if abs(eye_width) < 1e-6:
            return 0.5

        ratio = (
            iris_x
            - left_corner_x
        ) / eye_width

        return max(
            0.0,
            min(1.0, ratio),
        )

    @staticmethod
    def _vertical_ratio(
        iris_y,
        top_y,
        bottom_y,
    ):
        """Calculate vertical iris position."""

        eye_height = (
            bottom_y
            - top_y
        )

        if abs(eye_height) < 1e-6:
            return 0.5

        ratio = (
            iris_y
            - top_y
        ) / eye_height

        return max(
            0.0,
            min(1.0, ratio),
        )

    # ---------------------------------------------------------
    # Direction classification
    # ---------------------------------------------------------

    def _horizontal_direction(
        self,
        ratio,
    ):
        if ratio < self.horizontal_left_threshold:
            return "LEFT"

        if ratio > self.horizontal_right_threshold:
            return "RIGHT"

        return "CENTER"

    def _vertical_direction(
        self,
        ratio,
    ):
        if ratio < self.vertical_up_threshold:
            return "UP"

        if ratio > self.vertical_down_threshold:
            return "DOWN"

        return "CENTER"

    # ---------------------------------------------------------
    # Main calculation
    # ---------------------------------------------------------

    def estimate(
        self,
        landmarks,
        timestamp=None,
    ):
        """
        Estimate gaze direction.

        Returns:
            left eye ratio
            right eye ratio
            average horizontal ratio
            average vertical ratio
            horizontal direction
            vertical direction
            gaze direction
            gaze-away duration
        """

        if timestamp is None:
            timestamp = time.time()

        # -----------------------------------------------------
        # Iris centers
        # -----------------------------------------------------

        left_iris_x, left_iris_y = (
            self._average_point(
                landmarks,
                self.LEFT_IRIS,
            )
        )

        right_iris_x, right_iris_y = (
            self._average_point(
                landmarks,
                self.RIGHT_IRIS,
            )
        )

        # -----------------------------------------------------
        # Left eye corners
        # -----------------------------------------------------

        left_corner_a = self._single_point(
            landmarks,
            self.LEFT_EYE_LEFT_CORNER,
        )

        left_corner_b = self._single_point(
            landmarks,
            self.LEFT_EYE_RIGHT_CORNER,
        )

        left_eye_left_x = min(
            left_corner_a[0],
            left_corner_b[0],
        )

        left_eye_right_x = max(
            left_corner_a[0],
            left_corner_b[0],
        )

        # -----------------------------------------------------
        # Right eye corners
        # -----------------------------------------------------

        right_corner_a = self._single_point(
            landmarks,
            self.RIGHT_EYE_LEFT_CORNER,
        )

        right_corner_b = self._single_point(
            landmarks,
            self.RIGHT_EYE_RIGHT_CORNER,
        )

        right_eye_left_x = min(
            right_corner_a[0],
            right_corner_b[0],
        )

        right_eye_right_x = max(
            right_corner_a[0],
            right_corner_b[0],
        )

        # -----------------------------------------------------
        # Eye vertical references
        # -----------------------------------------------------

        left_top_x, left_top_y = (
            self._single_point(
                landmarks,
                self.LEFT_EYE_TOP,
            )
        )

        left_bottom_x, left_bottom_y = (
            self._single_point(
                landmarks,
                self.LEFT_EYE_BOTTOM,
            )
        )

        right_top_x, right_top_y = (
            self._single_point(
                landmarks,
                self.RIGHT_EYE_TOP,
            )
        )

        right_bottom_x, right_bottom_y = (
            self._single_point(
                landmarks,
                self.RIGHT_EYE_BOTTOM,
            )
        )

        # -----------------------------------------------------
        # Ratios
        # -----------------------------------------------------

        left_horizontal_ratio = (
            self._horizontal_ratio(
                left_iris_x,
                left_eye_left_x,
                left_eye_right_x,
            )
        )

        right_horizontal_ratio = (
            self._horizontal_ratio(
                right_iris_x,
                right_eye_left_x,
                right_eye_right_x,
            )
        )

        left_vertical_ratio = (
            self._vertical_ratio(
                left_iris_y,
                min(left_top_y, left_bottom_y),
                max(left_top_y, left_bottom_y),
            )
        )

        right_vertical_ratio = (
            self._vertical_ratio(
                right_iris_y,
                min(right_top_y, right_bottom_y),
                max(right_top_y, right_bottom_y),
            )
        )

        horizontal_ratio = (
            left_horizontal_ratio
            + right_horizontal_ratio
        ) / 2.0

        vertical_ratio = (
            left_vertical_ratio
            + right_vertical_ratio
        ) / 2.0

        # -----------------------------------------------------
        # Temporal smoothing
        # -----------------------------------------------------

        self.history.append(
            (
                horizontal_ratio,
                vertical_ratio,
            )
        )

        smooth_horizontal = (
            sum(
                item[0]
                for item in self.history
            )
            / len(self.history)
        )

        smooth_vertical = (
            sum(
                item[1]
                for item in self.history
            )
            / len(self.history)
        )

        # -----------------------------------------------------
        # Direction
        # -----------------------------------------------------

        horizontal_direction = (
            self._horizontal_direction(
                smooth_horizontal
            )
        )

        vertical_direction = (
            self._vertical_direction(
                smooth_vertical
            )
        )

        # -----------------------------------------------------
        # Overall gaze
        # -----------------------------------------------------

        if horizontal_direction != "CENTER":

            gaze_direction = (
                horizontal_direction
            )

        elif vertical_direction != "CENTER":

            gaze_direction = (
                vertical_direction
            )

        else:

            gaze_direction = "CENTER"

        # -----------------------------------------------------
        # Gaze-away duration
        # -----------------------------------------------------

        is_gaze_away = (
            gaze_direction != "CENTER"
        )

        if is_gaze_away:

            if self.away_start_time is None:

                self.away_start_time = (
                    timestamp
                )

            self.gaze_away_duration = (
                timestamp
                - self.away_start_time
            )

        else:

            self.away_start_time = None

            self.gaze_away_duration = 0.0

        prolonged_gaze_away = (
            self.gaze_away_duration
            >= self.gaze_away_threshold
        )

        return {
            "left_horizontal_ratio":
                left_horizontal_ratio,

            "right_horizontal_ratio":
                right_horizontal_ratio,

            "horizontal_ratio":
                smooth_horizontal,

            "vertical_ratio":
                smooth_vertical,

            "horizontal_direction":
                horizontal_direction,

            "vertical_direction":
                vertical_direction,

            "gaze_direction":
                gaze_direction,

            "gaze_away_duration":
                self.gaze_away_duration,

            "prolonged_gaze_away":
                prolonged_gaze_away,
        }