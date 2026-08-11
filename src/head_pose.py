"""
Head pose estimation for ADAPTIVE-DMS.

Estimates:
    - Pitch
    - Yaw
    - Roll
"""

import cv2
import numpy as np


class HeadPoseEstimator:
    """Estimate head orientation from MediaPipe landmarks."""

    LANDMARKS = {
        "nose": 1,
        "chin": 152,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291,
    }

    def estimate(
        self,
        landmarks,
        frame_width,
        frame_height,
    ):
        """Estimate pitch, yaw and roll."""

        image_points = np.array(
            [
                self._point(
                    landmarks[self.LANDMARKS["nose"]],
                    frame_width,
                    frame_height,
                ),
                self._point(
                    landmarks[self.LANDMARKS["chin"]],
                    frame_width,
                    frame_height,
                ),
                self._point(
                    landmarks[self.LANDMARKS["left_eye"]],
                    frame_width,
                    frame_height,
                ),
                self._point(
                    landmarks[self.LANDMARKS["right_eye"]],
                    frame_width,
                    frame_height,
                ),
                self._point(
                    landmarks[self.LANDMARKS["left_mouth"]],
                    frame_width,
                    frame_height,
                ),
                self._point(
                    landmarks[self.LANDMARKS["right_mouth"]],
                    frame_width,
                    frame_height,
                ),
            ],
            dtype=np.float64,
        )

        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )

        focal_length = float(frame_width)

        center = (
            frame_width / 2.0,
            frame_height / 2.0,
        )

        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        distortion_coefficients = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return {
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "valid": False,
            }

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        pitch, yaw, roll = self._rotation_matrix_to_euler(
            rotation_matrix
        )

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "valid": True,
        }

    @staticmethod
    def _rotation_matrix_to_euler(rotation_matrix):
        """Convert rotation matrix to intuitive Euler angles."""

        r11 = rotation_matrix[0, 0]
        r21 = rotation_matrix[1, 0]
        r31 = rotation_matrix[2, 0]

        r32 = rotation_matrix[2, 1]
        r33 = rotation_matrix[2, 2]

        sy = np.sqrt(
            r11 * r11 + r21 * r21
        )

        singular = sy < 1e-6

        if not singular:

            pitch = np.arctan2(
                r32,
                r33,
            )

            yaw = np.arctan2(
                -r31,
                sy,
            )

            roll = np.arctan2(
                r21,
                r11,
            )

        else:

            pitch = np.arctan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1],
            )

            yaw = np.arctan2(
                -r31,
                sy,
            )

            roll = 0.0

        pitch = float(np.degrees(pitch))
        yaw = float(np.degrees(yaw))
        roll = float(np.degrees(roll))

        # Convert the pitch from the OpenCV orientation
        # into a more intuitive driver-facing orientation.
        if pitch > 90.0:
            pitch -= 180.0

        elif pitch < -90.0:
            pitch += 180.0

        yaw = HeadPoseEstimator._normalize_angle(yaw)
        roll = HeadPoseEstimator._normalize_angle(roll)

        return pitch, yaw, roll

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-180, 180]."""

        return (angle + 180.0) % 360.0 - 180.0

    @staticmethod
    def _point(
        landmark,
        width,
        height,
    ):
        """Convert normalized landmark coordinates to pixels."""

        return (
            landmark.x * width,
            landmark.y * height,
        )