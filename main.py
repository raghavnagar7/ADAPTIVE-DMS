"""
ADAPTIVE-DMS
Adaptive Multimodal Driver State Monitoring
and Predictive Safety Intervention System

Version:
    v0.4

Implemented:
    - Webcam input
    - MediaPipe Face Mesh
    - EAR
    - MAR
    - PERCLOS
    - Blink detection
    - Microsleep detection
    - Head pose estimation
    - Gaze estimation
"""

import time

import cv2

from src.camera import Camera
from src.detector import FaceDetector
from src.metrics import DriverMetrics
from src.head_pose import HeadPoseEstimator
from src.gaze import GazeEstimator


def main():

    print("=" * 65)

    print("ADAPTIVE-DMS")

    print(
        "Adaptive Multimodal Driver State Monitoring "
        "and Predictive Safety Intervention System"
    )

    print("=" * 65)

    print("Starting camera...")
    print("Press Q to quit.")
    print()

    # ---------------------------------------------------------
    # CAMERA
    # ---------------------------------------------------------

    camera = Camera(
        source=0,
        width=960,
        height=540,
    )

    # ---------------------------------------------------------
    # FACE DETECTOR
    # ---------------------------------------------------------

    detector = FaceDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ---------------------------------------------------------
    # DRIVER METRICS
    # ---------------------------------------------------------

    metrics = DriverMetrics(
        ear_threshold=0.21,
        mar_threshold=0.60,
        perclos_window_seconds=30,
        microsleep_threshold=1.5,
    )

    # ---------------------------------------------------------
    # HEAD POSE
    # ---------------------------------------------------------

    head_pose = HeadPoseEstimator()

    # ---------------------------------------------------------
    # GAZE
    # ---------------------------------------------------------

    gaze = GazeEstimator(
        horizontal_left_threshold=0.35,
        horizontal_right_threshold=0.65,
        vertical_up_threshold=0.35,
        vertical_down_threshold=0.65,
        gaze_away_threshold=1.0,
        smoothing_window=5,
    )

    # ---------------------------------------------------------
    # FPS
    # ---------------------------------------------------------

    previous_time = time.time()

    try:

        while True:

            # -------------------------------------------------
            # CAMERA FRAME
            # -------------------------------------------------

            frame = camera.read()

            if frame is None:

                print(
                    "Unable to read frame from camera."
                )

                break

            # -------------------------------------------------
            # TIME
            # -------------------------------------------------

            current_time = time.time()

            time_difference = (
                current_time
                - previous_time
            )

            if time_difference > 0:

                fps = (
                    1.0
                    / time_difference
                )

            else:

                fps = 0.0

            previous_time = current_time

            # -------------------------------------------------
            # FACE DETECTION
            # -------------------------------------------------

            results = detector.process(
                frame
            )

            face_detected = bool(
                results.multi_face_landmarks
            )

            # -------------------------------------------------
            # FACE FOUND
            # -------------------------------------------------

            if face_detected:

                face_landmarks = (
                    results.multi_face_landmarks[0]
                )

                landmarks = (
                    face_landmarks.landmark
                )

                # -------------------------------------------------
                # DRIVER METRICS
                # -------------------------------------------------

                values = metrics.calculate(
                    landmarks,
                    timestamp=current_time,
                )

                # -------------------------------------------------
                # HEAD POSE
                # -------------------------------------------------

                pose = head_pose.estimate(
                    landmarks,
                    frame.shape[1],
                    frame.shape[0],
                )

                # -------------------------------------------------
                # GAZE
                # -------------------------------------------------

                gaze_values = gaze.estimate(
                    landmarks,
                    timestamp=current_time,
                )

                # -------------------------------------------------
                # DRAW FACE LANDMARKS
                # -------------------------------------------------

                frame = (
                    detector.draw_landmarks(
                        frame,
                        results,
                    )
                )

                # -------------------------------------------------
                # EAR
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"EAR: {values['ear']:.3f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                # -------------------------------------------------
                # MAR
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"MAR: {values['mar']:.3f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                # -------------------------------------------------
                # PERCLOS
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"PERCLOS: {values['perclos']:.3f}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                # -------------------------------------------------
                # BLINKS
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"Blinks: {values['blink_count']}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                # -------------------------------------------------
                # EYE STATUS
                # -------------------------------------------------

                if values["eyes_closed"]:

                    eye_status = "EYES CLOSED"

                    eye_color = (
                        0,
                        165,
                        255,
                    )

                else:

                    eye_status = "EYES OPEN"

                    eye_color = (
                        0,
                        255,
                        0,
                    )

                cv2.putText(
                    frame,
                    eye_status,
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    eye_color,
                    2,
                )

                # -------------------------------------------------
                # YAWNING
                # -------------------------------------------------

                if values["yawning"]:

                    cv2.putText(
                        frame,
                        "YAWNING",
                        (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )

                # -------------------------------------------------
                # MICROSLEEP
                # -------------------------------------------------

                if values["microsleep"]:

                    cv2.putText(
                        frame,
                        (
                            "MICROSLEEP: "
                            f"{values['microsleep_duration']:.1f}s"
                        ),
                        (20, 230),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                else:

                    cv2.putText(
                        frame,
                        (
                            "Eye closure: "
                            f"{values['microsleep_duration']:.1f}s"
                        ),
                        (20, 230),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                    )

                # -------------------------------------------------
                # HEAD POSE
                # -------------------------------------------------

                if pose["valid"]:

                    cv2.putText(
                        frame,
                        (
                            f"Pitch: "
                            f"{pose['pitch']:.1f}"
                        ),
                        (20, 265),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Yaw: "
                            f"{pose['yaw']:.1f}"
                        ),
                        (20, 295),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Roll: "
                            f"{pose['roll']:.1f}"
                        ),
                        (20, 325),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                # -------------------------------------------------
                # GAZE DIRECTION
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    (
                        "Gaze: "
                        f"{gaze_values['gaze_direction']}"
                    ),
                    (20, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                )

                # -------------------------------------------------
                # GAZE RATIOS
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    (
                        "Gaze X: "
                        f"{gaze_values['horizontal_ratio']:.2f}"
                    ),
                    (20, 390),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                cv2.putText(
                    frame,
                    (
                        "Gaze Y: "
                        f"{gaze_values['vertical_ratio']:.2f}"
                    ),
                    (20, 415),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                # -------------------------------------------------
                # GAZE AWAY
                # -------------------------------------------------

                if gaze_values[
                    "prolonged_gaze_away"
                ]:

                    cv2.putText(
                        frame,
                        (
                            "GAZE AWAY: "
                            f"{gaze_values['gaze_away_duration']:.1f}s"
                        ),
                        (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # -------------------------------------------------
                # FACE STATUS
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    "FACE DETECTED",
                    (
                        20,
                        frame.shape[0] - 50,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

            # -------------------------------------------------
            # NO FACE
            # -------------------------------------------------

            else:

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # -------------------------------------------------
            # FPS
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (
                    frame.shape[1] - 130,
                    35,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # -------------------------------------------------
            # PROJECT NAME
            # -------------------------------------------------

            cv2.putText(
                frame,
                "ADAPTIVE-DMS | Press Q to quit",
                (
                    20,
                    frame.shape[0] - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # -------------------------------------------------
            # SHOW WINDOW
            # -------------------------------------------------

            cv2.imshow(
                "ADAPTIVE-DMS",
                frame,
            )

            # -------------------------------------------------
            # KEYBOARD
            # -------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print()
        print(
            "Program interrupted by user."
        )

    finally:

        detector.close()

        camera.release()

        cv2.destroyAllWindows()

        print()
        print("=" * 65)
        print("Camera released.")
        print("ADAPTIVE-DMS stopped.")
        print("=" * 65)


if __name__ == "__main__":
    main()