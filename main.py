"""
ADAPTIVE-DMS
Adaptive Multimodal Driver State Monitoring
and Predictive Safety Intervention System

Version:
    v0.5

Implemented:
    - Webcam
    - MediaPipe Face Mesh
    - EAR
    - MAR
    - PERCLOS
    - Blink detection
    - Microsleep detection
    - Head pose
    - Gaze estimation
    - Signal reliability estimation
"""

import time

import cv2

from src.camera import Camera
from src.detector import FaceDetector
from src.metrics import DriverMetrics
from src.head_pose import HeadPoseEstimator
from src.gaze import GazeEstimator
from src.reliability import SignalReliabilityEstimator


def main():

    print("=" * 70)

    print("ADAPTIVE-DMS")

    print(
        "Adaptive Multimodal Driver State Monitoring "
        "and Predictive Safety Intervention System"
    )

    print("=" * 70)

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
    # RELIABILITY
    # ---------------------------------------------------------

    reliability = (
        SignalReliabilityEstimator()
    )

    # ---------------------------------------------------------
    # FPS
    # ---------------------------------------------------------

    previous_time = time.time()

    try:

        while True:

            # -------------------------------------------------
            # CAMERA
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

            delta_time = (
                current_time
                - previous_time
            )

            if delta_time > 0:

                fps = (
                    1.0
                    / delta_time
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
            # DEFAULT VALUES
            # -------------------------------------------------

            pose_values = {
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "valid": False,
            }

            gaze_values = {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "gaze_direction": "UNKNOWN",
                "gaze_away_duration": 0.0,
                "prolonged_gaze_away": False,
            }

            driver_values = {
                "ear": 0.0,
                "mar": 0.0,
                "perclos": 0.0,
                "eyes_closed": False,
                "yawning": False,
                "blink_count": 0,
                "blink_duration": 0.0,
                "microsleep_duration": 0.0,
                "microsleep": False,
            }

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

                driver_values = (
                    metrics.calculate(
                        landmarks,
                        timestamp=current_time,
                    )
                )

                # -------------------------------------------------
                # HEAD POSE
                # -------------------------------------------------

                pose_values = (
                    head_pose.estimate(
                        landmarks,
                        frame.shape[1],
                        frame.shape[0],
                    )
                )

                # -------------------------------------------------
                # GAZE
                # -------------------------------------------------

                gaze_values = (
                    gaze.estimate(
                        landmarks,
                        timestamp=current_time,
                    )
                )

                # -------------------------------------------------
                # DRAW LANDMARKS
                # -------------------------------------------------

                frame = (
                    detector.draw_landmarks(
                        frame,
                        results,
                    )
                )

            # -------------------------------------------------
            # PERCLOS SAMPLE COUNT
            # -------------------------------------------------

            perclos_sample_count = len(
                metrics.eye_history
            )

            # -------------------------------------------------
            # RELIABILITY
            # -------------------------------------------------

            reliability_values = (
                reliability.calculate(
                    frame=frame,
                    driver_values=driver_values,
                    pose_values=pose_values,
                    gaze_values=gaze_values,
                    perclos_sample_count=(
                        perclos_sample_count
                    ),
                    face_detected=face_detected,
                )
            )

            reliability_vector = (
                reliability_values[
                    "reliability_vector"
                ]
            )

            # -------------------------------------------------
            # DISPLAY DRIVER METRICS
            # -------------------------------------------------

            if face_detected:

                # EAR
                cv2.putText(
                    frame,
                    f"EAR: {driver_values['ear']:.3f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 0),
                    2,
                )

                # MAR
                cv2.putText(
                    frame,
                    f"MAR: {driver_values['mar']:.3f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 0),
                    2,
                )

                # PERCLOS
                cv2.putText(
                    frame,
                    (
                        f"PERCLOS: "
                        f"{driver_values['perclos']:.3f}"
                    ),
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 0),
                    2,
                )

                # BLINK
                cv2.putText(
                    frame,
                    (
                        f"Blinks: "
                        f"{driver_values['blink_count']}"
                    ),
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 0),
                    2,
                )

                # EYE STATE
                eye_status = (
                    "EYES CLOSED"
                    if driver_values["eyes_closed"]
                    else "EYES OPEN"
                )

                cv2.putText(
                    frame,
                    eye_status,
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 165, 255)
                    if driver_values["eyes_closed"]
                    else (0, 255, 0),
                    2,
                )

                # YAWNING
                if driver_values["yawning"]:

                    cv2.putText(
                        frame,
                        "YAWNING",
                        (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2,
                    )

                # MICROSLEEP
                if driver_values["microsleep"]:

                    cv2.putText(
                        frame,
                        (
                            "MICROSLEEP: "
                            f"{driver_values['microsleep_duration']:.1f}s"
                        ),
                        (20, 230),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )

                # HEAD POSE
                if pose_values["valid"]:

                    cv2.putText(
                        frame,
                        (
                            f"Pitch: "
                            f"{pose_values['pitch']:.1f}"
                        ),
                        (20, 265),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Yaw: "
                            f"{pose_values['yaw']:.1f}"
                        ),
                        (20, 295),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Roll: "
                            f"{pose_values['roll']:.1f}"
                        ),
                        (20, 325),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )

                # GAZE
                cv2.putText(
                    frame,
                    (
                        f"Gaze: "
                        f"{gaze_values['gaze_direction']}"
                    ),
                    (20, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 255),
                    2,
                )

            else:

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            # -------------------------------------------------
            # RELIABILITY DISPLAY
            # ---------------------------------------------------------

            cv2.putText(
                frame,
                "SIGNAL RELIABILITY",
                (520, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            y_position = 65

            for name, score in reliability_vector.items():

                cv2.putText(
                    frame,
                    f"{name}: {score:.2f}",
                    (520, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                )

                y_position += 25

            # -------------------------------------------------
            # OVERALL RELIABILITY
            # -------------------------------------------------

            overall = (
                reliability_values[
                    "overall_reliability"
                ]
            )

            cv2.putText(
                frame,
                (
                    f"Overall Reliability: "
                    f"{overall:.2f}"
                ),
                (520, y_position + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            # -------------------------------------------------
            # IMAGE QUALITY
            # -------------------------------------------------

            cv2.putText(
                frame,
                (
                    f"Image Quality: "
                    f"{reliability_values['image_quality']:.2f}"
                ),
                (520, y_position + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
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
                0.60,
                (255, 255, 255),
                2,
            )

            # -------------------------------------------------
            # STATUS
            # -------------------------------------------------

            cv2.putText(
                frame,
                (
                    "FACE DETECTED"
                    if face_detected
                    else "FACE NOT DETECTED"
                ),
                (
                    20,
                    frame.shape[0] - 50,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (
                    (0, 255, 0)
                    if face_detected
                    else (0, 0, 255)
                ),
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
                0.50,
                (255, 255, 255),
                2,
            )

            # -------------------------------------------------
            # SHOW
            # -------------------------------------------------

            cv2.imshow(
                "ADAPTIVE-DMS",
                frame,
            )

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
        print("=" * 70)
        print("Camera released.")
        print("ADAPTIVE-DMS stopped.")
        print("=" * 70)


if __name__ == "__main__":
    main()