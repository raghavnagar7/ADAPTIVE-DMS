"""
ADAPTIVE-DMS
Adaptive Multimodal Driver State Monitoring
and Predictive Safety Intervention System

Version:
    v0.7

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
    - Signal reliability
    - Adaptive multimodal fusion
    - Temporal fatigue analysis
"""

import time

import cv2

from src.camera import Camera
from src.detector import FaceDetector
from src.metrics import DriverMetrics
from src.head_pose import HeadPoseEstimator
from src.gaze import GazeEstimator
from src.reliability import SignalReliabilityEstimator
from src.fusion import AdaptiveMultimodalFusion
from src.temporal import TemporalFatiguePredictor


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

    reliability = SignalReliabilityEstimator()

    # ---------------------------------------------------------
    # ADAPTIVE FUSION
    # ---------------------------------------------------------

    fusion = AdaptiveMultimodalFusion()

    # ---------------------------------------------------------
    # TEMPORAL FATIGUE ANALYSIS
    # ---------------------------------------------------------

    temporal = TemporalFatiguePredictor(
        history_seconds=30.0,
        sample_interval=0.5,
        short_window_seconds=5.0,
        medium_window_seconds=15.0,
        increasing_threshold=0.015,
        high_risk_threshold=0.60,
    )

    # ---------------------------------------------------------
    # FPS
    # ---------------------------------------------------------

    previous_time = time.time()

    try:

        while True:

            # -------------------------------------------------
            # READ CAMERA
            # -------------------------------------------------

            frame = camera.read()

            if frame is None:

                print(
                    "Unable to read frame from camera."
                )

                break

            # -------------------------------------------------
            # FPS
            # -------------------------------------------------

            current_time = time.time()

            delta_time = (
                current_time
                - previous_time
            )

            if delta_time > 0:

                fps = 1.0 / delta_time

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

            # -------------------------------------------------
            # FACE DETECTED
            # -------------------------------------------------

            if face_detected:

                face_landmarks = (
                    results.multi_face_landmarks[0]
                )

                landmarks = (
                    face_landmarks.landmark
                )

                # ---------------------------------------------
                # DRIVER METRICS
                # ---------------------------------------------

                driver_values = metrics.calculate(
                    landmarks,
                    timestamp=current_time,
                )

                # ---------------------------------------------
                # HEAD POSE
                # ---------------------------------------------

                pose_values = head_pose.estimate(
                    landmarks,
                    frame.shape[1],
                    frame.shape[0],
                )

                # ---------------------------------------------
                # GAZE
                # ---------------------------------------------

                gaze_values = gaze.estimate(
                    landmarks,
                    timestamp=current_time,
                )

                # ---------------------------------------------
                # DRAW FACE LANDMARKS
                # ---------------------------------------------

                frame = detector.draw_landmarks(
                    frame,
                    results,
                )

            # -------------------------------------------------
            # PERCLOS HISTORY
            # -------------------------------------------------

            try:

                perclos_sample_count = len(
                    metrics.eye_history
                )

            except AttributeError:

                perclos_sample_count = 0

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

            # -------------------------------------------------
            # ADAPTIVE FUSION
            # -------------------------------------------------

            fusion_values = fusion.calculate(
                driver_values=driver_values,
                pose_values=pose_values,
                gaze_values=gaze_values,
                reliability_values=(
                    reliability_values
                ),
            )

            # -------------------------------------------------
            # TEMPORAL ANALYSIS
            # -------------------------------------------------

            temporal_values = temporal.update(
                fatigue_risk=(
                    fusion_values["fatigue_risk"]
                ),
                timestamp=current_time,
            )

            # =================================================
            # LEFT PANEL
            # =================================================

            if face_detected:

                # EAR
                cv2.putText(
                    frame,
                    f"EAR: {driver_values['ear']:.3f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                # MAR
                cv2.putText(
                    frame,
                    f"MAR: {driver_values['mar']:.3f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
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
                    0.55,
                    (0, 255, 0),
                    2,
                )

                # BLINKS
                cv2.putText(
                    frame,
                    (
                        f"Blinks: "
                        f"{driver_values['blink_count']}"
                    ),
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                # EYE STATUS
                eye_status = (
                    "EYES CLOSED"
                    if driver_values["eyes_closed"]
                    else "EYES OPEN"
                )

                eye_color = (
                    (0, 165, 255)
                    if driver_values["eyes_closed"]
                    else (0, 255, 0)
                )

                cv2.putText(
                    frame,
                    eye_status,
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    eye_color,
                    2,
                )

                # YAWNING
                if driver_values["yawning"]:

                    cv2.putText(
                        frame,
                        "YAWNING",
                        (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 165, 255),
                        2,
                    )

                # MICROSLEEP
                if driver_values["microsleep"]:

                    cv2.putText(
                        frame,
                        (
                            "MICROSLEEP "
                            f"{driver_values['microsleep_duration']:.1f}s"
                        ),
                        (20, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
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
                        (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Yaw: "
                            f"{pose_values['yaw']:.1f}"
                        ),
                        (20, 285),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Roll: "
                            f"{pose_values['roll']:.1f}"
                        ),
                        (20, 310),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
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
                    (20, 345),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 0, 255),
                    2,
                )

                # GAZE AWAY
                if gaze_values[
                    "prolonged_gaze_away"
                ]:

                    cv2.putText(
                        frame,
                        (
                            "GAZE AWAY "
                            f"{gaze_values['gaze_away_duration']:.1f}s"
                        ),
                        (20, 380),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
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

            # =================================================
            # RIGHT PANEL
            # =================================================

            cv2.putText(
                frame,
                "ADAPTIVE FUSION",
                (500, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            # -------------------------------------------------
            # FATIGUE RISK
            # -------------------------------------------------

            fatigue_risk = (
                fusion_values["fatigue_risk"]
            )

            risk_level = (
                fusion_values["risk_level"]
            )

            cv2.putText(
                frame,
                (
                    f"Fatigue Risk: "
                    f"{fatigue_risk:.2f}"
                ),
                (500, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
            )

            if risk_level == "NORMAL":

                risk_color = (0, 255, 0)

            elif risk_level in (
                "LOW",
                "MODERATE",
            ):

                risk_color = (0, 255, 255)

            else:

                risk_color = (0, 0, 255)

            cv2.putText(
                frame,
                (
                    f"Risk Level: "
                    f"{risk_level}"
                ),
                (500, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                risk_color,
                2,
            )

            # -------------------------------------------------
            # ADAPTIVE WEIGHTS
            # -------------------------------------------------

            cv2.putText(
                frame,
                "ADAPTIVE WEIGHTS",
                (500, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
            )

            weights = (
                fusion_values[
                    "adaptive_weights"
                ]
            )

            y = 165

            for signal, weight in weights.items():

                cv2.putText(
                    frame,
                    (
                        f"{signal}: "
                        f"{weight:.2f}"
                    ),
                    (500, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                )

                y += 20

            # -------------------------------------------------
            # RELIABILITY
            # -------------------------------------------------

            overall_reliability = (
                reliability_values[
                    "overall_reliability"
                ]
            )

            cv2.putText(
                frame,
                (
                    f"Reliability: "
                    f"{overall_reliability:.2f}"
                ),
                (500, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
            )

            # =================================================
            # TEMPORAL ANALYSIS
            # =================================================

            temporal_y = y + 40

            cv2.putText(
                frame,
                "TEMPORAL ANALYSIS",
                (500, temporal_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
            )

            # Temporal state
            temporal_state = (
                temporal_values["state"]
            )

            if temporal_state == "STABLE":

                temporal_color = (
                    0,
                    255,
                    0,
                )

            elif temporal_state in (
                "RISING",
                "INCREASING",
            ):

                temporal_color = (
                    0,
                    255,
                    255,
                )

            elif temporal_state in (
                "HIGH",
                "PERSISTENT",
            ):

                temporal_color = (
                    0,
                    165,
                    255,
                )

            elif temporal_state == "CRITICAL":

                temporal_color = (
                    0,
                    0,
                    255,
                )

            else:

                temporal_color = (
                    255,
                    255,
                    255,
                )

            cv2.putText(
                frame,
                (
                    f"Temporal State: "
                    f"{temporal_state}"
                ),
                (500, temporal_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                temporal_color,
                2,
            )

            # 5 second average
            cv2.putText(
                frame,
                (
                    f"5s Avg: "
                    f"{temporal_values['short_average']:.2f}"
                ),
                (500, temporal_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )

            # 15 second average
            cv2.putText(
                frame,
                (
                    f"15s Avg: "
                    f"{temporal_values['medium_average']:.2f}"
                ),
                (500, temporal_y + 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )

            # 30 second average
            cv2.putText(
                frame,
                (
                    f"30s Avg: "
                    f"{temporal_values['long_average']:.2f}"
                ),
                (500, temporal_y + 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )

            # Trend
            cv2.putText(
                frame,
                (
                    f"Trend: "
                    f"{temporal_values['trend']:.3f}"
                ),
                (500, temporal_y + 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )

            # Samples
            cv2.putText(
                frame,
                (
                    f"Samples: "
                    f"{temporal_values['history_samples']}"
                ),
                (500, temporal_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )

            # =================================================
            # FPS
            # =================================================

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

            # =================================================
            # FACE STATUS
            # =================================================

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

            # =================================================
            # PROJECT NAME
            # =================================================

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

            # =================================================
            # SHOW FRAME
            # =================================================

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