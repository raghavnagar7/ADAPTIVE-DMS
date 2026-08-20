"""
=============================================================
ADAPTIVE-DMS
=============================================================

Adaptive Multimodal Driver State Monitoring
and Predictive Safety Intervention System.

Version:
    v1.8 - Step 4

Features:
    - Webcam
    - Face Detection
    - EAR
    - MAR
    - PERCLOS
    - Blink Detection
    - Microsleep Detection
    - Head Pose
    - Gaze Estimation
    - Signal Reliability
    - Adaptive Multimodal Fusion
    - Temporal Fatigue Analysis
    - GRU Temporal Fatigue Prediction
    - Adaptive Safety Intervention
    - Direct 1.5-second Eye Closure Alert
    - Session Data Logging
    - Live Dashboard Camera Frame
    - Event Tracking
    - Alert/Event History
    - GRU Prediction Logging

IMPORTANT:
    Existing fusion risk remains the primary safety risk.

    GRU prediction is added as an independent temporal
    prediction signal in Step 4.

=============================================================
"""

import os
import csv
import time
from datetime import datetime

import cv2

from src.camera import Camera
from src.detector import FaceDetector
from src.metrics import DriverMetrics
from src.head_pose import HeadPoseEstimator
from src.gaze import GazeEstimator
from src.reliability import SignalReliabilityEstimator
from src.fusion import AdaptiveMultimodalFusion
from src.temporal import TemporalFatiguePredictor
from src.intervention import AdaptiveSafetyIntervention
from src.logger import SessionLogger
from src.event_tracker import EventTracker
from src.gru_predictor import GRUFatiguePredictor


# =============================================================
# VERSION
# =============================================================

VERSION = "v1.8 - STEP 4"


# =============================================================
# LIVE DASHBOARD FRAME
# =============================================================

LIVE_FRAME_DIRECTORY = "logs"

LIVE_FRAME_PATH = os.path.join(
    LIVE_FRAME_DIRECTORY,
    "live_frame.jpg",
)

LIVE_FRAME_TEMP_PATH = os.path.join(
    LIVE_FRAME_DIRECTORY,
    "live_frame_tmp.jpg",
)


# =============================================================
# GRU PREDICTION LOG
# =============================================================

GRU_LOG_DIRECTORY = "logs"

GRU_LOG_PATH = os.path.join(
    GRU_LOG_DIRECTORY,
    "gru_predictions.csv",
)


# =============================================================
# SAVE LIVE FRAME
# =============================================================

def save_live_frame(frame):
    """
    Save processed camera frame for Streamlit.

    Uses a temporary file before replacing the old frame.
    """

    try:

        os.makedirs(
            LIVE_FRAME_DIRECTORY,
            exist_ok=True,
        )

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                75,
            ],
        )

        if not success:
            return

        with open(
            LIVE_FRAME_TEMP_PATH,
            "wb",
        ) as file:

            file.write(
                encoded.tobytes()
            )

        os.replace(
            LIVE_FRAME_TEMP_PATH,
            LIVE_FRAME_PATH,
        )

    except Exception:

        pass


# =============================================================
# SAFE FLOAT
# =============================================================

def safe_float(
    value,
    default=0.0,
):

    try:

        result = float(value)

        if result != result:
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):

        return default


# =============================================================
# GRU LOG INITIALIZATION
# =============================================================

def initialize_gru_log():

    try:

        os.makedirs(
            GRU_LOG_DIRECTORY,
            exist_ok=True,
        )

        if not os.path.exists(
            GRU_LOG_PATH
        ):

            with open(
                GRU_LOG_PATH,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow(
                    [
                        "timestamp",

                        "fusion_fatigue_risk",

                        "gru_prediction",

                        "gru_confidence",

                        "gru_classification",

                        "sequence_length",

                        "sequence_capacity",

                        "sequence_ready",

                        "model_available",

                        "model_loaded",

                        "model_trained",

                        "prediction_count",
                    ]
                )

    except Exception as error:

        print(
            f"GRU log initialization error: {error}"
        )


# =============================================================
# LOG GRU PREDICTION
# =============================================================

def log_gru_prediction(
    timestamp,
    fusion_fatigue_risk,
    gru_values,
):

    try:

        os.makedirs(
            GRU_LOG_DIRECTORY,
            exist_ok=True,
        )

        with open(
            GRU_LOG_PATH,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    datetime.fromtimestamp(
                        timestamp
                    ).isoformat(),

                    f"{safe_float(fusion_fatigue_risk):.6f}",

                    f"{safe_float(gru_values.get('prediction', 0.0)):.6f}",

                    f"{safe_float(gru_values.get('confidence', 0.0)):.6f}",

                    gru_values.get(
                        "classification",
                        "UNKNOWN",
                    ),

                    gru_values.get(
                        "sequence_length",
                        0,
                    ),

                    gru_values.get(
                        "sequence_capacity",
                        0,
                    ),

                    gru_values.get(
                        "sequence_ready",
                        False,
                    ),

                    gru_values.get(
                        "model_available",
                        False,
                    ),

                    gru_values.get(
                        "model_loaded",
                        False,
                    ),

                    gru_values.get(
                        "model_trained",
                        False,
                    ),

                    gru_values.get(
                        "prediction_count",
                        0,
                    ),
                ]
            )

    except Exception as error:

        print(
            f"GRU logging error: {error}"
        )


# =============================================================
# MAIN
# =============================================================

def main():

    print("=" * 70)

    print(
        "ADAPTIVE-DMS"
    )

    print(
        "Adaptive Multimodal Driver State Monitoring "
        "and Predictive Safety Intervention System"
    )

    print(
        VERSION
    )

    print("=" * 70)

    print(
        "Starting camera..."
    )

    print(
        "Starting session logger..."
    )

    print(
        "Starting event tracker..."
    )

    print(
        "Starting GRU temporal predictor..."
    )

    print(
        "Starting live dashboard frame..."
    )

    print(
        "Press Q to quit."
    )

    print()

    # =========================================================
    # CAMERA
    # =========================================================

    camera = Camera(
        source=0,
        width=960,
        height=540,
    )

    # =========================================================
    # FACE DETECTOR
    # =========================================================

    detector = FaceDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # =========================================================
    # DRIVER METRICS
    # =========================================================

    metrics = DriverMetrics(
        ear_threshold=0.21,
        mar_threshold=0.60,
        perclos_window_seconds=30,
        microsleep_threshold=1.5,
    )

    # =========================================================
    # HEAD POSE
    # =========================================================

    head_pose = HeadPoseEstimator()

    # =========================================================
    # GAZE
    # =========================================================

    gaze = GazeEstimator(
        horizontal_left_threshold=0.35,
        horizontal_right_threshold=0.65,
        vertical_up_threshold=0.35,
        vertical_down_threshold=0.65,
        gaze_away_threshold=1.0,
        smoothing_window=5,
    )

    # =========================================================
    # RELIABILITY
    # =========================================================

    reliability = SignalReliabilityEstimator()

    # =========================================================
    # ADAPTIVE FUSION
    # =========================================================

    fusion = AdaptiveMultimodalFusion()

    # =========================================================
    # TEMPORAL ANALYSIS
    # =========================================================

    temporal = TemporalFatiguePredictor(
        history_seconds=30.0,
        sample_interval=0.5,
        short_window_seconds=5.0,
        medium_window_seconds=15.0,
        increasing_threshold=0.015,
        high_risk_threshold=0.60,
    )

    # =========================================================
    # GRU TEMPORAL PREDICTOR
    # =========================================================

    gru_predictor = GRUFatiguePredictor(
        sequence_length=20,
        model_path=(
            "models/"
            "gru_fatigue_model.keras"
        ),
        prediction_threshold=0.50,
        minimum_sequence_ratio=0.50,
    )

    # =========================================================
    # GRU STATUS
    # =========================================================

    gru_status = (
        gru_predictor.get_status()
    )

    print(
        "GRU STATUS"
    )

    print(
        f"TensorFlow available: "
        f"{gru_status['tensorflow_available']}"
    )

    print(
        f"Model available: "
        f"{gru_status['model_available']}"
    )

    print(
        f"Model loaded: "
        f"{gru_status['model_loaded']}"
    )

    print(
        f"Model trained: "
        f"{gru_status['model_trained']}"
    )

    print(
        f"Sequence length: "
        f"{gru_status['sequence_length']}"
    )

    print(
        f"Feature count: "
        f"{gru_status['feature_count']}"
    )

    print()

    # =========================================================
    # SAFETY INTERVENTION
    # =========================================================

    intervention = AdaptiveSafetyIntervention(
        low_threshold=0.20,
        moderate_threshold=0.35,
        high_threshold=0.55,
        critical_threshold=0.75,
        minimum_reliability=0.40,
        persistence_seconds=1.5,
        cooldown_seconds=5.0,
        eye_closure_trigger_seconds=1.5,
    )

    # =========================================================
    # SESSION LOGGER
    # =========================================================

    logger = SessionLogger(
        log_directory="logs"
    )

    # =========================================================
    # EVENT TRACKER
    # =========================================================

    event_tracker = EventTracker(
        log_directory="logs",
        eye_closure_threshold=1.5,
        event_cooldown=2.0,
    )

    # =========================================================
    # GRU LOG
    # =========================================================

    initialize_gru_log()

    print(
        f"Session log: "
        f"{logger.file_path}"
    )

    print(
        f"Event log: "
        f"{event_tracker.file_path}"
    )

    print(
        f"GRU prediction log: "
        f"{GRU_LOG_PATH}"
    )

    print()

    # =========================================================
    # FPS
    # =========================================================

    previous_time = time.time()

    # =========================================================
    # EYE CLOSURE TIMER
    # =========================================================

    eyes_closed_start_time = None

    current_eye_closure_duration = 0.0

    # =========================================================
    # LIVE FRAME CONTROL
    # =========================================================

    frame_counter = 0

    live_frame_interval = 2

    # =========================================================
    # MAIN LOOP
    # =========================================================

    try:

        while True:

            # =================================================
            # CAMERA FRAME
            # =================================================

            frame = camera.read()

            if frame is None:

                print(
                    "Unable to read frame from camera."
                )

                break

            # =================================================
            # FRAME COUNTER
            # =================================================

            frame_counter += 1

            # =================================================
            # CURRENT TIME
            # =================================================

            current_time = time.time()

            # =================================================
            # FPS
            # =================================================

            delta_time = (
                current_time
                - previous_time
            )

            if delta_time > 0:

                fps = 1.0 / delta_time

            else:

                fps = 0.0

            previous_time = current_time

            # =================================================
            # FACE DETECTION
            # =================================================

            results = detector.process(
                frame
            )

            face_detected = bool(
                results.multi_face_landmarks
            )

            # =================================================
            # DEFAULT DRIVER VALUES
            # =================================================

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

            # =================================================
            # DEFAULT HEAD POSE
            # =================================================

            pose_values = {

                "pitch": 0.0,

                "yaw": 0.0,

                "roll": 0.0,

                "valid": False,
            }

            # =================================================
            # DEFAULT GAZE
            # =================================================

            gaze_values = {

                "horizontal_ratio": 0.5,

                "vertical_ratio": 0.5,

                "gaze_direction": "UNKNOWN",

                "gaze_away_duration": 0.0,

                "prolonged_gaze_away": False,
            }

            # =================================================
            # FACE DETECTED
            # =================================================

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

                driver_values = (
                    metrics.calculate(
                        landmarks,
                        timestamp=current_time,
                    )
                )

                # ---------------------------------------------
                # HEAD POSE
                # ---------------------------------------------

                pose_values = (
                    head_pose.estimate(
                        landmarks,
                        frame.shape[1],
                        frame.shape[0],
                    )
                )

                # ---------------------------------------------
                # GAZE
                # ---------------------------------------------

                gaze_values = (
                    gaze.estimate(
                        landmarks,
                        timestamp=current_time,
                    )
                )

                # ---------------------------------------------
                # DRAW LANDMARKS
                # ---------------------------------------------

                frame = (
                    detector.draw_landmarks(
                        frame,
                        results,
                    )
                )

            # =================================================
            # DIRECT EYE CLOSURE TIMER
            # =================================================

            if (
                face_detected
                and driver_values[
                    "eyes_closed"
                ]
            ):

                if eyes_closed_start_time is None:

                    eyes_closed_start_time = (
                        current_time
                    )

                current_eye_closure_duration = (
                    current_time
                    - eyes_closed_start_time
                )

            else:

                eyes_closed_start_time = None

                current_eye_closure_duration = 0.0

            # =================================================
            # PERCLOS SAMPLE COUNT
            # =================================================

            try:

                perclos_sample_count = len(
                    metrics.eye_history
                )

            except AttributeError:

                perclos_sample_count = 0

            # =================================================
            # SIGNAL RELIABILITY
            # =================================================

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

            # =================================================
            # ADAPTIVE MULTIMODAL FUSION
            # =================================================

            fusion_values = (
                fusion.calculate(
                    driver_values=driver_values,
                    pose_values=pose_values,
                    gaze_values=gaze_values,
                    reliability_values=(
                        reliability_values
                    ),
                )
            )

            # =================================================
            # EXISTING FUSION FATIGUE RISK
            # =================================================

            fusion_fatigue_risk = safe_float(
                fusion_values[
                    "fatigue_risk"
                ]
            )

            # =================================================
            # TEMPORAL ANALYSIS
            # =================================================

            temporal_values = (
                temporal.update(
                    fatigue_risk=(
                        fusion_fatigue_risk
                    ),
                    timestamp=current_time,
                )
            )

            # =================================================
            # GRU TEMPORAL PREDICTION
            # =================================================

            try:

                gru_values = (
                    gru_predictor.update(
                        driver_values=(
                            driver_values
                        ),
                        pose_values=(
                            pose_values
                        ),
                        gaze_values=(
                            gaze_values
                        ),
                        reliability_values=(
                            reliability_values
                        ),
                        fatigue_risk=(
                            fusion_fatigue_risk
                        ),
                    )
                )

            except Exception as error:

                print(
                    f"GRU prediction error: {error}"
                )

                gru_values = {

                    "prediction": (
                        fusion_fatigue_risk
                    ),

                    "fatigue_risk": (
                        fusion_fatigue_risk
                    ),

                    "confidence": 0.0,

                    "classification": "UNKNOWN",

                    "sequence_length": 0,

                    "sequence_capacity": 20,

                    "sequence_ready": False,

                    "model_available": False,

                    "model_loaded": False,

                    "model_trained": False,

                    "prediction_count": 0,
                }

            # =================================================
            # GRU LOGGING
            # =================================================

            log_gru_prediction(
                timestamp=current_time,
                fusion_fatigue_risk=(
                    fusion_fatigue_risk
                ),
                gru_values=gru_values,
            )

            # =================================================
            # SAFETY INTERVENTION
            #
            # IMPORTANT:
            # Existing fusion risk remains primary safety risk.
            # =================================================

            intervention_values = (
                intervention.update(
                    fatigue_risk=(
                        fusion_fatigue_risk
                    ),
                    temporal_state=(
                        temporal_values[
                            "state"
                        ]
                    ),
                    reliability=(
                        reliability_values[
                            "overall_reliability"
                        ]
                    ),
                    timestamp=current_time,
                    eyes_closed=(
                        face_detected
                        and
                        driver_values[
                            "eyes_closed"
                        ]
                    ),
                    eye_closure_duration=(
                        current_eye_closure_duration
                    ),
                )
            )

            # =================================================
            # SESSION LOGGING
            # =================================================

            logger.log(
                timestamp=current_time,
                driver_values=driver_values,
                pose_values=pose_values,
                gaze_values=gaze_values,
                reliability_values=reliability_values,
                fusion_values=fusion_values,
                temporal_values=temporal_values,
                intervention_values=(
                    intervention_values
                ),
                eye_closure_duration=(
                    current_eye_closure_duration
                ),
            )

            # =================================================
            # EVENT TRACKING
            # =================================================

            event_values = (
                event_tracker.update(
                    timestamp=current_time,
                    driver_values=driver_values,
                    gaze_values=gaze_values,
                    fusion_values=fusion_values,
                    intervention_values=(
                        intervention_values
                    ),
                    eye_closure_duration=(
                        current_eye_closure_duration
                    ),
                )
            )

            # =================================================
            # LEFT PANEL
            # =================================================

            if face_detected:

                cv2.putText(
                    frame,
                    (
                        f"EAR: "
                        f"{driver_values['ear']:.3f}"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"MAR: "
                        f"{driver_values['mar']:.3f}"
                    ),
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

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

                eye_status = (
                    "EYES CLOSED"
                    if driver_values[
                        "eyes_closed"
                    ]
                    else "EYES OPEN"
                )

                eye_color = (
                    (0, 165, 255)
                    if driver_values[
                        "eyes_closed"
                    ]
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

                cv2.putText(
                    frame,
                    (
                        f"Eye Closed: "
                        f"{current_eye_closure_duration:.1f}s"
                    ),
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (
                        (0, 0, 255)
                        if current_eye_closure_duration >= 1.0
                        else (255, 255, 255)
                    ),
                    2,
                )

                if driver_values[
                    "yawning"
                ]:

                    cv2.putText(
                        frame,
                        "YAWNING",
                        (20, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 165, 255),
                        2,
                    )

                if driver_values[
                    "microsleep"
                ]:

                    cv2.putText(
                        frame,
                        (
                            "MICROSLEEP "
                            f"{driver_values['microsleep_duration']:.1f}s"
                        ),
                        (20, 255),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 0, 255),
                        2,
                    )

                if pose_values[
                    "valid"
                ]:

                    cv2.putText(
                        frame,
                        (
                            f"Pitch: "
                            f"{pose_values['pitch']:.1f}"
                        ),
                        (20, 290),
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
                        (20, 315),
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
                        (20, 340),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    (
                        f"Gaze: "
                        f"{gaze_values['gaze_direction']}"
                    ),
                    (20, 375),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 0, 255),
                    2,
                )

                if gaze_values[
                    "prolonged_gaze_away"
                ]:

                    cv2.putText(
                        frame,
                        (
                            "GAZE AWAY "
                            f"{gaze_values['gaze_away_duration']:.1f}s"
                        ),
                        (20, 410),
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
                "ADAPTIVE DMS",
                (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            # =================================================
            # FUSION FATIGUE RISK
            # =================================================

            risk_level = (
                fusion_values[
                    "risk_level"
                ]
            )

            cv2.putText(
                frame,
                (
                    f"Fusion Risk: "
                    f"{fusion_fatigue_risk:.2f}"
                ),
                (500, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            # =================================================
            # RISK COLOR
            # =================================================

            if risk_level == "NORMAL":

                risk_color = (
                    0,
                    255,
                    0,
                )

            elif risk_level in (
                "LOW",
                "MODERATE",
            ):

                risk_color = (
                    0,
                    255,
                    255,
                )

            elif risk_level == "HIGH":

                risk_color = (
                    0,
                    165,
                    255,
                )

            else:

                risk_color = (
                    0,
                    0,
                    255,
                )

            cv2.putText(
                frame,
                (
                    f"Risk Level: "
                    f"{risk_level}"
                ),
                (500, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                risk_color,
                2,
            )

            # =================================================
            # GRU PREDICTION
            # =================================================

            gru_prediction = safe_float(
                gru_values.get(
                    "prediction",
                    0.0,
                )
            )

            gru_confidence = safe_float(
                gru_values.get(
                    "confidence",
                    0.0,
                )
            )

            gru_classification = (
                gru_values.get(
                    "classification",
                    "UNKNOWN",
                )
            )

            gru_sequence_length = (
                gru_values.get(
                    "sequence_length",
                    0,
                )
            )

            gru_sequence_capacity = (
                gru_values.get(
                    "sequence_capacity",
                    20,
                )
            )

            cv2.putText(
                frame,
                (
                    f"GRU Risk: "
                    f"{gru_prediction:.2f}"
                ),
                (500, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"GRU Class: "
                    f"{gru_classification}"
                ),
                (500, 148),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"GRU Confidence: "
                    f"{gru_confidence:.2f}"
                ),
                (500, 174),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                (
                    f"GRU Sequence: "
                    f"{gru_sequence_length}/"
                    f"{gru_sequence_capacity}"
                ),
                (500, 198),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
            )

            # =================================================
            # RELIABILITY
            # =================================================

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
                (500, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
            )

            # =================================================
            # TEMPORAL
            # =================================================

            temporal_state = (
                temporal_values[
                    "state"
                ]
            )

            cv2.putText(
                frame,
                (
                    f"Temporal: "
                    f"{temporal_state}"
                ),
                (500, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"5s Avg: "
                    f"{temporal_values['short_average']:.2f}"
                ),
                (500, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                (
                    f"15s Avg: "
                    f"{temporal_values['medium_average']:.2f}"
                ),
                (500, 298),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                (
                    f"Trend: "
                    f"{temporal_values['trend']:.3f}"
                ),
                (500, 321),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
            )

            # =================================================
            # SAFETY INTERVENTION
            # =================================================

            intervention_level = (
                intervention_values[
                    "level"
                ]
            )

            intervention_action = (
                intervention_values[
                    "action"
                ]
            )

            cv2.putText(
                frame,
                "SAFETY INTERVENTION",
                (500, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Level: "
                    f"{intervention_level}"
                ),
                (500, 375),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                risk_color,
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Action: "
                    f"{intervention_action}"
                ),
                (500, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                (
                    f"Alerts: "
                    f"{intervention_values['alert_count']}"
                ),
                (500, 425),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
            )

            # =================================================
            # EYE CLOSURE WARNING
            # =================================================

            if (
                current_eye_closure_duration
                >= 1.0
            ):

                cv2.putText(
                    frame,
                    "EYE CLOSURE WARNING",
                    (500, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2,
                )

            # =================================================
            # HIGH / CRITICAL WARNING
            # =================================================

            if intervention_level in (
                "HIGH",
                "CRITICAL",
            ):

                cv2.rectangle(
                    frame,
                    (10, 10),
                    (
                        frame.shape[1] - 10,
                        frame.shape[0] - 10,
                    ),
                    (0, 0, 255),
                    3,
                )

                cv2.putText(
                    frame,
                    intervention_values[
                        "message"
                    ],
                    (500, 480),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2,
                )

            elif intervention_level == "MODERATE":

                cv2.putText(
                    frame,
                    intervention_values[
                        "message"
                    ],
                    (500, 480),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    2,
                )

            # =================================================
            # EVENT COUNT
            # =================================================

            try:

                event_count = event_values[
                    "event_count"
                ]

            except Exception:

                event_count = 0

            cv2.putText(
                frame,
                (
                    f"Events: "
                    f"{event_count}"
                ),
                (500, 505),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
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
                0.55,
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
                0.55,
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
                "ADAPTIVE-DMS v1.8 | Press Q to quit",
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
            # SAVE LIVE FRAME
            # =================================================

            if (
                frame_counter
                % live_frame_interval
                == 0
            ):

                save_live_frame(
                    frame
                )

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                "ADAPTIVE-DMS",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

    # =========================================================
    # KEYBOARD INTERRUPT
    # =========================================================

    except KeyboardInterrupt:

        print()

        print(
            "Program interrupted by user."
        )

    # =========================================================
    # CLEANUP
    # =========================================================

    finally:

        # =====================================================
        # CLOSE SESSION LOGGER
        # =====================================================

        try:

            logger.close()

            print(
                f"Session log saved: "
                f"{logger.file_path}"
            )

        except Exception as error:

            print(
                f"Logger close error: "
                f"{error}"
            )

        # =====================================================
        # CLOSE EVENT TRACKER
        # =====================================================

        try:

            event_tracker.close()

            print(
                f"Event log saved: "
                f"{event_tracker.file_path}"
            )

        except Exception as error:

            print(
                f"Event tracker close error: "
                f"{error}"
            )

        # =====================================================
        # RELEASE DETECTOR
        # =====================================================

        try:

            detector.close()

        except Exception as error:

            print(
                f"Detector close error: "
                f"{error}"
            )

        # =====================================================
        # RELEASE CAMERA
        # =====================================================

        try:

            camera.release()

        except Exception as error:

            print(
                f"Camera release error: "
                f"{error}"
            )

        # =====================================================
        # CLOSE OPENCV WINDOWS
        # =====================================================

        cv2.destroyAllWindows()

        print()

        print("=" * 70)

        print(
            "Camera released."
        )

        print(
            "ADAPTIVE-DMS v1.8 stopped."
        )

        print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()