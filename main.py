"""
=============================================================
ADAPTIVE-DMS
=============================================================

Adaptive Multimodal Driver State Monitoring
and Predictive Safety Intervention System.

Version:
    v1.8 + Step 7A Steering Integration

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
    - Steering Behaviour Analysis
    - Steering Irregularity Detection
    - Steering Reliability
    - Steering Event Logging

STEERING INPUT:

    A = steer LEFT
    D = steer RIGHT
    S = return to CENTER
    Q = quit

IMPORTANT:

    The current steering implementation uses keyboard input
    as a simulator/controller interface.

    It does NOT claim to estimate real steering-wheel angle
    from the webcam.

    Later this input can be replaced by:
        - steering wheel controller
        - simulator API
        - CAN/vehicle interface
        - game controller
        - other external steering source

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
from src.steering import SteeringBehaviourAnalyzer
from src.heart_rate import NonContactHeartRateEstimator
from src.respiration import NonContactRespirationEstimator
from src.respiration_logger import RespirationLogger
from src.stress_indicator import DriverStressIndicator
from src.driver_state_estimator import DriverStateEstimator
from src.safety_decision_engine import AdaptiveSafetyDecisionEngine
from src.intervention_prioritizer import InterventionPrioritizer


# =============================================================
# VERSION
# =============================================================

VERSION = "v2.0 + STEP 9C"

# =============================================================
# CAMERA WINDOW CONTROL
# =============================================================

WINDOW_NAME = "ADAPTIVE-DMS"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


def set_camera_window_fullscreen(fullscreen):
    try:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )

        if not fullscreen:
            cv2.resizeWindow(
                WINDOW_NAME,
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
            )

        return True

    except Exception as error:
        print(f"Camera window control error: {error}")
        return False


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
# GRU LOG
# =============================================================

GRU_LOG_DIRECTORY = "logs"

GRU_LOG_PATH = os.path.join(
    GRU_LOG_DIRECTORY,
    "gru_predictions.csv",
)


# =============================================================
# STEERING LOG
# =============================================================

STEERING_LOG_DIRECTORY = "logs"

STEERING_LOG_PATH = os.path.join(
    STEERING_LOG_DIRECTORY,
    "steering_predictions.csv",
)


# =============================================================
# HEART RATE LOG
# =============================================================

HEART_RATE_LOG_DIRECTORY = "logs"

HEART_RATE_LOG_PATH = os.path.join(
    HEART_RATE_LOG_DIRECTORY,
    "heart_rate_predictions.csv",
)


# =============================================================
# RESPIRATION LOG
# =============================================================

RESPIRATION_LOG_DIRECTORY = "logs"

RESPIRATION_LOG_PATH = os.path.join(
    RESPIRATION_LOG_DIRECTORY,
    "respiration_predictions.csv",
)


# =============================================================
# DRIVER STATE LOG
# =============================================================

DRIVER_STATE_LOG_DIRECTORY = "logs"

DRIVER_STATE_LOG_PATH = os.path.join(
    DRIVER_STATE_LOG_DIRECTORY,
    "driver_state_predictions.csv",
)


# =============================================================
# STEP 9B/9C SAFETY DECISION LOG
# =============================================================

SAFETY_DECISION_LOG_DIRECTORY = "logs"

SAFETY_DECISION_LOG_PATH = os.path.join(
    SAFETY_DECISION_LOG_DIRECTORY,
    "safety_decision_predictions.csv",
)


# =============================================================
# DRIVER STATE LOG
# =============================================================

def initialize_driver_state_log():

    try:

        os.makedirs(
            DRIVER_STATE_LOG_DIRECTORY,
            exist_ok=True,
        )

        if not os.path.exists(DRIVER_STATE_LOG_PATH):

            with open(
                DRIVER_STATE_LOG_PATH,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "timestamp",
                    "driver_state",
                    "state_score",
                    "confidence",
                    "reliability",
                    "risk_level",
                    "state_reason",
                    "signal_coverage",
                ])

    except Exception as error:

        print(
            f"Driver state log initialization error: {error}"
        )


def log_driver_state(
    timestamp,
    driver_state_values,
):

    try:

        os.makedirs(
            DRIVER_STATE_LOG_DIRECTORY,
            exist_ok=True,
        )

        with open(
            DRIVER_STATE_LOG_PATH,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                datetime.fromtimestamp(
                    timestamp
                ).isoformat(),
                driver_state_values.get(
                    "driver_state",
                    "UNKNOWN",
                ),
                f"{safe_float(driver_state_values.get('state_score', 0.0)):.6f}",
                f"{safe_float(driver_state_values.get('confidence', 0.0)):.6f}",
                f"{safe_float(driver_state_values.get('reliability', 0.0)):.6f}",
                driver_state_values.get(
                    "risk_level",
                    "UNKNOWN",
                ),
                driver_state_values.get(
                    "state_reason",
                    "NO_DATA",
                ),
                f"{safe_float(driver_state_values.get('signal_coverage', 0.0)):.6f}",
            ])

    except Exception as error:

        print(
            f"Driver state logging error: {error}"
        )


# =============================================================
# STEP 9B/9C SAFETY DECISION LOGGING
# =============================================================

def initialize_safety_decision_log():

    try:

        os.makedirs(
            SAFETY_DECISION_LOG_DIRECTORY,
            exist_ok=True,
        )

        if not os.path.exists(
            SAFETY_DECISION_LOG_PATH
        ):

            with open(
                SAFETY_DECISION_LOG_PATH,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "timestamp",
                    "decision",
                    "decision_score",
                    "confidence",
                    "reliability",
                    "reason",
                    "priority",
                    "priority_score",
                    "action",
                    "driver_state",
                    "state_score",
                ])

    except Exception as error:

        print(
            f"Safety decision log initialization error: {error}"
        )


def log_safety_decision(
    timestamp,
    decision_values,
    priority_values,
):

    try:

        os.makedirs(
            SAFETY_DECISION_LOG_DIRECTORY,
            exist_ok=True,
        )

        with open(
            SAFETY_DECISION_LOG_PATH,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                datetime.fromtimestamp(
                    timestamp
                ).isoformat(),
                decision_values.get(
                    "decision",
                    "NO_ACTION",
                ),
                f"{safe_float(decision_values.get('decision_score', 0.0)):.6f}",
                f"{safe_float(decision_values.get('confidence', 0.0)):.6f}",
                f"{safe_float(decision_values.get('reliability', 0.0)):.6f}",
                decision_values.get(
                    "reason",
                    "NO_DATA",
                ),
                priority_values.get(
                    "priority",
                    "NONE",
                ),
                priority_values.get(
                    "priority_score",
                    0,
                ),
                priority_values.get(
                    "action",
                    "NO_ACTION",
                ),
                priority_values.get(
                    "driver_state",
                    "UNKNOWN",
                ),
                f"{safe_float(priority_values.get('state_score', 0.0)):.6f}",
            ])

    except Exception as error:

        print(
            f"Safety decision logging error: {error}"
        )


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
# SAVE LIVE FRAME
# =============================================================

def save_live_frame(frame):

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
# INITIALIZE GRU LOG
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
# INITIALIZE STEERING LOG
# =============================================================

def initialize_steering_log():

    try:

        os.makedirs(
            STEERING_LOG_DIRECTORY,
            exist_ok=True,
        )

        if not os.path.exists(
            STEERING_LOG_PATH
        ):

            with open(
                STEERING_LOG_PATH,
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
                        "steering_angle",
                        "steering_change",
                        "steering_rate",
                        "steering_variability",
                        "sudden_correction",
                        "irregularity_score",
                        "reliability",
                        "driving_state",
                        "sample_count",
                    ]
                )

    except Exception as error:

        print(
            f"Steering log initialization error: {error}"
        )


# =============================================================
# LOG STEERING
# =============================================================

def log_steering(
    timestamp,
    steering_values,
):

    try:

        os.makedirs(
            STEERING_LOG_DIRECTORY,
            exist_ok=True,
        )

        with open(
            STEERING_LOG_PATH,
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

                    f"{safe_float(steering_values.get('steering_angle', 0.0)):.6f}",

                    f"{safe_float(steering_values.get('steering_change', 0.0)):.6f}",

                    f"{safe_float(steering_values.get('steering_rate', 0.0)):.6f}",

                    f"{safe_float(steering_values.get('steering_variability', 0.0)):.6f}",

                    steering_values.get(
                        "sudden_correction",
                        False,
                    ),

                    f"{safe_float(steering_values.get('irregularity_score', 0.0)):.6f}",

                    f"{safe_float(steering_values.get('reliability', 0.0)):.6f}",

                    steering_values.get(
                        "driving_state",
                        "UNKNOWN",
                    ),

                    steering_values.get(
                        "sample_count",
                        0,
                    ),
                ]
            )

    except Exception as error:

        print(
            f"Steering logging error: {error}"
        )


# =============================================================
# INITIALIZE PHYSIOLOGICAL LOGS
# =============================================================

def initialize_heart_rate_log():
    try:
        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(HEART_RATE_LOG_PATH):
            with open(HEART_RATE_LOG_PATH, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp",
                    "heart_rate_bpm",
                    "raw_signal",
                    "filtered_signal",
                    "signal_quality",
                    "reliability",
                    "state",
                    "sample_count",
                    "signal_ready",
                    "roi_available",
                    "method",
                ])
    except Exception as error:
        print(f"Heart-rate log initialization error: {error}")


def log_heart_rate(timestamp, heart_rate_values):
    try:
        with open(HEART_RATE_LOG_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.fromtimestamp(timestamp).isoformat(),
                f"{safe_float(heart_rate_values.get('heart_rate_bpm', 0.0)):.6f}",
                f"{safe_float(heart_rate_values.get('raw_signal', 0.0)):.6f}",
                f"{safe_float(heart_rate_values.get('filtered_signal', 0.0)):.6f}",
                f"{safe_float(heart_rate_values.get('signal_quality', 0.0)):.6f}",
                f"{safe_float(heart_rate_values.get('reliability', 0.0)):.6f}",
                heart_rate_values.get("state", "UNKNOWN"),
                heart_rate_values.get("sample_count", 0),
                heart_rate_values.get("signal_ready", False),
                heart_rate_values.get("roi_available", False),
                heart_rate_values.get("method", "GREEN_CHANNEL_RPPG"),
            ])
    except Exception as error:
        print(f"Heart-rate logging error: {error}")


def initialize_respiration_log():
    try:
        RespirationLogger(RESPIRATION_LOG_PATH)
    except Exception as error:
        print(f"Respiration logger initialization error: {error}")


def log_respiration(timestamp, respiration_values):
    try:
        logger_instance = RespirationLogger(RESPIRATION_LOG_PATH)
        logger_instance.log(
            timestamp=datetime.fromtimestamp(timestamp).isoformat(),
            respiration_values=respiration_values,
        )
    except Exception as error:
        print(f"Respiration logging error: {error}")


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
        "Starting steering behaviour analyzer..."
    )

    print(
        "Starting live dashboard frame..."
    )

    print(
        "Starting safety decision engine..."
    )

    print(
        "Starting intervention prioritizer..."
    )

    print()

    print(
        "STEERING CONTROLS:"
    )

    print(
        "  A = LEFT"
    )

    print(
        "  D = RIGHT"
    )

    print(
        "  S = CENTER"
    )

    print(
        "  Q = QUIT"
    )

    print()

    # =========================================================
    # CAMERA
    # =========================================================

    camera = Camera(
        source=0,
        width=1280,
        height=720,
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
    # GRU PREDICTOR
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
    # STEERING ANALYZER
    # =========================================================

    steering_analyzer = (
        SteeringBehaviourAnalyzer(
            history_size=50,
            maximum_steering_angle=450.0,
            sudden_change_threshold=45.0,
            high_rate_threshold=180.0,
            irregularity_threshold=0.55,
        )
    )

    # =========================================================
    # HEART RATE ESTIMATOR
    # =========================================================

    heart_rate = NonContactHeartRateEstimator(
        history_seconds=12.0,
        sample_rate=30.0,
        min_bpm=45.0,
        max_bpm=180.0,
    )

    # =========================================================
    # RESPIRATION ESTIMATOR
    # =========================================================

    respiration = NonContactRespirationEstimator(
        history_seconds=20.0,
        sample_rate=30.0,
        min_breaths_per_minute=6.0,
        max_breaths_per_minute=40.0,
        minimum_samples=180,
    )

    # =========================================================
    # STEP 8F - STRESS INDICATOR
    # =========================================================

    stress_indicator = DriverStressIndicator()

    # =========================================================
    # STEP 9A - DRIVER STATE ESTIMATOR
    # =========================================================

    driver_state_estimator = DriverStateEstimator()


    # =========================================================
    # STEP 9B - ADAPTIVE SAFETY DECISION ENGINE
    # =========================================================

    safety_decision_engine = (
        AdaptiveSafetyDecisionEngine()
    )

    # =========================================================
    # STEP 9C - INTERVENTION PRIORITIZER
    # =========================================================

    intervention_prioritizer = (
        InterventionPrioritizer()
    )

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
    # INITIALIZE LOGS
    # =========================================================

    initialize_gru_log()

    initialize_steering_log()

    initialize_heart_rate_log()

    initialize_respiration_log()

    initialize_driver_state_log()

    initialize_safety_decision_log()

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

    print(
        f"Steering log: "
        f"{STEERING_LOG_PATH}"
    )

    print(
        f"Heart-rate log: "
        f"{HEART_RATE_LOG_PATH}"
    )

    print(
        f"Respiration log: "
        f"{RESPIRATION_LOG_PATH}"
    )

    print(
        f"Driver state log: "
        f"{DRIVER_STATE_LOG_PATH}"
    )

    print(
        f"Safety decision log: "
        f"{SAFETY_DECISION_LOG_PATH}"
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
    # STEERING STATE
    # =========================================================

    steering_angle = 0.0

    steering_step = 15.0

    steering_input_available = True

    # =========================================================
    # CAMERA WINDOW
    # =========================================================

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )

    fullscreen = False

    print("CAMERA WINDOW CONTROLS:")
    print("  F   = FULLSCREEN")
    print("  ESC = WINDOWED")
    print("  M   = MAXIMIZE")
    print("  R   = RESIZE TO 1280x720")
    print()

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
            # NON-CONTACT HEART RATE
            # =================================================

            try:
                heart_rate_values = heart_rate.update(
                    frame=frame,
                    timestamp=current_time,
                    input_available=True,
                )
            except Exception as error:
                print(f"Heart-rate estimation error: {error}")
                heart_rate_values = {
                    "heart_rate_bpm": 0.0,
                    "raw_signal": 0.0,
                    "filtered_signal": 0.0,
                    "signal_quality": 0.0,
                    "reliability": 0.0,
                    "state": "ERROR",
                    "sample_count": 0,
                    "signal_ready": False,
                    "roi_available": False,
                    "method": "GREEN_CHANNEL_RPPG",
                }

            log_heart_rate(
                timestamp=current_time,
                heart_rate_values=heart_rate_values,
            )

            # =================================================
            # NON-CONTACT RESPIRATION
            # =================================================

            try:
                respiration_values = respiration.update(
                    frame=frame,
                    timestamp=current_time,
                    input_available=True,
                )
            except Exception as error:
                print(f"Respiration estimation error: {error}")
                respiration_values = {
                    "respiration_rate_bpm": 0.0,
                    "raw_signal": 0.0,
                    "filtered_signal": 0.0,
                    "signal_quality": 0.0,
                    "reliability": 0.0,
                    "state": "ERROR",
                    "sample_count": 0,
                    "signal_ready": False,
                    "roi_available": False,
                    "method": "VISUAL_INTENSITY_RESPIRATION",
                }

            log_respiration(
                timestamp=current_time,
                respiration_values=respiration_values,
            )

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
            # FUSION RISK
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

                    "prediction":
                        fusion_fatigue_risk,

                    "fatigue_risk":
                        fusion_fatigue_risk,

                    "confidence":
                        0.0,

                    "classification":
                        "UNKNOWN",

                    "sequence_length":
                        0,

                    "sequence_capacity":
                        20,

                    "sequence_ready":
                        False,

                    "model_available":
                        False,

                    "model_loaded":
                        False,

                    "model_trained":
                        False,

                    "prediction_count":
                        0,
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
            # STEERING ANALYSIS
            # =================================================

            try:

                steering_values = (
                    steering_analyzer.update(
                        steering_angle=(
                            steering_angle
                        ),
                        timestamp=current_time,
                        input_available=(
                            steering_input_available
                        ),
                    )
                )

            except Exception as error:

                print(
                    f"Steering analysis error: {error}"
                )

                steering_values = {

                    "steering_angle":
                        steering_angle,

                    "steering_change":
                        0.0,

                    "steering_rate":
                        0.0,

                    "steering_variability":
                        0.0,

                    "sudden_correction":
                        False,

                    "irregularity_score":
                        0.0,

                    "reliability":
                        0.0,

                    "driving_state":
                        "UNKNOWN",

                    "sample_count":
                        0,
                }

            # =================================================
            # STEERING LOG
            # =================================================

            log_steering(
                timestamp=current_time,
                steering_values=(
                    steering_values
                ),
            )

            # =================================================
            # STEP 8F - STRESS INDICATOR
            # =================================================

            try:

                stress_values = (
                    stress_indicator.calculate(
                        respiration_values=(
                            respiration_values
                        ),
                        heart_rate_values=(
                            heart_rate_values
                        ),
                        pose_values=(
                            pose_values
                        ),
                        gaze_values=(
                            gaze_values
                        ),
                        steering_values=(
                            steering_values
                        ),
                    )
                )

            except Exception as error:

                print(
                    f"Stress indicator error: {error}"
                )

                stress_values = {
                    "stress_score": 0.0,
                    "stress_level": "NO_SIGNAL",
                    "reliability": 0.0,
                    "signal_contributions": {},
                    "signal_scores": {},
                    "adaptive_weights": {},
                }

            # =================================================
            # STEP 9A - DRIVER STATE ESTIMATOR
            # =================================================

            try:

                driver_state_values = (
                    driver_state_estimator.estimate(
                        fusion_values=(
                            fusion_values
                        ),
                        gru_values=(
                            gru_values
                        ),
                        stress_values=(
                            stress_values
                        ),
                        steering_values=(
                            steering_values
                        ),
                        respiration_values=(
                            respiration_values
                        ),
                    )
                )

            except Exception as error:

                print(
                    f"Driver state estimation error: {error}"
                )

                driver_state_values = {
                    "driver_state": "UNKNOWN",
                    "state_score": 0.0,
                    "confidence": 0.0,
                    "reliability": 0.0,
                    "signal_contributions": {},
                    "signal_risks": {},
                    "state_reason": "ESTIMATOR_ERROR",
                    "risk_level": "UNKNOWN",
                    "signal_coverage": 0.0,
                }

            log_driver_state(
                timestamp=current_time,
                driver_state_values=(
                    driver_state_values
                ),
            )

            # =================================================
            # STEP 9B - ADAPTIVE SAFETY DECISION
            # =================================================

            try:

                safety_decision_values = (
                    safety_decision_engine.decide(
                        driver_state_values=(
                            driver_state_values
                        ),
                        fusion_values=(
                            fusion_values
                        ),
                        gru_values=(
                            gru_values
                        ),
                        stress_values=(
                            stress_values
                        ),
                        steering_values=(
                            steering_values
                        ),
                    )
                )

            except Exception as error:

                print(
                    f"Safety decision error: {error}"
                )

                safety_decision_values = {
                    "decision": "NO_ACTION",
                    "decision_score": 0.0,
                    "confidence": 0.0,
                    "reliability": 0.0,
                    "reason": "DECISION_ENGINE_ERROR",
                }

            # =================================================
            # STEP 9C - INTERVENTION PRIORITIZATION
            # =================================================

            try:

                intervention_priority_values = (
                    intervention_prioritizer.prioritize(
                        decision_values=(
                            safety_decision_values
                        ),
                        driver_state_values=(
                            driver_state_values
                        ),
                    )
                )

            except Exception as error:

                print(
                    f"Intervention prioritizer error: {error}"
                )

                intervention_priority_values = {
                    "priority": "NONE",
                    "priority_score": 0,
                    "action": "NO_ACTION",
                    "reason": "PRIORITIZER_ERROR",
                    "confidence": 0.0,
                    "reliability": 0.0,
                    "source_decision": "NO_ACTION",
                    "driver_state": "UNKNOWN",
                    "state_score": 0.0,
                }

            log_safety_decision(
                timestamp=current_time,
                decision_values=(
                    safety_decision_values
                ),
                priority_values=(
                    intervention_priority_values
                ),
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
            # HIGH-QUALITY ADAPTIVE DASHBOARD OVERLAY
            # =================================================
            # The previous overlay used fixed coordinates for a
            # 960x540 frame. When the window was enlarged/fullscreen,
            # text became crowded and several lines overlapped.
            #
            # This dashboard:
            #   - uses the actual frame size
            #   - uses anti-aliased text
            #   - keeps safe margins between sections
            #   - uses translucent panels for readability
            #   - is designed for 16:9 1280x720 capture
            # =================================================

            h, w = frame.shape[:2]

            # ---------- Helpers ----------

            def dashboard_panel(x1, y1, x2, y2, alpha=0.48):
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (18, 18, 18),
                    -1,
                )
                cv2.addWeighted(
                    overlay,
                    alpha,
                    frame,
                    1.0 - alpha,
                    0,
                    frame,
                )

            def dashboard_text(
                text_value,
                x,
                y,
                scale=0.62,
                color=(255, 255, 255),
                thickness=2,
            ):
                cv2.putText(
                    frame,
                    str(text_value),
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

            # Column layout adapts to the actual frame.
            left_x = max(24, int(w * 0.025))
            divider_x = int(w * 0.515)
            right_x = divider_x + 28

            # Keep panels away from the very edge.
            left_panel_right = divider_x - 20
            right_panel_right = w - 24

            # ---------- Values ----------

            steering_state = steering_values.get(
                "driving_state",
                "UNKNOWN",
            )

            steering_irregularity = safe_float(
                steering_values.get(
                    "irregularity_score",
                    0.0,
                )
            )

            steering_reliability = safe_float(
                steering_values.get(
                    "reliability",
                    0.0,
                )
            )

            steering_angle_display = safe_float(
                steering_values.get(
                    "steering_angle",
                    0.0,
                )
            )

            steering_change = safe_float(
                steering_values.get(
                    "steering_change",
                    0.0,
                )
            )

            steering_rate = safe_float(
                steering_values.get(
                    "steering_rate",
                    0.0,
                )
            )

            steering_variability = safe_float(
                steering_values.get(
                    "steering_variability",
                    0.0,
                )
            )

            heart_rate_bpm = safe_float(
                heart_rate_values.get(
                    "heart_rate_bpm",
                    0.0,
                )
            )

            heart_rate_reliability = safe_float(
                heart_rate_values.get(
                    "reliability",
                    0.0,
                )
            )

            respiration_bpm = safe_float(
                respiration_values.get(
                    "respiration_rate_bpm",
                    0.0,
                )
            )

            respiration_reliability = safe_float(
                respiration_values.get(
                    "reliability",
                    0.0,
                )
            )

            risk_level = fusion_values.get(
                "risk_level",
                "UNKNOWN",
            )

            fusion_fatigue_risk = safe_float(
                fusion_values.get(
                    "fatigue_risk",
                    0.0,
                )
            )

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

            gru_classification = gru_values.get(
                "classification",
                "UNKNOWN",
            )

            gru_sequence_length = gru_values.get(
                "sequence_length",
                0,
            )

            gru_sequence_capacity = gru_values.get(
                "sequence_capacity",
                20,
            )

            overall_reliability = safe_float(
                reliability_values.get(
                    "overall_reliability",
                    0.0,
                )
            )

            temporal_state = temporal_values.get(
                "state",
                "UNKNOWN",
            )

            stress_score = safe_float(
                stress_values.get(
                    "stress_score",
                    0.0,
                )
            )

            stress_level = stress_values.get(
                "stress_level",
                "NO_SIGNAL",
            )

            stress_reliability = safe_float(
                stress_values.get(
                    "reliability",
                    0.0,
                )
            )

            driver_state = driver_state_values.get(
                "driver_state",
                "UNKNOWN",
            )

            driver_state_score = safe_float(
                driver_state_values.get(
                    "state_score",
                    0.0,
                )
            )

            driver_state_confidence = safe_float(
                driver_state_values.get(
                    "confidence",
                    0.0,
                )
            )

            driver_state_reliability = safe_float(
                driver_state_values.get(
                    "reliability",
                    0.0,
                )
            )

            driver_state_reason = driver_state_values.get(
                "state_reason",
                "NO_DATA",
            )

            safety_decision = safety_decision_values.get(
                "decision",
                "NO_ACTION",
            )

            safety_decision_score = safe_float(
                safety_decision_values.get(
                    "decision_score",
                    0.0,
                )
            )

            safety_decision_confidence = safe_float(
                safety_decision_values.get(
                    "confidence",
                    0.0,
                )
            )

            safety_decision_reliability = safe_float(
                safety_decision_values.get(
                    "reliability",
                    0.0,
                )
            )

            safety_decision_reason = safety_decision_values.get(
                "reason",
                "NO_DATA",
            )

            intervention_priority = intervention_priority_values.get(
                "priority",
                "NONE",
            )

            intervention_priority_action = intervention_priority_values.get(
                "action",
                "NO_ACTION",
            )

            intervention_level = intervention_values.get(
                "level",
                "NORMAL",
            )

            intervention_action = intervention_values.get(
                "action",
                "NO_ACTION",
            )

            event_count = event_values.get(
                "event_count",
                0,
            )

            # ---------- Risk color ----------

            if risk_level == "NORMAL":
                risk_color = (0, 255, 0)
            elif risk_level in ("LOW", "MODERATE"):
                risk_color = (0, 255, 255)
            elif risk_level == "HIGH":
                risk_color = (0, 165, 255)
            else:
                risk_color = (0, 0, 255)

            if driver_state == "NORMAL":
                driver_state_color = (0, 255, 0)
            elif driver_state == "LOW_RISK":
                driver_state_color = (0, 255, 255)
            elif driver_state == "MODERATE_RISK":
                driver_state_color = (0, 165, 255)
            elif driver_state == "HIGH_RISK":
                driver_state_color = (0, 100, 255)
            elif driver_state == "CRITICAL":
                driver_state_color = (0, 0, 255)
            else:
                driver_state_color = (255, 255, 255)

            # ---------- Panels ----------
            #
            # Top-left: driver metrics
            # Bottom-left: pose/gaze + steering
            # Top-right: fusion/GRU
            # Bottom-right: physiological + intervention
            #
            # These panels deliberately do not cover the center
            # face area.

            top_y1 = 18
            top_y2 = int(h * 0.43)

            bottom_y1 = int(h * 0.45)
            bottom_y2 = h - 54

            dashboard_panel(
                left_x - 12,
                top_y1,
                left_panel_right,
                top_y2,
                alpha=0.42,
            )

            dashboard_panel(
                left_x - 12,
                bottom_y1,
                left_panel_right,
                bottom_y2,
                alpha=0.42,
            )

            dashboard_panel(
                right_x - 14,
                top_y1,
                right_panel_right,
                top_y2,
                alpha=0.48,
            )

            dashboard_panel(
                right_x - 14,
                bottom_y1,
                right_panel_right,
                bottom_y2,
                alpha=0.48,
            )

            # ---------- Top-left: DRIVER ----------
            y = 48
            line = 38

            dashboard_text(
                "DRIVER MONITOR",
                left_x,
                y,
                0.72,
                (255, 255, 255),
                2,
            )

            y += line

            if face_detected:
                dashboard_text(
                    f"EAR: {safe_float(driver_values['ear']):.3f}",
                    left_x,
                    y,
                    0.60,
                    (0, 255, 0),
                    2,
                )

                y += line

                dashboard_text(
                    f"MAR: {safe_float(driver_values['mar']):.3f}",
                    left_x,
                    y,
                    0.60,
                    (0, 255, 0),
                    2,
                )

                y += line

                dashboard_text(
                    f"PERCLOS: {safe_float(driver_values['perclos']):.3f}",
                    left_x,
                    y,
                    0.60,
                    (0, 255, 0),
                    2,
                )

                y += line

                dashboard_text(
                    f"Blinks: {driver_values.get('blink_count', 0)}",
                    left_x,
                    y,
                    0.60,
                    (0, 255, 0),
                    2,
                )

                y += line

                eye_status = (
                    "EYES CLOSED"
                    if driver_values.get("eyes_closed", False)
                    else "EYES OPEN"
                )

                eye_color = (
                    (0, 165, 255)
                    if driver_values.get("eyes_closed", False)
                    else (0, 255, 0)
                )

                dashboard_text(
                    eye_status,
                    left_x,
                    y,
                    0.58,
                    eye_color,
                    2,
                )

                y += line

                dashboard_text(
                    f"Eye Closed: {current_eye_closure_duration:.1f}s",
                    left_x,
                    y,
                    0.50,
                    (
                        (0, 0, 255)
                        if current_eye_closure_duration >= 1.0
                        else (255, 255, 255)
                    ),
                    2,
                )

                if driver_values.get("yawning", False):
                    y += line
                    dashboard_text(
                        "YAWNING",
                        left_x,
                        y,
                        0.58,
                        (0, 165, 255),
                        2,
                    )

                if driver_values.get("microsleep", False):
                    y += line
                    dashboard_text(
                        f"MICROSLEEP {safe_float(driver_values.get('microsleep_duration', 0.0)):.1f}s",
                        left_x,
                        y,
                        0.54,
                        (0, 0, 255),
                        2,
                    )

            else:
                dashboard_text(
                    "NO FACE DETECTED",
                    left_x,
                    y,
                    0.62,
                    (0, 0, 255),
                    2,
                )

            # ---------- Bottom-left: POSE / GAZE ----------
            y = bottom_y1 + 34

            dashboard_text(
                "DRIVER ORIENTATION",
                left_x,
                y,
                0.62,
                (255, 255, 255),
                2,
            )

            y += 34

            if pose_values.get("valid", False):
                dashboard_text(
                    f"Pitch: {safe_float(pose_values.get('pitch')):.1f}",
                    left_x,
                    y,
                    0.52,
                    (255, 255, 0),
                    2,
                )

                y += 30

                dashboard_text(
                    f"Yaw: {safe_float(pose_values.get('yaw')):.1f}",
                    left_x,
                    y,
                    0.52,
                    (255, 255, 0),
                    2,
                )

                y += 30

                dashboard_text(
                    f"Roll: {safe_float(pose_values.get('roll')):.1f}",
                    left_x,
                    y,
                    0.52,
                    (255, 255, 0),
                    2,
                )

            y += 30

            dashboard_text(
                f"Gaze: {gaze_values.get('gaze_direction', 'UNKNOWN')}",
                left_x,
                y,
                0.56,
                (255, 0, 255),
                2,
            )

            y += 32

            if gaze_values.get("prolonged_gaze_away", False):
                dashboard_text(
                    f"GAZE AWAY {safe_float(gaze_values.get('gaze_away_duration', 0.0)):.1f}s",
                    left_x,
                    y,
                    0.50,
                    (0, 0, 255),
                    2,
                )

            # ---------- Bottom-left: STEERING ----------
            steering_x = int(w * 0.27)

            dashboard_text(
                "STEERING",
                steering_x,
                bottom_y1 + 34,
                0.62,
                (255, 255, 255),
                2,
            )

            dashboard_text(
                f"Angle: {steering_angle_display:.1f}",
                steering_x,
                bottom_y1 + 66,
                0.46,
                (255, 255, 255),
                1,
            )

            dashboard_text(
                f"Change: {steering_change:.1f}",
                steering_x,
                bottom_y1 + 94,
                0.46,
                (255, 255, 255),
                1,
            )

            dashboard_text(
                f"Rate: {steering_rate:.1f}",
                steering_x,
                bottom_y1 + 122,
                0.46,
                (255, 255, 255),
                1,
            )

            steering_color = (
                (0, 255, 0)
                if steering_irregularity < 0.40
                else (
                    (0, 165, 255)
                    if steering_irregularity < 0.75
                    else (0, 0, 255)
                )
            )

            dashboard_text(
                f"Irregularity: {steering_irregularity:.2f}",
                steering_x,
                bottom_y1 + 150,
                0.44,
                steering_color,
                1,
            )

            dashboard_text(
                f"State: {steering_state}",
                steering_x,
                bottom_y1 + 178,
                0.44,
                (255, 255, 255),
                1,
            )

            dashboard_text(
                f"Reliability: {steering_reliability:.2f}",
                steering_x,
                bottom_y1 + 206,
                0.44,
                (255, 255, 255),
                1,
            )

            # ---------- Top-right: SYSTEM / FUSION ----------
            y = 48

            dashboard_text(
                "ADAPTIVE DMS",
                right_x,
                y,
                0.72,
                (255, 255, 255),
                2,
            )

            y += 38

            dashboard_text(
                f"Fusion Risk: {fusion_fatigue_risk:.2f}",
                right_x,
                y,
                0.60,
                (0, 255, 255),
                2,
            )

            y += 34

            dashboard_text(
                f"Risk Level: {risk_level}",
                right_x,
                y,
                0.58,
                risk_color,
                2,
            )

            y += 34

            dashboard_text(
                f"GRU Risk: {gru_prediction:.2f}",
                right_x,
                y,
                0.58,
                (255, 0, 255),
                2,
            )

            y += 32

            dashboard_text(
                f"GRU Class: {gru_classification}",
                right_x,
                y,
                0.52,
                (255, 0, 255),
                2,
            )

            y += 30

            dashboard_text(
                f"GRU Confidence: {gru_confidence:.2f}",
                right_x,
                y,
                0.47,
                (255, 255, 255),
                1,
            )

            y += 28

            dashboard_text(
                f"GRU Sequence: {gru_sequence_length}/{gru_sequence_capacity}",
                right_x,
                y,
                0.47,
                (255, 255, 255),
                1,
            )

            y += 28

            dashboard_text(
                f"Reliability: {overall_reliability:.2f}",
                right_x,
                y,
                0.47,
                (255, 255, 255),
                1,
            )

            y += 28

            dashboard_text(
                f"Temporal: {temporal_state}",
                right_x,
                y,
                0.48,
                (255, 255, 255),
                1,
            )

            y += 30

            dashboard_text(
                f"Driver State: {driver_state}",
                right_x,
                y,
                0.54,
                driver_state_color,
                2,
            )

            y += 28

            dashboard_text(
                f"State Score: {driver_state_score:.2f} | Confidence: {driver_state_confidence:.2f}",
                right_x,
                y,
                0.40,
                (255, 255, 255),
                1,
            )

            y += 28

            dashboard_text(
                f"Stress: {stress_score:.2f} ({stress_level}) | Rel: {stress_reliability:.2f}",
                right_x,
                y,
                0.40,
                (255, 0, 255),
                1,
            )

            # ---------- Bottom-right: SAFETY DECISION ----------
            y = bottom_y1 + 34

            dashboard_text(
                "SAFETY DECISION ENGINE",
                right_x,
                y,
                0.58,
                (255, 255, 255),
                2,
            )

            y += 30

            if safety_decision == "NO_ACTION":
                decision_color = (0, 255, 0)
            elif safety_decision == "ADVISORY":
                decision_color = (0, 255, 255)
            elif safety_decision == "WARNING":
                decision_color = (0, 165, 255)
            elif safety_decision == "URGENT_WARNING":
                decision_color = (0, 100, 255)
            else:
                decision_color = (0, 0, 255)

            dashboard_text(
                f"Decision: {safety_decision}",
                right_x,
                y,
                0.48,
                decision_color,
                2,
            )

            y += 28

            dashboard_text(
                f"Score: {safety_decision_score:.2f} | Conf: {safety_decision_confidence:.2f}",
                right_x,
                y,
                0.40,
                (255, 255, 255),
                1,
            )

            y += 26

            dashboard_text(
                f"Rel: {safety_decision_reliability:.2f} | Priority: {intervention_priority}",
                right_x,
                y,
                0.40,
                (255, 255, 255),
                1,
            )

            y += 26

            dashboard_text(
                f"Action: {intervention_priority_action}",
                right_x,
                y,
                0.40,
                (255, 255, 255),
                1,
            )

            # ---------- Bottom-right: PHYSIOLOGY / SAFETY ----------
            y += 34

            dashboard_text(
                "PHYSIOLOGICAL SIGNALS",
                right_x,
                y,
                0.62,
                (255, 255, 255),
                2,
            )

            y += 34

            dashboard_text(
                f"Heart Rate: {heart_rate_bpm:.1f} BPM",
                right_x,
                y,
                0.52,
                (0, 255, 255),
                2,
            )

            dashboard_text(
                f"HR Rel: {heart_rate_reliability:.2f}",
                right_x + 265,
                y,
                0.42,
                (255, 255, 255),
                1,
            )

            y += 30

            dashboard_text(
                f"Respiration: {respiration_bpm:.1f} /min",
                right_x,
                y,
                0.52,
                (0, 255, 255),
                2,
            )

            dashboard_text(
                f"Resp Rel: {respiration_reliability:.2f}",
                right_x + 265,
                y,
                0.42,
                (255, 255, 255),
                1,
            )

            y += 36

            dashboard_text(
                "SAFETY INTERVENTION",
                right_x,
                y,
                0.58,
                (255, 255, 255),
                2,
            )

            y += 32

            intervention_color = risk_color

            dashboard_text(
                f"Level: {intervention_level}",
                right_x,
                y,
                0.52,
                intervention_color,
                2,
            )

            y += 30

            dashboard_text(
                f"Action: {intervention_action}",
                right_x,
                y,
                0.44,
                (255, 255, 255),
                1,
            )

            y += 28

            dashboard_text(
                f"Alerts: {intervention_values.get('alert_count', 0)}",
                right_x,
                y,
                0.44,
                (255, 255, 255),
                1,
            )

            y += 28

            dashboard_text(
                f"Events: {event_count}",
                right_x,
                y,
                0.44,
                (255, 255, 255),
                1,
            )

            # ---------- Warning overlays ----------
            if current_eye_closure_duration >= 1.0:
                dashboard_text(
                    "EYE CLOSURE WARNING",
                    right_x,
                    h - 102,
                    0.50,
                    (0, 0, 255),
                    2,
                )

            if intervention_level in ("HIGH", "CRITICAL"):
                cv2.rectangle(
                    frame,
                    (8, 8),
                    (w - 8, h - 8),
                    (0, 0, 255),
                    4,
                )

                dashboard_text(
                    intervention_values.get(
                        "message",
                        "HIGH RISK",
                    ),
                    right_x,
                    h - 72,
                    0.48,
                    (0, 0, 255),
                    2,
                )

            elif intervention_level == "MODERATE":
                dashboard_text(
                    intervention_values.get(
                        "message",
                        "MODERATE RISK",
                    ),
                    right_x,
                    h - 72,
                    0.44,
                    (0, 165, 255),
                    2,
                )

            # ---------- Bottom status bar ----------
            status_y = h - 18

            dashboard_text(
                (
                    "FACE DETECTED"
                    if face_detected
                    else "FACE NOT DETECTED"
                ),
                left_x,
                status_y,
                0.46,
                (
                    (0, 255, 0)
                    if face_detected
                    else (0, 0, 255)
                ),
                2,
            )

            dashboard_text(
                "A: LEFT | D: RIGHT | S: CENTER | Q: QUIT",
                int(w * 0.28),
                status_y,
                0.42,
                (255, 255, 255),
                1,
            )

            dashboard_text(
                f"FPS: {fps:.1f}",
                w - 120,
                38,
                0.55,
                (255, 255, 255),
                2,
            )

            dashboard_text(
                "ADAPTIVE-DMS | STEP 9C",
                int(w * 0.40),
                status_y,
                0.38,
                (255, 255, 255),
                1,
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
                WINDOW_NAME,
                frame,
            )

            # =================================================
            # KEYBOARD INPUT
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # -------------------------------------------------
            # CAMERA WINDOW: FULLSCREEN
            # -------------------------------------------------

            if key == ord("f"):

                fullscreen = not fullscreen

                set_camera_window_fullscreen(
                    fullscreen
                )

            # -------------------------------------------------
            # CAMERA WINDOW: WINDOWED
            # -------------------------------------------------

            elif key == 27:

                fullscreen = False

                set_camera_window_fullscreen(
                    False
                )

            # -------------------------------------------------
            # CAMERA WINDOW: MAXIMIZE
            # -------------------------------------------------

            elif key == ord("m"):

                fullscreen = False

                try:
                    import ctypes

                    hwnd = ctypes.windll.user32.FindWindowW(
                        None,
                        WINDOW_NAME,
                    )

                    if hwnd:
                        ctypes.windll.user32.ShowWindow(
                            hwnd,
                            3,
                        )

                except Exception:
                    pass

            # -------------------------------------------------
            # CAMERA WINDOW: RESET SIZE
            # -------------------------------------------------

            elif key == ord("r"):

                fullscreen = False

                set_camera_window_fullscreen(
                    False
                )

            # -------------------------------------------------
            # LEFT
            # -------------------------------------------------

            elif key == ord("a"):

                steering_angle -= (
                    steering_step
                )

                steering_angle = max(
                    -450.0,
                    steering_angle,
                )

            # -------------------------------------------------
            # RIGHT
            # -------------------------------------------------

            elif key == ord("d"):

                steering_angle += (
                    steering_step
                )

                steering_angle = min(
                    450.0,
                    steering_angle,
                )

            # -------------------------------------------------
            # CENTER
            # -------------------------------------------------

            elif key == ord("s"):

                steering_angle = 0.0

            # -------------------------------------------------
            # QUIT
            # -------------------------------------------------

            elif key == ord("q"):

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
        # SESSION LOGGER
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
        # EVENT TRACKER
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
        # DETECTOR
        # =====================================================

        try:

            detector.close()

        except Exception as error:

            print(
                f"Detector close error: "
                f"{error}"
            )

        # =====================================================
        # CAMERA
        # =====================================================

        try:

            camera.release()

        except Exception as error:

            print(
                f"Camera release error: "
                f"{error}"
            )

        # =====================================================
        # OPENCV WINDOWS
        # =====================================================

        cv2.destroyAllWindows()

        print()

        print("=" * 70)

        print(
            "Camera released."
        )

        print(
            "ADAPTIVE-DMS stopped."
        )

        print(
            "Steering analysis session completed."
        )

        print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()