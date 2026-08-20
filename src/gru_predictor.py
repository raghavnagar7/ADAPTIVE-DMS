"""
=============================================================
ADAPTIVE-DMS
=============================================================

GRU Temporal Fatigue Predictor
Version:
    v1.8 - Step 1

Purpose:
    Temporal prediction of driver fatigue using a GRU model.

Input features:
    - Fatigue risk
    - EAR
    - MAR
    - PERCLOS
    - Head pose
    - Gaze
    - Signal reliability

Features:
    - Fixed-length temporal sequence
    - Feature normalization
    - GRU model
    - Prediction confidence
    - Safe fallback before a trained model exists
    - Model save/load support
    - No modification of existing v1.7 modules

IMPORTANT:
    This module is NOT connected to main.py yet.

    Step 1 only creates and tests the GRU predictor.

=============================================================
"""

import os
from collections import deque

import numpy as np


# =============================================================
# OPTIONAL TENSORFLOW IMPORT
# =============================================================

try:

    import tensorflow as tf

    from tensorflow.keras import (
        Sequential,
    )

    from tensorflow.keras.layers import (
        GRU,
        Dense,
        Dropout,
        Input,
    )

    from tensorflow.keras.models import (
        load_model,
    )

    TENSORFLOW_AVAILABLE = True

except ImportError:

    tf = None

    Sequential = None
    GRU = None
    Dense = None
    Dropout = None
    Input = None
    load_model = None

    TENSORFLOW_AVAILABLE = False


# =============================================================
# CLASS
# =============================================================

class GRUFatiguePredictor:
    """
    GRU-based temporal fatigue predictor.

    The predictor receives one feature vector at a time and
    maintains a rolling temporal sequence.

    Once enough samples are available, the sequence is passed
    through the GRU model.

    Before a trained model is available, a safe heuristic
    fallback prediction is returned.

    This allows the module to be integrated into the existing
    ADAPTIVE-DMS pipeline without breaking the system.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        sequence_length=20,
        feature_names=None,
        model_path="models/gru_fatigue_model.keras",
        prediction_threshold=0.50,
        minimum_sequence_ratio=0.50,
    ):
        """
        Parameters
        ----------
        sequence_length : int
            Number of historical samples used by the GRU.

        feature_names : list[str]
            Names of input features.

        model_path : str
            Location of the saved GRU model.

        prediction_threshold : float
            Threshold used to classify predicted fatigue.

        minimum_sequence_ratio : float
            Minimum percentage of sequence required before
            prediction begins.
        """

        # -----------------------------------------------------
        # Validate sequence length
        # -----------------------------------------------------

        sequence_length = int(
            sequence_length
        )

        if sequence_length < 2:

            raise ValueError(
                "sequence_length must be at least 2."
            )

        self.sequence_length = (
            sequence_length
        )

        # -----------------------------------------------------
        # Default features
        # -----------------------------------------------------

        if feature_names is None:

            feature_names = [

                "fatigue_risk",

                "ear",

                "mar",

                "perclos",

                "pitch",

                "yaw",

                "roll",

                "horizontal_ratio",

                "vertical_ratio",

                "gaze_away_duration",

                "reliability",

            ]

        self.feature_names = list(
            feature_names
        )

        self.feature_count = len(
            self.feature_names
        )

        # -----------------------------------------------------
        # Model configuration
        # -----------------------------------------------------

        self.model_path = (
            model_path
        )

        self.prediction_threshold = float(
            prediction_threshold
        )

        self.minimum_sequence_ratio = float(
            minimum_sequence_ratio
        )

        # -----------------------------------------------------
        # Sequence buffer
        # -----------------------------------------------------

        self.sequence = deque(
            maxlen=self.sequence_length
        )

        # -----------------------------------------------------
        # Normalization statistics
        #
        # These defaults are intentionally conservative.
        # They can later be replaced with statistics generated
        # from the training dataset.
        # -----------------------------------------------------

        self.feature_min = (
            np.zeros(
                self.feature_count,
                dtype=np.float32,
            )
        )

        self.feature_max = (
            np.ones(
                self.feature_count,
                dtype=np.float32,
            )
        )

        self._configure_default_normalization()

        # -----------------------------------------------------
        # Model
        # -----------------------------------------------------

        self.model = None

        self.model_loaded = False

        self.model_trained = False

        # -----------------------------------------------------
        # Prediction state
        # -----------------------------------------------------

        self.last_prediction = 0.0

        self.last_confidence = 0.0

        self.last_classification = (
            "UNKNOWN"
        )

        self.prediction_count = 0

        # -----------------------------------------------------
        # Try loading an existing model
        # -----------------------------------------------------

        self.load_model(
            self.model_path
        )

    # =========================================================
    # DEFAULT NORMALIZATION
    # =========================================================

    def _configure_default_normalization(
        self,
    ):
        """
        Configure reasonable feature ranges.

        These are used only when a training normalization file
        is not supplied.

        The values are intentionally kept simple and can later
        be replaced with training-data statistics.
        """

        ranges = {

            "fatigue_risk": (
                0.0,
                1.0,
            ),

            "ear": (
                0.0,
                0.60,
            ),

            "mar": (
                0.0,
                1.50,
            ),

            "perclos": (
                0.0,
                1.0,
            ),

            "pitch": (
                -90.0,
                90.0,
            ),

            "yaw": (
                -90.0,
                90.0,
            ),

            "roll": (
                -90.0,
                90.0,
            ),

            "horizontal_ratio": (
                0.0,
                1.0,
            ),

            "vertical_ratio": (
                0.0,
                1.0,
            ),

            "gaze_away_duration": (
                0.0,
                10.0,
            ),

            "reliability": (
                0.0,
                1.0,
            ),
        }

        for index, name in enumerate(
            self.feature_names
        ):

            minimum, maximum = ranges.get(
                name,
                (
                    0.0,
                    1.0,
                ),
            )

            self.feature_min[
                index
            ] = minimum

            self.feature_max[
                index
            ] = maximum

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):
        """
        Safely convert value to float.
        """

        try:

            if value is None:

                return default

            result = float(
                value
            )

            if not np.isfinite(
                result
            ):

                return default

            return result

        except (
            TypeError,
            ValueError,
        ):

            return default

    # =========================================================
    # FEATURE EXTRACTION
    # =========================================================

    def extract_features(
        self,
        driver_values=None,
        pose_values=None,
        gaze_values=None,
        reliability_values=None,
        fatigue_risk=0.0,
    ):
        """
        Convert existing ADAPTIVE-DMS outputs into a feature
        vector compatible with the GRU model.

        Returns
        -------
        numpy.ndarray
            Shape:

                (feature_count,)
        """

        driver_values = (
            driver_values
            or {}
        )

        pose_values = (
            pose_values
            or {}
        )

        gaze_values = (
            gaze_values
            or {}
        )

        reliability_values = (
            reliability_values
            or {}
        )

        features = []

        for name in self.feature_names:

            # -------------------------------------------------
            # Fatigue risk
            # -------------------------------------------------

            if name == "fatigue_risk":

                value = (
                    fatigue_risk
                )

            # -------------------------------------------------
            # Driver metrics
            # -------------------------------------------------

            elif name == "ear":

                value = driver_values.get(
                    "ear",
                    0.0,
                )

            elif name == "mar":

                value = driver_values.get(
                    "mar",
                    0.0,
                )

            elif name == "perclos":

                value = driver_values.get(
                    "perclos",
                    0.0,
                )

            # -------------------------------------------------
            # Head pose
            # -------------------------------------------------

            elif name == "pitch":

                value = pose_values.get(
                    "pitch",
                    0.0,
                )

            elif name == "yaw":

                value = pose_values.get(
                    "yaw",
                    0.0,
                )

            elif name == "roll":

                value = pose_values.get(
                    "roll",
                    0.0,
                )

            # -------------------------------------------------
            # Gaze
            # -------------------------------------------------

            elif name == "horizontal_ratio":

                value = gaze_values.get(
                    "horizontal_ratio",
                    0.5,
                )

            elif name == "vertical_ratio":

                value = gaze_values.get(
                    "vertical_ratio",
                    0.5,
                )

            elif name == "gaze_away_duration":

                value = gaze_values.get(
                    "gaze_away_duration",
                    0.0,
                )

            # -------------------------------------------------
            # Reliability
            # -------------------------------------------------

            elif name == "reliability":

                value = reliability_values.get(
                    "overall_reliability",
                    reliability_values.get(
                        "reliability",
                        0.0,
                    ),
                )

            # -------------------------------------------------
            # Unknown feature
            # -------------------------------------------------

            else:

                value = 0.0

            features.append(
                self._safe_float(
                    value
                )
            )

        return np.asarray(
            features,
            dtype=np.float32,
        )

    # =========================================================
    # NORMALIZATION
    # =========================================================

    def normalize_features(
        self,
        features,
    ):
        """
        Min-max normalize features to [0, 1].
        """

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        if features.shape != (
            self.feature_count,
        ):

            raise ValueError(
                "Feature vector has incorrect shape. "
                f"Expected ({self.feature_count},), "
                f"received {features.shape}."
            )

        denominator = (
            self.feature_max
            - self.feature_min
        )

        denominator = np.where(
            denominator == 0,
            1.0,
            denominator,
        )

        normalized = (
            features
            - self.feature_min
        ) / denominator

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        return normalized.astype(
            np.float32
        )

    # =========================================================
    # ADD SAMPLE
    # =========================================================

    def add_sample(
        self,
        features,
    ):
        """
        Add one normalized feature vector to the temporal
        sequence.
        """

        normalized = (
            self.normalize_features(
                features
            )
        )

        self.sequence.append(
            normalized
        )

        return len(
            self.sequence
        )

    # =========================================================
    # SEQUENCE READY
    # =========================================================

    def is_sequence_ready(
        self,
    ):
        """
        Determine whether enough samples are available.
        """

        minimum_samples = max(
            2,
            int(
                np.ceil(
                    self.sequence_length
                    * self.minimum_sequence_ratio
                )
            ),
        )

        return (
            len(self.sequence)
            >= minimum_samples
        )

    # =========================================================
    # SEQUENCE ARRAY
    # =========================================================

    def get_sequence(
        self,
        pad=True,
    ):
        """
        Return current temporal sequence.

        Returns
        -------
        numpy.ndarray

        Shape:

            (1, sequence_length, feature_count)

        if pad=True.

        Otherwise:

            (1, current_length, feature_count)
        """

        if len(
            self.sequence
        ) == 0:

            sequence = np.zeros(
                (
                    1,
                    self.sequence_length,
                    self.feature_count,
                ),
                dtype=np.float32,
            )

            return sequence

        values = np.asarray(
            list(
                self.sequence
            ),
            dtype=np.float32,
        )

        if pad:

            missing = (
                self.sequence_length
                - values.shape[0]
            )

            if missing > 0:

                first = values[
                    0
                ]

                padding = np.repeat(
                    first[
                        np.newaxis,
                        :,
                    ],
                    missing,
                    axis=0,
                )

                values = np.concatenate(
                    [
                        padding,
                        values,
                    ],
                    axis=0,
                )

        return values[
            np.newaxis,
            :,
            :,
        ]

    # =========================================================
    # BUILD MODEL
    # =========================================================

    def build_model(
        self,
    ):
        """
        Build the GRU neural network.

        Architecture:

            Input
              ↓
            GRU(64)
              ↓
            Dropout
              ↓
            GRU(32)
              ↓
            Dropout
              ↓
            Dense(16)
              ↓
            Dense(1, sigmoid)

        The output represents predicted fatigue risk
        between 0 and 1.
        """

        if not TENSORFLOW_AVAILABLE:

            raise RuntimeError(
                "TensorFlow is not installed. "
                "Install TensorFlow before building "
                "the GRU model."
            )

        model = Sequential(
            [

                Input(
                    shape=(
                        self.sequence_length,
                        self.feature_count,
                    )
                ),

                GRU(
                    64,
                    return_sequences=True,
                ),

                Dropout(
                    0.20
                ),

                GRU(
                    32,
                    return_sequences=False,
                ),

                Dropout(
                    0.20
                ),

                Dense(
                    16,
                    activation="relu",
                ),

                Dense(
                    1,
                    activation="sigmoid",
                ),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
            ],
        )

        self.model = model

        return model

    # =========================================================
    # LOAD MODEL
    # =========================================================

    def load_model(
        self,
        model_path=None,
    ):
        """
        Load an existing trained GRU model.

        Returns
        -------
        bool
            True if successfully loaded.
        """

        if model_path is not None:

            self.model_path = (
                model_path
            )

        if not TENSORFLOW_AVAILABLE:

            self.model_loaded = False

            return False

        if not self.model_path:

            return False

        if not os.path.exists(
            self.model_path
        ):

            self.model_loaded = False

            return False

        try:

            self.model = load_model(
                self.model_path
            )

            self.model_loaded = True

            self.model_trained = True

            return True

        except Exception:

            self.model = None

            self.model_loaded = False

            self.model_trained = False

            return False

    # =========================================================
    # SAVE MODEL
    # =========================================================

    def save_model(
        self,
        model_path=None,
    ):
        """
        Save the current GRU model.
        """

        if self.model is None:

            raise RuntimeError(
                "No GRU model exists."
            )

        if model_path is not None:

            self.model_path = (
                model_path
            )

        if not self.model_path:

            raise ValueError(
                "No model path specified."
            )

        directory = os.path.dirname(
            self.model_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        self.model.save(
            self.model_path
        )

        self.model_loaded = True

        return self.model_path

    # =========================================================
    # HEURISTIC FALLBACK
    # =========================================================

    def _heuristic_prediction(
        self,
        sequence,
    ):
        """
        Fallback prediction used when a trained GRU model
        is not available.

        This is NOT the final machine-learning prediction.

        It provides a stable temporary output so the system
        can be integrated and tested before a trained model
        is available.
        """

        if sequence.size == 0:

            return 0.0

        latest = sequence[
            -1
        ]

        # -----------------------------------------------------
        # Locate features
        # -----------------------------------------------------

        def index_of(
            name
        ):

            try:

                return self.feature_names.index(
                    name
                )

            except ValueError:

                return None

        risk_index = index_of(
            "fatigue_risk"
        )

        ear_index = index_of(
            "ear"
        )

        mar_index = index_of(
            "mar"
        )

        perclos_index = index_of(
            "perclos"
        )

        reliability_index = index_of(
            "reliability"
        )

        gaze_index = index_of(
            "gaze_away_duration"
        )

        # -----------------------------------------------------
        # Start with current fusion risk
        # -----------------------------------------------------

        score = 0.0

        weight = 0.0

        if risk_index is not None:

            score += (
                latest[
                    risk_index
                ]
                * 0.55
            )

            weight += 0.55

        # -----------------------------------------------------
        # PERCLOS
        # -----------------------------------------------------

        if perclos_index is not None:

            score += (
                latest[
                    perclos_index
                ]
                * 0.20
            )

            weight += 0.20

        # -----------------------------------------------------
        # EAR
        #
        # Lower EAR = greater fatigue indication.
        # -----------------------------------------------------

        if ear_index is not None:

            ear = latest[
                ear_index
            ]

            ear_risk = 1.0 - np.clip(
                ear,
                0.0,
                1.0,
            )

            score += (
                ear_risk
                * 0.10
            )

            weight += 0.10

        # -----------------------------------------------------
        # MAR
        #
        # Higher MAR can indicate yawning.
        # -----------------------------------------------------

        if mar_index is not None:

            mar = latest[
                mar_index
            ]

            mar_risk = np.clip(
                mar,
                0.0,
                1.0,
            )

            score += (
                mar_risk
                * 0.05
            )

            weight += 0.05

        # -----------------------------------------------------
        # Gaze away
        # -----------------------------------------------------

        if gaze_index is not None:

            gaze_away = latest[
                gaze_index
            ]

            gaze_risk = np.clip(
                gaze_away / 10.0,
                0.0,
                1.0,
            )

            score += (
                gaze_risk
                * 0.05
            )

            weight += 0.05

        # -----------------------------------------------------
        # Reliability adjustment
        # -----------------------------------------------------

        if reliability_index is not None:

            reliability = np.clip(
                latest[
                    reliability_index
                ],
                0.0,
                1.0,
            )

            # Low reliability should reduce confidence,
            # not artificially increase fatigue.
            score *= (
                0.60
                + 0.40
                * reliability
            )

        if weight <= 0:

            return 0.0

        prediction = (
            score
            / weight
        )

        return float(
            np.clip(
                prediction,
                0.0,
                1.0,
            )
        )

    # =========================================================
    # MODEL PREDICTION
    # =========================================================

    def predict_sequence(
        self,
        sequence=None,
    ):
        """
        Predict fatigue from a sequence.

        Returns
        -------
        tuple
            prediction,
            confidence
        """

        if sequence is None:

            sequence = self.get_sequence(
                pad=True
            )

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        if sequence.ndim != 3:

            raise ValueError(
                "GRU sequence must have "
                "shape (batch, time, features)."
            )

        # -----------------------------------------------------
        # GRU model
        # -----------------------------------------------------

        if (
            self.model is not None
            and self.model_loaded
        ):

            try:

                output = (
                    self.model.predict(
                        sequence,
                        verbose=0,
                    )
                )

                prediction = float(
                    np.asarray(
                        output
                    ).reshape(
                        -1
                    )[0]
                )

                prediction = float(
                    np.clip(
                        prediction,
                        0.0,
                        1.0,
                    )
                )

                # Confidence increases as prediction moves
                # away from the middle.
                confidence = float(
                    min(
                        1.0,
                        abs(
                            prediction
                            - 0.5
                        )
                        * 2.0
                        + 0.50,
                    )
                )

                return (
                    prediction,
                    confidence,
                )

            except Exception:

                # Fall back safely.

                pass

        # -----------------------------------------------------
        # Temporary fallback
        # -----------------------------------------------------

        prediction = (
            self._heuristic_prediction(
                sequence[0]
            )
        )

        # Since this is not a trained GRU prediction,
        # confidence is intentionally lower.
        confidence = 0.40

        return (
            prediction,
            confidence,
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        driver_values=None,
        pose_values=None,
        gaze_values=None,
        reliability_values=None,
        fatigue_risk=0.0,
    ):
        """
        Main runtime method.

        Called once per frame/sample.

        Returns a dictionary containing:

            prediction
            confidence
            classification
            sequence_length
            sequence_ready
            model_available
            model_trained
        """

        # -----------------------------------------------------
        # Extract features
        # -----------------------------------------------------

        features = self.extract_features(
            driver_values=driver_values,
            pose_values=pose_values,
            gaze_values=gaze_values,
            reliability_values=reliability_values,
            fatigue_risk=fatigue_risk,
        )

        # -----------------------------------------------------
        # Add to sequence
        # -----------------------------------------------------

        current_length = (
            self.add_sample(
                features
            )
        )

        # -----------------------------------------------------
        # Prediction
        # -----------------------------------------------------

        if self.is_sequence_ready():

            sequence = (
                self.get_sequence(
                    pad=True
                )
            )

            prediction, confidence = (
                self.predict_sequence(
                    sequence
                )
            )

        else:

            # Before enough temporal history exists,
            # use current fusion risk as a temporary estimate.
            prediction = float(
                np.clip(
                    self._safe_float(
                        fatigue_risk
                    ),
                    0.0,
                    1.0,
                )
            )

            confidence = (
                current_length
                / max(
                    1,
                    int(
                        np.ceil(
                            self.sequence_length
                            * self.minimum_sequence_ratio
                        )
                    ),
                )
            )

            confidence = float(
                np.clip(
                    confidence,
                    0.0,
                    0.50,
                )
            )

        # -----------------------------------------------------
        # Classification
        # -----------------------------------------------------

        if prediction >= 0.75:

            classification = (
                "CRITICAL"
            )

        elif prediction >= 0.55:

            classification = (
                "HIGH"
            )

        elif prediction >= 0.35:

            classification = (
                "MODERATE"
            )

        elif prediction >= 0.20:

            classification = (
                "LOW"
            )

        else:

            classification = (
                "NORMAL"
            )

        # -----------------------------------------------------
        # Save state
        # -----------------------------------------------------

        self.last_prediction = (
            prediction
        )

        self.last_confidence = (
            confidence
        )

        self.last_classification = (
            classification
        )

        self.prediction_count += 1

        # -----------------------------------------------------
        # Return
        # -----------------------------------------------------

        return {

            "prediction": (
                float(
                    prediction
                )
            ),

            "fatigue_risk": (
                float(
                    prediction
                )
            ),

            "confidence": (
                float(
                    confidence
                )
            ),

            "classification": (
                classification
            ),

            "sequence_length": (
                current_length
            ),

            "sequence_capacity": (
                self.sequence_length
            ),

            "sequence_ready": (
                self.is_sequence_ready()
            ),

            "model_available": (
                self.model is not None
            ),

            "model_loaded": (
                self.model_loaded
            ),

            "model_trained": (
                self.model_trained
            ),

            "prediction_count": (
                self.prediction_count
            ),
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
    ):
        """
        Clear temporal history.
        """

        self.sequence.clear()

        self.last_prediction = 0.0

        self.last_confidence = 0.0

        self.last_classification = (
            "UNKNOWN"
        )

        self.prediction_count = 0

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(
        self,
    ):
        """
        Return current predictor status.
        """

        return {

            "tensorflow_available": (
                TENSORFLOW_AVAILABLE
            ),

            "model_available": (
                self.model is not None
            ),

            "model_loaded": (
                self.model_loaded
            ),

            "model_trained": (
                self.model_trained
            ),

            "sequence_length": (
                self.sequence_length
            ),

            "current_sequence_length": (
                len(
                    self.sequence
                )
            ),

            "feature_count": (
                self.feature_count
            ),

            "feature_names": (
                self.feature_names
            ),

            "last_prediction": (
                self.last_prediction
            ),

            "last_confidence": (
                self.last_confidence
            ),

            "last_classification": (
                self.last_classification
            ),

            "prediction_count": (
                self.prediction_count
            ),
        }


# =============================================================
# SIMPLE SELF TEST
# =============================================================

def self_test():
    """
    Run a basic standalone test.

    This test does NOT require main.py.

    It verifies:

        - Module import
        - Predictor creation
        - Feature extraction
        - Normalization
        - Sequence buffering
        - Prediction output
    """

    print(
        "=" * 65
    )

    print(
        "ADAPTIVE-DMS"
    )

    print(
        "GRU FATIGUE PREDICTOR"
    )

    print(
        "v1.8 - STEP 1 SELF TEST"
    )

    print(
        "=" * 65
    )

    # ---------------------------------------------------------
    # Create predictor
    # ---------------------------------------------------------

    predictor = (
        GRUFatiguePredictor(
            sequence_length=20,
            model_path=(
                "models/"
                "gru_fatigue_model.keras"
            ),
        )
    )

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    status = (
        predictor.get_status()
    )

    print(
        f"TensorFlow available: "
        f"{status['tensorflow_available']}"
    )

    print(
        f"Model loaded: "
        f"{status['model_loaded']}"
    )

    print(
        f"Sequence length: "
        f"{status['sequence_length']}"
    )

    print(
        f"Feature count: "
        f"{status['feature_count']}"
    )

    print()

    # ---------------------------------------------------------
    # Simulated driver values
    # ---------------------------------------------------------

    driver_values = {

        "ear": 0.28,

        "mar": 0.20,

        "perclos": 0.18,

    }

    pose_values = {

        "pitch": 4.0,

        "yaw": 3.0,

        "roll": 1.0,

    }

    gaze_values = {

        "horizontal_ratio": 0.50,

        "vertical_ratio": 0.50,

        "gaze_away_duration": 0.2,

    }

    reliability_values = {

        "overall_reliability": 0.90,

    }

    # ---------------------------------------------------------
    # Feed temporal samples
    # ---------------------------------------------------------

    for index in range(
        20
    ):

        # Slowly increase fatigue risk
        # to simulate a deteriorating driver state.

        simulated_risk = min(
            0.80,
            0.20
            + (
                index
                * 0.025
            ),
        )

        result = predictor.update(
            driver_values=driver_values,
            pose_values=pose_values,
            gaze_values=gaze_values,
            reliability_values=(
                reliability_values
            ),
            fatigue_risk=simulated_risk,
        )

    # ---------------------------------------------------------
    # Print result
    # ---------------------------------------------------------

    print(
        "Prediction:"
    )

    print(
        f"  Fatigue Risk: "
        f"{result['prediction']:.3f}"
    )

    print(
        f"  Confidence: "
        f"{result['confidence']:.3f}"
    )

    print(
        f"  Classification: "
        f"{result['classification']}"
    )

    print(
        f"  Sequence: "
        f"{result['sequence_length']}/"
        f"{result['sequence_capacity']}"
    )

    print(
        f"  Sequence Ready: "
        f"{result['sequence_ready']}"
    )

    print(
        f"  Model Available: "
        f"{result['model_available']}"
    )

    print(
        f"  Model Trained: "
        f"{result['model_trained']}"
    )

    print()

    print(
        "SELF TEST COMPLETE"
    )

    print(
        "=" * 65
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    self_test()