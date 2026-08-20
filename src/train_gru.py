"""
=============================================================
ADAPTIVE-DMS
=============================================================

GRU Temporal Fatigue Model Training
Version:
    v1.8 - Step 2

Purpose:
    Train the GRU temporal fatigue prediction model using
    existing ADAPTIVE-DMS session CSV logs.

Input:
    logs/*.csv
    logs/session/*.csv
    logs/**/*.csv

Output:
    models/gru_fatigue_model.keras
    models/gru_normalization.npz

Features:
    - fatigue_risk
    - ear
    - mar
    - perclos
    - pitch
    - yaw
    - roll
    - horizontal_ratio
    - vertical_ratio
    - gaze_away_duration
    - reliability

Sequence:
    20 samples

Target:
    Binary fatigue classification.

    fatigue_risk >= 0.50 -> FATIGUED
    fatigue_risk <  0.50 -> NORMAL

IMPORTANT:
    This file only trains the GRU model.

    It does NOT modify:
        main.py
        dashboard.py
        existing v1.7 modules
=============================================================
"""

import os
import glob
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =============================================================
# PROJECT PATHS
# =============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

LOG_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "logs",
)

SESSION_DIRECTORY = os.path.join(
    LOG_DIRECTORY,
    "session",
)

MODEL_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "models",
)

MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "gru_fatigue_model.keras",
)

NORMALIZATION_PATH = os.path.join(
    MODEL_DIRECTORY,
    "gru_normalization.npz",
)


# =============================================================
# TRAINING CONFIGURATION
# =============================================================

SEQUENCE_LENGTH = 20

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.20

RANDOM_STATE = 42

EPOCHS = 30

BATCH_SIZE = 32

FATIGUE_THRESHOLD = 0.50

MINIMUM_ROWS_PER_SESSION = 25


# =============================================================
# FEATURE CONFIGURATION
# =============================================================

FEATURE_NAMES = [

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


# =============================================================
# RANDOM SEEDS
# =============================================================

np.random.seed(
    RANDOM_STATE
)

random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)


# =============================================================
# HEADER
# =============================================================

def print_header():

    print()

    print(
        "=" * 70
    )

    print(
        "ADAPTIVE-DMS"
    )

    print(
        "GRU TEMPORAL FATIGUE MODEL TRAINING"
    )

    print(
        "v1.8 - STEP 2"
    )

    print(
        "=" * 70
    )

    print()


# =============================================================
# FIND SESSION FILES
# =============================================================

def find_session_files():

    """
    Search for CSV files in all possible ADAPTIVE-DMS
    logging locations.

    Supported:

        logs/*.csv
        logs/session/*.csv
        logs/**/*.csv
    """

    patterns = [

        os.path.join(
            PROJECT_ROOT,
            "logs",
            "*.csv",
        ),

        os.path.join(
            PROJECT_ROOT,
            "logs",
            "session",
            "*.csv",
        ),

        os.path.join(
            PROJECT_ROOT,
            "logs",
            "**",
            "*.csv",
        ),

    ]

    files = []

    for pattern in patterns:

        try:

            files.extend(
                glob.glob(
                    pattern,
                    recursive=True,
                )
            )

        except Exception as error:

            print(
                f"Warning while searching "
                f"{pattern}: {error}"
            )

    # ---------------------------------------------------------
    # Remove duplicates
    # ---------------------------------------------------------

    files = sorted(
        list(
            set(
                os.path.abspath(
                    file
                )
                for file in files
            )
        )
    )

    # ---------------------------------------------------------
    # Ignore temporary files
    # ---------------------------------------------------------

    files = [

        file

        for file in files

        if not os.path.basename(
            file
        ).startswith(
            "~$"
        )

    ]

    return files


# =============================================================
# NORMALIZE COLUMN NAME
# =============================================================

def normalize_column_name(
    column,
):

    name = str(
        column
    ).strip().lower()

    name = name.replace(
        " ",
        "_",
    )

    name = name.replace(
        "-",
        "_",
    )

    name = name.replace(
        "/",
        "_",
    )

    name = name.replace(
        "(",
        "",
    )

    name = name.replace(
        ")",
        "",
    )

    name = name.replace(
        "[",
        "",
    )

    name = name.replace(
        "]",
        "",
    )

    return name


# =============================================================
# PREPARE COLUMNS
# =============================================================

def prepare_columns(
    df,
):

    df = df.copy()

    df.columns = [

        normalize_column_name(
            column
        )

        for column in df.columns

    ]

    return df


# =============================================================
# FIND COLUMN
# =============================================================

def find_column(
    df,
    candidates,
):

    columns = set(
        df.columns
    )

    # ---------------------------------------------------------
    # Exact match first
    # ---------------------------------------------------------

    for candidate in candidates:

        candidate = (
            normalize_column_name(
                candidate
            )
        )

        if candidate in columns:

            return candidate

    # ---------------------------------------------------------
    # Partial match
    # ---------------------------------------------------------

    for candidate in candidates:

        candidate = (
            normalize_column_name(
                candidate
            )
        )

        for column in df.columns:

            if (
                candidate in column
                or column in candidate
            ):

                return column

    return None


# =============================================================
# MAP COLUMNS
# =============================================================

def map_columns(
    df,
):

    mapping = {}

    # ---------------------------------------------------------
    # FATIGUE RISK
    # ---------------------------------------------------------

    mapping[
        "fatigue_risk"
    ] = find_column(
        df,
        [
            "fatigue_risk",
            "fusion_fatigue_risk",
            "fusion_risk",
            "risk",
            "fatigue",
            "predicted_fatigue_risk",
        ],
    )

    # ---------------------------------------------------------
    # EAR
    # ---------------------------------------------------------

    mapping[
        "ear"
    ] = find_column(
        df,
        [
            "ear",
            "eye_aspect_ratio",
            "eye_ratio",
        ],
    )

    # ---------------------------------------------------------
    # MAR
    # ---------------------------------------------------------

    mapping[
        "mar"
    ] = find_column(
        df,
        [
            "mar",
            "mouth_aspect_ratio",
            "mouth_ratio",
        ],
    )

    # ---------------------------------------------------------
    # PERCLOS
    # ---------------------------------------------------------

    mapping[
        "perclos"
    ] = find_column(
        df,
        [
            "perclos",
            "eye_closure_ratio",
            "perclos_value",
        ],
    )

    # ---------------------------------------------------------
    # PITCH
    # ---------------------------------------------------------

    mapping[
        "pitch"
    ] = find_column(
        df,
        [
            "pitch",
            "head_pitch",
        ],
    )

    # ---------------------------------------------------------
    # YAW
    # ---------------------------------------------------------

    mapping[
        "yaw"
    ] = find_column(
        df,
        [
            "yaw",
            "head_yaw",
        ],
    )

    # ---------------------------------------------------------
    # ROLL
    # ---------------------------------------------------------

    mapping[
        "roll"
    ] = find_column(
        df,
        [
            "roll",
            "head_roll",
        ],
    )

    # ---------------------------------------------------------
    # HORIZONTAL GAZE
    # ---------------------------------------------------------

    mapping[
        "horizontal_ratio"
    ] = find_column(
        df,
        [
            "horizontal_ratio",
            "gaze_horizontal_ratio",
            "horizontal_gaze_ratio",
            "gaze_x",
        ],
    )

    # ---------------------------------------------------------
    # VERTICAL GAZE
    # ---------------------------------------------------------

    mapping[
        "vertical_ratio"
    ] = find_column(
        df,
        [
            "vertical_ratio",
            "gaze_vertical_ratio",
            "vertical_gaze_ratio",
            "gaze_y",
        ],
    )

    # ---------------------------------------------------------
    # GAZE AWAY DURATION
    # ---------------------------------------------------------

    mapping[
        "gaze_away_duration"
    ] = find_column(
        df,
        [
            "gaze_away_duration",
            "gaze_away_time",
            "prolonged_gaze_duration",
        ],
    )

    # ---------------------------------------------------------
    # RELIABILITY
    # ---------------------------------------------------------

    mapping[
        "reliability"
    ] = find_column(
        df,
        [
            "overall_reliability",
            "reliability",
            "signal_reliability",
            "fusion_reliability",
        ],
    )

    return mapping


# =============================================================
# SHOW CSV COLUMNS
# =============================================================

def show_csv_columns(
    df,
):

    print()

    print(
        "Available CSV columns:"
    )

    for index, column in enumerate(
        df.columns,
        start=1,
    ):

        print(
            f"  {index:02d}. {column}"
        )

    print()


# =============================================================
# LOAD CSV
# =============================================================

def load_csv(
    file_path,
):

    print()

    print(
        "-" * 70
    )

    print(
        f"Loading:"
    )

    print(
        f"  {file_path}"
    )

    print(
        "-" * 70
    )

    try:

        df = pd.read_csv(
            file_path,
            low_memory=False,
        )

    except Exception as error:

        print(
            f"ERROR reading CSV:"
        )

        print(
            f"  {error}"
        )

        return None

    if df.empty:

        print(
            "WARNING: CSV is empty."
        )

        return None

    print(
        f"Original rows: {len(df)}"
    )

    print(
        f"Original columns: {len(df.columns)}"
    )

    df = prepare_columns(
        df
    )

    mapping = map_columns(
        df
    )

    # ---------------------------------------------------------
    # Display mappings
    # ---------------------------------------------------------

    print()

    print(
        "Detected feature columns:"
    )

    for feature in FEATURE_NAMES:

        source = mapping.get(
            feature
        )

        if source is None:

            print(
                f"  {feature:<22} -> NOT FOUND"
            )

        else:

            print(
                f"  {feature:<22} -> {source}"
            )

    # ---------------------------------------------------------
    # If fatigue risk missing, stop this file.
    # ---------------------------------------------------------

    if mapping[
        "fatigue_risk"
    ] is None:

        print()

        print(
            "WARNING:"
        )

        print(
            "fatigue_risk column was not found."
        )

        show_csv_columns(
            df
        )

        print(
            "Skipping this file."
        )

        return None

    # ---------------------------------------------------------
    # Standardized dataframe
    # ---------------------------------------------------------

    output = pd.DataFrame(
        index=df.index
    )

    for feature in FEATURE_NAMES:

        source_column = mapping.get(
            feature
        )

        if source_column is None:

            # -------------------------------------------------
            # Safe defaults
            # -------------------------------------------------

            if feature in (
                "horizontal_ratio",
                "vertical_ratio",
            ):

                default_value = 0.5

            elif feature == "reliability":

                default_value = 0.0

            else:

                default_value = 0.0

            output[
                feature
            ] = default_value

        else:

            output[
                feature
            ] = pd.to_numeric(
                df[
                    source_column
                ],
                errors="coerce",
            )

    # ---------------------------------------------------------
    # Replace infinities
    # ---------------------------------------------------------

    output = output.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # ---------------------------------------------------------
    # Fill missing values
    # ---------------------------------------------------------

    output = output.ffill()

    output = output.bfill()

    output = output.fillna(
        0.0
    )

    # ---------------------------------------------------------
    # Clip values
    # ---------------------------------------------------------

    output[
        "fatigue_risk"
    ] = np.clip(
        output[
            "fatigue_risk"
        ],
        0.0,
        1.0,
    )

    output[
        "perclos"
    ] = np.clip(
        output[
            "perclos"
        ],
        0.0,
        1.0,
    )

    output[
        "horizontal_ratio"
    ] = np.clip(
        output[
            "horizontal_ratio"
        ],
        0.0,
        1.0,
    )

    output[
        "vertical_ratio"
    ] = np.clip(
        output[
            "vertical_ratio"
        ],
        0.0,
        1.0,
    )

    output[
        "reliability"
    ] = np.clip(
        output[
            "reliability"
        ],
        0.0,
        1.0,
    )

    # ---------------------------------------------------------
    # Remove completely invalid rows
    # ---------------------------------------------------------

    output = output.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    output = output.fillna(
        0.0
    )

    print()

    print(
        f"Usable rows: {len(output)}"
    )

    return output


# =============================================================
# CREATE LABELS
# =============================================================

def create_labels(
    df,
):

    labels = (

        df[
            "fatigue_risk"
        ]

        >= FATIGUE_THRESHOLD

    ).astype(
        np.int32
    )

    return labels.values


# =============================================================
# CREATE TEMPORAL SEQUENCES
# =============================================================

def create_sequences(
    data,
    labels,
):

    X = []

    y = []

    total_rows = len(
        data
    )

    if total_rows < (
        SEQUENCE_LENGTH
    ):

        return (

            np.empty(
                (
                    0,
                    SEQUENCE_LENGTH,
                    data.shape[1],
                ),
                dtype=np.float32,
            ),

            np.empty(
                (
                    0,
                ),
                dtype=np.int32,
            ),

        )

    for end in range(
        SEQUENCE_LENGTH,
        total_rows + 1,
    ):

        start = (
            end
            - SEQUENCE_LENGTH
        )

        sequence = data[
            start:end
        ]

        # -----------------------------------------------------
        # Label from final timestep
        # -----------------------------------------------------

        target = labels[
            end - 1
        ]

        X.append(
            sequence
        )

        y.append(
            target
        )

    return (

        np.asarray(
            X,
            dtype=np.float32,
        ),

        np.asarray(
            y,
            dtype=np.int32,
        ),

    )


# =============================================================
# BUILD DATASET
# =============================================================

def build_dataset(
    files,
):

    print()

    print(
        "=" * 70
    )

    print(
        "BUILDING TEMPORAL DATASET"
    )

    print(
        "=" * 70
    )

    all_sequences = []

    all_labels = []

    sessions_used = 0

    total_rows = 0

    for file_path in files:

        df = load_csv(
            file_path
        )

        if df is None:

            continue

        rows = len(
            df
        )

        total_rows += rows

        if rows < (
            MINIMUM_ROWS_PER_SESSION
        ):

            print()

            print(
                f"WARNING: Only {rows} rows."
            )

            print(
                f"Minimum recommended: "
                f"{MINIMUM_ROWS_PER_SESSION}"
            )

            print(
                "Skipping this session."
            )

            continue

        # -----------------------------------------------------
        # Convert to NumPy
        # -----------------------------------------------------

        data = df[
            FEATURE_NAMES
        ].values.astype(
            np.float32
        )

        labels = create_labels(
            df
        )

        # -----------------------------------------------------
        # Create sequences
        # -----------------------------------------------------

        X, y = create_sequences(
            data,
            labels,
        )

        if len(X) == 0:

            print(
                "WARNING: No sequences created."
            )

            continue

        all_sequences.append(
            X
        )

        all_labels.append(
            y
        )

        sessions_used += 1

        print()

        print(
            f"Sequences created: "
            f"{len(X)}"
        )

        print(
            f"Normal: "
            f"{int(np.sum(y == 0))}"
        )

        print(
            f"Fatigued: "
            f"{int(np.sum(y == 1))}"
        )

    # ---------------------------------------------------------
    # Nothing found
    # ---------------------------------------------------------

    if not all_sequences:

        raise RuntimeError(
            "No temporal sequences were created."
        )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    X = np.concatenate(
        all_sequences,
        axis=0,
    )

    y = np.concatenate(
        all_labels,
        axis=0,
    )

    print()

    print(
        "=" * 70
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Sessions used:       {sessions_used}"
    )

    print(
        f"Total raw rows:      {total_rows}"
    )

    print(
        f"Total sequences:     {len(X)}"
    )

    print(
        f"Sequence length:     {X.shape[1]}"
    )

    print(
        f"Feature count:       {X.shape[2]}"
    )

    print(
        f"Normal sequences:    "
        f"{int(np.sum(y == 0))}"
    )

    print(
        f"Fatigued sequences:  "
        f"{int(np.sum(y == 1))}"
    )

    print(
        "=" * 70
    )

    return X, y


# =============================================================
# VALIDATE CLASSES
# =============================================================

def validate_dataset(
    y,
):

    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    print()

    print(
        "CLASS DISTRIBUTION"
    )

    for class_value, count in zip(
        classes,
        counts,
    ):

        if class_value == 0:

            label_name = (
                "NORMAL"
            )

        else:

            label_name = (
                "FATIGUED"
            )

        print(
            f"  {label_name:<10}: {count}"
        )

    print()

    if len(classes) < 2:

        raise RuntimeError(
            "\n"
            "Training data contains only ONE class.\n\n"
            "The GRU requires both NORMAL and FATIGUED "
            "samples.\n\n"
            "Current labeling rule:\n"
            "    fatigue_risk <  0.50 -> NORMAL\n"
            "    fatigue_risk >= 0.50 -> FATIGUED\n\n"
            "Collect more session data with varying "
            "fatigue-risk values before training."
        )


# =============================================================
# SPLIT DATASET
# =============================================================

def split_dataset(
    X,
    y,
):

    print(
        "Splitting dataset..."
    )

    # ---------------------------------------------------------
    # First split: training + test
    # ---------------------------------------------------------

    try:

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y,

        )

    except ValueError as error:

        raise RuntimeError(
            "Unable to create stratified "
            f"train/test split: {error}"
        )

    # ---------------------------------------------------------
    # Second split: training + validation
    # ---------------------------------------------------------

    try:

        (
            X_train,
            X_validation,
            y_train,
            y_validation,
        ) = train_test_split(

            X_train,

            y_train,

            test_size=VALIDATION_SIZE,

            random_state=RANDOM_STATE,

            stratify=y_train,

        )

    except ValueError as error:

        raise RuntimeError(
            "Unable to create stratified "
            f"training/validation split: {error}"
        )

    print(
        f"Training samples:   {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_validation)}"
    )

    print(
        f"Test samples:       {len(X_test)}"
    )

    return (

        X_train,

        X_validation,

        X_test,

        y_train,

        y_validation,

        y_test,

    )


# =============================================================
# CALCULATE NORMALIZATION
# =============================================================

def calculate_normalization(
    X_train,
):

    # ---------------------------------------------------------
    # Flatten time dimension
    # ---------------------------------------------------------

    flat = X_train.reshape(
        -1,
        X_train.shape[-1],
    )

    feature_min = np.min(
        flat,
        axis=0,
    )

    feature_max = np.max(
        flat,
        axis=0,
    )

    # ---------------------------------------------------------
    # Prevent zero denominator
    # ---------------------------------------------------------

    equal_mask = (
        feature_max
        == feature_min
    )

    feature_max[
        equal_mask
    ] = (
        feature_min[
            equal_mask
        ]
        + 1.0
    )

    return (

        feature_min.astype(
            np.float32
        ),

        feature_max.astype(
            np.float32
        ),

    )


# =============================================================
# NORMALIZE
# =============================================================

def normalize_data(
    data,
    feature_min,
    feature_max,
):

    denominator = (
        feature_max
        - feature_min
    )

    denominator = np.where(
        denominator == 0,
        1.0,
        denominator,
    )

    normalized = (
        data
        - feature_min
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


# =============================================================
# NORMALIZE DATASETS
# =============================================================

def normalize_datasets(
    X_train,
    X_validation,
    X_test,
):

    print()
    print(
        "Calculating training normalization..."
    )

    feature_min, feature_max = (
        calculate_normalization(
            X_train
        )
    )

    X_train = normalize_data(
        X_train,
        feature_min,
        feature_max,
    )

    X_validation = normalize_data(
        X_validation,
        feature_min,
        feature_max,
    )

    X_test = normalize_data(
        X_test,
        feature_min,
        feature_max,
    )

    return (

        X_train,

        X_validation,

        X_test,

        feature_min,

        feature_max,

    )


# =============================================================
# SAVE NORMALIZATION
# =============================================================

def save_normalization(
    feature_min,
    feature_max,
):

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True,
    )

    np.savez(
        NORMALIZATION_PATH,

        feature_min=feature_min,

        feature_max=feature_max,

        feature_names=np.asarray(
            FEATURE_NAMES
        ),

    )

    print()

    print(
        "Normalization saved:"
    )

    print(
        f"  {NORMALIZATION_PATH}"
    )


# =============================================================
# BUILD GRU
# =============================================================

def build_gru_model(
    sequence_length,
    feature_count,
):

    print()

    print(
        "=" * 70
    )

    print(
        "BUILDING GRU MODEL"
    )

    print(
        "=" * 70
    )

    model = tf.keras.Sequential(

        [

            tf.keras.layers.Input(
                shape=(
                    sequence_length,
                    feature_count,
                )
            ),

            tf.keras.layers.GRU(
                64,
                return_sequences=True,
            ),

            tf.keras.layers.Dropout(
                0.20
            ),

            tf.keras.layers.GRU(
                32,
                return_sequences=False,
            ),

            tf.keras.layers.Dropout(
                0.20
            ),

            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
            ),

        ]

    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
        ],

    )

    model.summary()

    return model


# =============================================================
# CLASS WEIGHTS
# =============================================================

def calculate_class_weights(
    y,
):

    normal_count = int(
        np.sum(
            y == 0
        )
    )

    fatigue_count = int(
        np.sum(
            y == 1
        )
    )

    total = (
        normal_count
        + fatigue_count
    )

    if (
        normal_count == 0
        or fatigue_count == 0
    ):

        return None

    normal_weight = (
        total
        / (
            2.0
            * normal_count
        )
    )

    fatigue_weight = (
        total
        / (
            2.0
            * fatigue_count
        )
    )

    weights = {

        0: normal_weight,

        1: fatigue_weight,

    }

    print()

    print(
        "CLASS WEIGHTS"
    )

    print(
        f"  NORMAL:   "
        f"{normal_weight:.4f}"
    )

    print(
        f"  FATIGUED: "
        f"{fatigue_weight:.4f}"
    )

    return weights


# =============================================================
# TRAIN
# =============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    class_weights,
):

    print()

    print(
        "=" * 70
    )

    print(
        "STARTING GRU TRAINING"
    )

    print(
        "=" * 70
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=7,

            restore_best_weights=True,

            verbose=1,

        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=3,

            min_lr=1e-6,

            verbose=1,

        ),

    ]

    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_validation,

            y_validation,

        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1,

    )

    return history


# =============================================================
# EVALUATE
# =============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    print()

    print(
        "=" * 70
    )

    print(
        "MODEL EVALUATION"
    )

    print(
        "=" * 70
    )

    loss, accuracy = model.evaluate(

        X_test,

        y_test,

        verbose=0,

    )

    print()

    print(
        f"Test Loss:     {loss:.4f}"
    )

    print(
        f"Test Accuracy: {accuracy:.4f}"
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    probabilities = (

        model.predict(
            X_test,
            verbose=0,
        )

        .reshape(
            -1
        )

    )

    predictions = (

        probabilities
        >= 0.50

    ).astype(
        np.int32
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy_value = (
        accuracy_score(
            y_test,
            predictions,
        )
    )

    precision_value = (
        precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    recall_value = (
        recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    f1_value = (
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print()

    print(
        "Classification Metrics"
    )

    print(
        f"Accuracy:  "
        f"{accuracy_value:.4f}"
    )

    print(
        f"Precision: "
        f"{precision_value:.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_value:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{f1_value:.4f}"
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print()

    print(
        "Confusion Matrix"
    )

    print(
        matrix
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    print()

    print(
        "Classification Report"
    )

    print(
        classification_report(

            y_test,

            predictions,

            target_names=[
                "NORMAL",
                "FATIGUED",
            ],

            zero_division=0,

        )
    )

    return {

        "loss": float(
            loss
        ),

        "accuracy": float(
            accuracy_value
        ),

        "precision": float(
            precision_value
        ),

        "recall": float(
            recall_value
        ),

        "f1": float(
            f1_value
        ),

    }


# =============================================================
# SAVE MODEL
# =============================================================

def save_model(
    model,
):

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True,
    )

    model.save(
        MODEL_PATH
    )

    print()

    print(
        "GRU MODEL SAVED:"
    )

    print(
        f"  {MODEL_PATH}"
    )


# =============================================================
# MAIN
# =============================================================

def main():

    print_header()

    # ---------------------------------------------------------
    # TensorFlow
    # ---------------------------------------------------------

    print(
        f"TensorFlow version: "
        f"{tf.__version__}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Feature count: "
        f"{len(FEATURE_NAMES)}"
    )

    print()

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    print(
        "Project root:"
    )

    print(
        f"  {PROJECT_ROOT}"
    )

    print()

    print(
        "Searching for session CSV files..."
    )

    print()

    # ---------------------------------------------------------
    # Find CSVs
    # ---------------------------------------------------------

    files = find_session_files()

    if not files:

        print(
            "=" * 70
        )

        print(
            "ERROR"
        )

        print(
            "=" * 70
        )

        print(
            "No session CSV files were found."
        )

        print()

        print(
            "Searched:"
        )

        print(
            f"  {LOG_DIRECTORY}"
        )

        print(
            f"  {SESSION_DIRECTORY}"
        )

        print(
            f"  {os.path.join(LOG_DIRECTORY, '**', '*.csv')}"
        )

        print()

        print(
            "Run ADAPTIVE-DMS first so that "
            "SessionLogger creates a CSV."
        )

        print()

        return

    # ---------------------------------------------------------
    # Display files
    # ---------------------------------------------------------

    print(
        f"Session CSV files found: "
        f"{len(files)}"
    )

    print()

    for index, file_path in enumerate(
        files,
        start=1,
    ):

        print(
            f"{index:02d}. {file_path}"
        )

    print()

    # ---------------------------------------------------------
    # Build dataset
    # ---------------------------------------------------------

    try:

        X, y = build_dataset(
            files
        )

    except Exception as error:

        print()

        print(
            "=" * 70
        )

        print(
            "DATASET ERROR"
        )

        print(
            "=" * 70
        )

        print(
            str(error)
        )

        print()

        return

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    try:

        validate_dataset(
            y
        )

    except RuntimeError as error:

        print(
            str(error)
        )

        return

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    try:

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
        ) = split_dataset(
            X,
            y,
        )

    except RuntimeError as error:

        print()

        print(
            str(error)
        )

        return

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        feature_min,
        feature_max,
    ) = normalize_datasets(

        X_train,

        X_validation,

        X_test,

    )

    # ---------------------------------------------------------
    # Save normalization
    # ---------------------------------------------------------

    save_normalization(

        feature_min,

        feature_max,

    )

    # ---------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            y_train
        )
    )

    # ---------------------------------------------------------
    # Build GRU
    # ---------------------------------------------------------

    model = build_gru_model(

        sequence_length=(
            X_train.shape[1]
        ),

        feature_count=(
            X_train.shape[2]
        ),

    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    train_model(

        model=model,

        X_train=X_train,

        y_train=y_train,

        X_validation=X_validation,

        y_validation=y_validation,

        class_weights=class_weights,

    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    metrics = evaluate_model(

        model,

        X_test,

        y_test,

    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    save_model(
        model
    )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print()

    print(
        "Normalization:"
    )

    print(
        f"  {NORMALIZATION_PATH}"
    )

    print()

    print(
        "Final Test Metrics:"
    )

    print(
        f"  Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"  F1 Score : "
        f"{metrics['f1']:.4f}"
    )

    print()

    print(
        "v1.8 STEP 2 COMPLETE"
    )

    print(
        "=" * 70
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()