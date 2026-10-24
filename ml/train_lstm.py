import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


X_TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "X_train.npy"
)

Y_TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "y_train.npy"
)

X_VALIDATION_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "X_validation.npy"
)

Y_VALIDATION_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "y_validation.npy"
)

BEST_MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "ml",
    "best_lstm_model.keras"
)

FINAL_MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "ml",
    "final_lstm_model.keras"
)

HISTORY_FILE = os.path.join(
    PROJECT_ROOT,
    "ml",
    "training_history.npy"
)


X_train = np.load(
    X_TRAIN_FILE
)

y_train = np.load(
    Y_TRAIN_FILE
)

X_validation = np.load(
    X_VALIDATION_FILE
)

y_validation = np.load(
    Y_VALIDATION_FILE
)


print(
    "Training data:",
    X_train.shape
)

print(
    "Validation data:",
    X_validation.shape
)


model = Sequential([
    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(
        0.2
    ),

    LSTM(
        32
    ),

    Dropout(
        0.2
    ),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1
    )
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


model.summary()


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


checkpoint = ModelCheckpoint(
    BEST_MODEL_FILE,
    monitor="val_loss",
    save_best_only=True
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_validation,
        y_validation
    ),
    epochs=30,
    batch_size=256,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=1
)


model.save(
    FINAL_MODEL_FILE
)


np.save(
    HISTORY_FILE,
    history.history,
    allow_pickle=True
)


print(
    "\nTraining completed successfully."
)

print(
    "Best model:",
    BEST_MODEL_FILE
)

print(
    "Final model:",
    FINAL_MODEL_FILE
)

print(
    "Training history:",
    HISTORY_FILE
)