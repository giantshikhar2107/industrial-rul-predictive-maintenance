import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


X_TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "X_test.npy"
)

Y_TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "y_test.npy"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "ml",
    "best_lstm_model.keras"
)

PREDICTION_PLOT = os.path.join(
    PROJECT_ROOT,
    "ml",
    "actual_vs_predicted_rul.png"
)

ERROR_PLOT = os.path.join(
    PROJECT_ROOT,
    "ml",
    "rul_prediction_error.png"
)


X_test = np.load(
    X_TEST_FILE
)

y_test = np.load(
    Y_TEST_FILE
)


model = load_model(
    MODEL_FILE,
    compile=False
)


predictions = model(
    X_test,
    training=False
).numpy().flatten()


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    y_test,
    label="Actual RUL"
)

plt.plot(
    predictions,
    label="Predicted RUL"
)

plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "RUL (Cycles)"
)

plt.title(
    "Actual vs Predicted RUL"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    PREDICTION_PLOT,
    dpi=300
)

plt.close()


errors = predictions - y_test


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    errors
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Prediction Error (Cycles)"
)

plt.title(
    "LSTM RUL Prediction Error"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    ERROR_PLOT,
    dpi=300
)

plt.close()


print(
    "\nPlots generated successfully."
)

print(
    "Actual vs Predicted:"
)

print(
    PREDICTION_PLOT
)

print(
    "\nPrediction Error:"
)

print(
    ERROR_PLOT
)