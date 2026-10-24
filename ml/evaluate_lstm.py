import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mse
)


print(
    "\nLSTM MODEL EVALUATION"
)

print(
    "---------------------"
)

print(
    "Test samples:",
    len(y_test)
)

print(
    "MAE:",
    round(mae, 2)
)

print(
    "MSE:",
    round(mse, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)


print(
    "\nSample predictions:"
)

print(
    "-------------------"
)


for i in range(
    min(10, len(y_test))
):

    print(
        "Actual RUL:",
        round(y_test[i], 2),
        "| Predicted RUL:",
        round(predictions[i], 2)
    )