import warnings
import mlflow
import mlflow.keras
import utils
import forecasting
import tensorflow as tf
import optuna
import logging
import argparse

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--out-steps", dest="OUT_STEPS", type=int, required=True)
parser.add_argument(
    "--train-percentage", dest="TRAIN_PERCENTAGE", type=float, default=0.7
)
parser.add_argument(
    "--validation-percentage", dest="VALIDATION_PERCENTAGE", type=float, default=0.2
)
parser.add_argument("--aggregation", dest="AGGREGATION", type=str, default="D")
parser.add_argument("--input", dest="INPUT", type=str, default="dataset.csv")
args = parser.parse_args()

OUT_STEPS = args.OUT_STEPS
TRAIN_PERCENTAGE = args.TRAIN_PERCENTAGE
VALIDATION_PERCENTAGE = args.VALIDATION_PERCENTAGE
AGGREGATION = args.AGGREGATION
INPUT_FILE = args.INPUT


df = utils.read_dataset(INPUT_FILE)
df = utils.preprocess_dataset(df)
df = utils.resample_dataset(df, AGGREGATION)

train_df, val_df, test_df = utils.split_dataset(
    df, TRAIN_PERCENTAGE, VALIDATION_PERCENTAGE
)

# Define objective function to optimize.
def objective(trial):
    tf.keras.backend.clear_session()
    lstm_units = trial.suggest_int("lstm_units", 10, 200)
    epochs = trial.suggest_int("epochs", 10, 20)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.5)
    # input_width = trial.suggest_int('input_width', 30, 365)
    input_width = trial.suggest_int("input_width", 3, 36)
    batch_size = trial.suggest_int("batch_size", 2, 20)

    multi_window = utils.WindowGenerator(
        input_width=input_width,
        label_width=OUT_STEPS,
        shift=OUT_STEPS,
        batch_size=batch_size,
        label_columns=["value"],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    forecaster = forecasting.SingleShotForecaster(lstm_units, OUT_STEPS)
    forecaster.compile(learning_rate=learning_rate)
    forecaster.fit(multi_window, epochs)
    return forecaster.raw_model.evaluate(multi_window.val)[0]


# Run hyperparameter optmizer n times.
study = optuna.create_study()
study.optimize(objective, n_trials=10)
best_params = study.best_params

with mlflow.start_run():
    # Generate a model with the best parameters
    forecaster = forecasting.SingleShotForecaster(
        study.best_params["lstm_units"], OUT_STEPS
    )

    multi_window = utils.WindowGenerator(
        input_width=best_params["input_width"],
        label_width=OUT_STEPS,
        shift=OUT_STEPS,
        batch_size=best_params["batch_size"],
        label_columns=["value"],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    forecaster.compile(learning_rate=best_params["learning_rate"])
    history = forecaster.fit(multi_window, best_params["epochs"])
    model = forecaster.raw_model

    # Store model parameters and metrics
    mlflow.log_param("lstm_units", best_params["lstm_units"])
    mlflow.log_param("epochs", best_params["epochs"])
    mlflow.log_param("learning_rate", best_params["learning_rate"])
    mlflow.log_param("input_width", best_params["input_width"])
    mlflow.log_param("batch_size", best_params["batch_size"])

    train_metrics = model.evaluate(multi_window.train)
    validation_metrics = model.evaluate(multi_window.val)
    test_metrics = model.evaluate(multi_window.test)

    mlflow.log_metric("train_rmse", train_metrics[0])
    mlflow.log_metric("train_mae", train_metrics[1])
    mlflow.log_metric("val_rmse", validation_metrics[0])
    mlflow.log_metric("val_mae", validation_metrics[1])
    mlflow.log_metric("test_rmse", test_metrics[0])
    mlflow.log_metric("test_mae", test_metrics[1])
    for loss in history.history["val_loss"]:
        mlflow.log_metric("val_loss", loss)
    for loss in history.history["loss"]:
        mlflow.log_metric("loss", loss)

    mlflow.keras.log_model(forecaster.raw_model, "model_here")
    multi_window.plot(model)
