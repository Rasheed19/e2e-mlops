import tarfile
import json
import logging
import pathlib
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        # tar.extractall(path="..")
        tar.extractall(path="/opt/ml/processing/model")

    logger.info("Loading model.")
    # model: Pipeline = joblib.load("model.joblib")
    model: Pipeline = joblib.load("/opt/ml/processing/model/model.joblib")

    logger.info("Loading test input data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, index_col=False)

    logger.info("Reading test data.")
    y_test = df["iris"].values
    X_test = df.drop(["iris"], axis=1)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)
    prediction_probabilities = model.predict_proba(X_test)

    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1_sc = f1_score(y_test, predictions, average="weighted")
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-score: {f1_sc}")
    logger.info(f"Confusion matrix: {conf_matrix}")

    # Available metrics to add to model:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "f1": {"value": f1_sc, "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {
                    "0": int(conf_matrix[0][0]),
                    "1": int(conf_matrix[0][1]),
                    "2": int(conf_matrix[0][2]),
                },
                "1": {
                    "0": int(conf_matrix[1][0]),
                    "1": int(conf_matrix[1][1]),
                    "2": int(conf_matrix[1][2]),
                },
                "2": {
                    "0": int(conf_matrix[2][0]),
                    "1": int(conf_matrix[2][1]),
                    "2": int(conf_matrix[2][2]),
                },
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
