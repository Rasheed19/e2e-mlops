import argparse
import os
import joblib
import json
import pandas as pd
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n-estimators", type=str, default="100 200 300")
    parser.add_argument("--learning-rate", type=str, default="0.1 0.01")
    parser.add_argument("--max-depth", type=str, default="2 4 6")

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args, _ = parser.parse_known_args()

    dataset_trn = pd.read_csv(
        filepath_or_buffer=args.train,
        index_col=False,
        engine="python",
    )

    X_train = dataset_trn.drop(columns=["iris"])
    y_train = dataset_trn["iris"]

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier()),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [int(el) for el in args.n_estimators.split(" ")],
        "classifier__learning_rate": [
            float(el) for el in args.learning_rate.split(" ")
        ],
        "classifier__max_depth": [int(el) for el in args.max_depth.split(" ")],
    }

    logger.info(f"Training model...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        refit=True,
    )
    grid_search.fit(X=X_train, y=y_train)

    logger.info(
        f"best_params: {grid_search.best_params_}, "
        f"cv_accuracy: {grid_search.best_score_}"
    )

    joblib.dump(
        grid_search.best_estimator_, os.path.join(args.model_dir, "model.joblib")
    )


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        request_body = json.loads(request_body)
        input = request_body["Input"]
        return input
    else:
        raise ValueError("This model only supports application/json input")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, content_type):
    output = {"Output": prediction.tolist()}
    return output
