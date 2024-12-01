import logging

import pandas as pd
import yaml

from utils.constants import DataField


def load_yaml_file(path: str) -> dict:
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data


class CustomFormatter(logging.Formatter):
    purple = "\x1b[1;35m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s: [%(name)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: purple + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(root_logger: str) -> logging.Logger:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
    )

    return logging.getLogger(root_logger)


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def prepare_single_prediction_data(
    sepal_length: str, sepal_width: str, petal_length: str, petal_width: str
) -> tuple[pd.DataFrame, str | None]:
    data = pd.DataFrame(
        dict(
            zip(
                DataField.FEATURES,
                [[sepal_length], [sepal_width], [petal_length], [petal_width]],
            )
        )
    )

    is_valid = [
        is_number(n) for n in [sepal_length, sepal_width, petal_length, petal_width]
    ]

    if all(is_valid):
        return data, None

    return (
        pd.DataFrame(),
        "One or some of the inputs are invalid. All inputs must be float.",
    )


def prepare_batch_prediction_data(
    uploaded_data: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    if DataField.FEATURES != tuple(uploaded_data.columns):
        return pd.DataFrame(), (
            "Column names are invalid or not ordered correctly. "
            f"Column names must be and ordered as {DataField.FEATURES}."
        )

    # Check for strings
    contains_strings = uploaded_data.map(lambda x: isinstance(x, str)).any().any()

    # Check for NaN values
    contains_nan = uploaded_data.isnull().any().any()

    if contains_strings | contains_nan:
        return pd.DataFrame(), (
            "Some values in the uploaded CSV contains strings and/or "
            "NaN values. Please check the file and re-upload."
        )

    return uploaded_data, None


def get_iris_dictionary() -> dict[int, str]:
    return dict(zip([0, 1, 2], ["setosa", "versicolor", "virginica"]))
