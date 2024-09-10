import pandas as pd
import pytest


@pytest.fixture
def iris() -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer="./data/train.csv", index_col=False)


def test_column_match(iris):
    valid_column_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "iris",
    ]

    assert valid_column_names == list(iris.columns), (
        "Column names are invalid or not ordered correctly. "
        f"Column names must be and ordered as {valid_column_names}."
    )


def test_valid_values(iris):
    contains_strings = iris.map(lambda x: isinstance(x, str)).any().any()
    contains_nan = iris.isnull().any().any()

    print(contains_nan, contains_nan)

    assert not (contains_strings | contains_nan), (
        "Some values in the uploaded CSV contains strings and/or "
        "NaN values. Please check the file and re-upload."
    )
