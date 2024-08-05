import pandas as pd

data = pd.read_csv(filepath_or_buffer="./data/train.csv", index_col=False)


def test_column_match():

    valid_column_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "iris",
    ]

    assert valid_column_names == list(data.columns), (
        "Column names are invalid or not ordered correctly. "
        f"Column names must be and ordered as {valid_column_names}."
    )


def test_valid_values():

    contains_strings = data.map(lambda x: isinstance(x, str)).any().any()
    contains_nan = data.isnull().any().any()

    assert (contains_strings | contains_nan) == False, (
        "Some values in the uploaded CSV contains strings and/or "
        "NaN values. Please check the file and re-upload."
    )
