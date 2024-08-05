import pandas as pd
from sklearn.model_selection import train_test_split

from utils.helper import get_logger

logger = get_logger(__name__)


def train_data_splitter(
    dataset: pd.DataFrame, target_name: str, test_size: float = 0.2
) -> tuple[str, str]:

    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=dataset[target_name].values,
    )
    train_data = pd.DataFrame(train_data, columns=dataset.columns)
    test_data = pd.DataFrame(test_data, columns=dataset.columns)

    path_to_train_data = "./data/train.csv"
    path_to_test_data = "./data/test.csv"

    train_data.to_csv(path_or_buf=path_to_train_data, index=False)
    train_data.to_csv(
        path_or_buf=path_to_test_data,
        index=False,
    )

    logger.info(
        f"train data written to {path_to_train_data}, "
        f"test data written to {path_to_test_data}."
    )

    return path_to_train_data, path_to_test_data
