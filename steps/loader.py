import pandas as pd
from sklearn.datasets import load_iris

from utils.constants import DataField
from utils.helper import get_logger

logger = get_logger(__name__)


def data_loader() -> tuple[pd.DataFrame, str]:
    logger.info("Loading data...")

    target_name = DataField.TARGET
    raw_data = load_iris(as_frame=True)
    dataset = raw_data.data
    dataset[target_name] = raw_data.target

    logger.info(f"Dataset with {len(dataset)} records loaded!")

    return dataset, target_name
