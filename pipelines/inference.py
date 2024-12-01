from datetime import datetime as dt

import pandas as pd
from sagemaker.session import Session

from steps import predict_iris
from utils.constants import DataField
from utils.helper import (
    get_logger,
    prepare_batch_prediction_data,
)

logger = get_logger(__name__)


def inference_pipeline(
    session: Session,
    deployed_endpoint_name: str,
    input_path_prefix: str,
    inference_file_name: str,
    output_path_prefix: str,
) -> None:
    logger.info("Inference pipeline has started.")

    input_path = (
        f"s3://{session.default_bucket()}/{input_path_prefix}/{inference_file_name}"
    )
    data = pd.read_csv(
        filepath_or_buffer=input_path,
    )
    data, err = prepare_batch_prediction_data(uploaded_data=data)

    if err is not None:
        logger.error(err)
        return None

    logger.info("Getting predictions...")
    request_body = {"Input": data.values.tolist()}
    predictions = predict_iris(
        request_body=request_body,
        endpoint_name=deployed_endpoint_name,
    )

    data[DataField.PREDICTION] = predictions

    output_path = f"s3://{session.default_bucket()}/{output_path_prefix}/{inference_file_name.split('.')[0]}_predictions_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
    data.to_csv(path_or_buf=output_path, index=False)

    logger.info(
        "Inference pipeline finished successfully. "
        f"Predictions are written to {output_path}."
    )

    return None
