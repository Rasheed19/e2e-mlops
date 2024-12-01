from typing import Any

import pandas as pd
from sagemaker.session import Session

from steps import (
    create_data_drift_baseline,
    create_data_drift_report,
    create_data_monitoring_schedule,
    create_model_drift_report,
    extract_captured_data,
    fetch_model_monitoring_data,
    predict_iris,
)
from utils.constants import DataField
from utils.helper import get_logger

logger = get_logger(__name__)


def sm_data_drift_monitoring_pipeline(
    role: str,
    session: Session,
    instance_count: int,
    instance_type: str,
    baseline_dataset_s3_uri: str,
    data_drift_baseline_results_s3_uri: str,
    data_drift_monitoring_results_s3_uri: str,
    data_drift_monitoring_schedule_name: str,
    endpoint_input: str,
    schedule_cron_expression: str,
) -> None:
    logger.info(
        f"Starting data drift monitoring pipeline with name {data_drift_monitoring_schedule_name} from {endpoint_input}..."
    )

    logger.info("Creating data drift baseline...")
    data_drift_monitor = create_data_drift_baseline(
        role=role,
        session=session,
        instance_count=instance_count,
        instance_type=instance_type,
        baseline_dataset_s3_uri=baseline_dataset_s3_uri,
        data_drift_baseline_results_s3_uri=data_drift_baseline_results_s3_uri,
    )

    logger.info("Creating data monitoring schedule...")

    create_data_monitoring_schedule(
        monitor=data_drift_monitor,
        monitor_schedule_name=data_drift_monitoring_schedule_name,
        endpoint_input=endpoint_input,
        data_drift_monitoring_results_s3_uri=data_drift_monitoring_results_s3_uri,
        schedule_cron_expression=schedule_cron_expression,
    )

    logger.info(
        "Data drift monitoring pipeline completed. "
        f"Baseline results written to {data_drift_baseline_results_s3_uri}. "
        f"Monitoring results written to {data_drift_monitoring_results_s3_uri}."
    )

    return None


def evidently_data_drift_monitoring_pipeline(
    s3_clent: Any,
    s3_bucket_name: str,
    data_capture_prefix: str,
    baseline_data_s3_uri: str,
    endpoint_name: str,
    evidently_api_token: str,
    evidently_propject_id: str,
) -> None:
    logger.info(
        f"Starting data drift monitoring pipeline for the endpoint {endpoint_name}..."
    )

    logger.info("Extracting captured data up to date...")
    captured_data, err = extract_captured_data(
        s3_client=s3_clent,
        s3_bucket_name=s3_bucket_name,
        data_capture_prefix=data_capture_prefix,
        endpoint_name=endpoint_name,
    )

    if err is not None:
        logger.error(err)
        return None

    logger.info("Fetching baseline data for data monitoring...")
    baseline_data = pd.read_csv(
        filepath_or_buffer=baseline_data_s3_uri,
    )

    # check if "target" is in the colums; if so, remove it as it is not needed here
    if DataField.TARGET in baseline_data.columns:
        baseline_data = baseline_data.drop([DataField.TARGET], axis=1)
    if DataField.TARGET in captured_data.columns:
        captured_data = captured_data.drop([DataField.TARGET], axis=1)

    logger.info("Creating data drift report...")
    create_data_drift_report(
        evidently_api_token=evidently_api_token,
        evidently_project_id=evidently_propject_id,
        baseline_data=baseline_data,
        current_data=captured_data,
        report_metadata={
            "endpoint": {
                "name": endpoint_name,
                "data_capture_prefix": data_capture_prefix,
                "baseline_dataset_uri": baseline_data_s3_uri,
            },
            "data": {
                "baseline_datapoints": baseline_data.shape[0],
                "captured_datapoints": captured_data.shape[0],
            },
        },
    )

    logger.info("Report created. Log in to Evidently AI to see the results.")

    return None


def evidently_model_drift_monitoring_pipeline(
    baseline_data_s3_uri: str,
    current_data_s3_uri: str,
    endpoint_name: str,
    evidently_api_token: str,
    evidently_propject_id: str,
) -> None:
    logger.info(
        f"Starting model drift monitoring pipeline for the endpoint {endpoint_name}..."
    )

    logger.info("Fetching baseline and current data up to date...")
    baseline_data, current_data, err = fetch_model_monitoring_data(
        baseline_data_s3_uri=baseline_data_s3_uri,
        current_data_s3_uri=current_data_s3_uri,
    )

    if err is not None:
        logger.error(err)
        return None

    # create a prediction column in the baseline dataset;
    # infact, the current data must have the colums named
    # "target" and "prediction" representing the true
    # and predicted labels consecutively
    assert {DataField.TARGET, DataField.PREDICTION}.issubset(
        set(list(current_data.columns))
    ), f"data must have {DataField.TARGET} and {DataField.PREDICTION} columns"

    request_body = {
        "Input": baseline_data.drop([DataField.TARGET], axis=1).values.tolist()
    }
    predictions = predict_iris(
        request_body=request_body,
        endpoint_name=endpoint_name,
    )
    baseline_data[DataField.PREDICTION] = predictions

    logger.info("Creating model drift report...")
    create_model_drift_report(
        evidently_api_token=evidently_api_token,
        evidently_project_id=evidently_propject_id,
        baseline_data=baseline_data,
        current_data=current_data,
        report_metadata={
            "endpoint": {
                "name": endpoint_name,
                "current_data_s3_uri": current_data_s3_uri,
                "baseline_dataset_uri": baseline_data_s3_uri,
            },
            "data": {
                "baseline_datapoints": baseline_data.shape[0],
                "current_datapoints": current_data.shape[0],
            },
        },
    )

    logger.info("Report created. Log in to Evidently AI to see the results.")

    return None
