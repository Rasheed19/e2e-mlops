import json
from datetime import datetime as dt
from typing import Any

import pandas as pd
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.report import Report
from evidently.ui.workspace.cloud import CloudWorkspace
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.session import Session

from utils.constants import DataField


def create_data_drift_baseline(
    role: str,
    session: Session,
    instance_count: int,
    instance_type: str,
    baseline_dataset_s3_uri: str,
    data_drift_baseline_results_s3_uri: str,
) -> DefaultModelMonitor:
    data_drift_monitor = DefaultModelMonitor(
        role=role,
        sagemaker_session=session,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
    )

    data_drift_monitor.suggest_baseline(
        baseline_dataset=baseline_dataset_s3_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=data_drift_baseline_results_s3_uri,
        wait=True,
        logs=True,
    )  # TODO: this is in wait mode; would be better to disable wating and implement  a pipeline mechanism to wait for the baseline job to complete
    return data_drift_monitor


def create_data_monitoring_schedule(
    monitor: DefaultModelMonitor,
    monitor_schedule_name: str,
    endpoint_input: str,
    data_drift_monitoring_results_s3_uri: str,
    schedule_cron_expression: str,
) -> None:
    monitor_schedule_name = (
        f"{monitor_schedule_name}-{dt.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    monitor.create_monitoring_schedule(
        endpoint_input=endpoint_input,
        monitor_schedule_name=monitor_schedule_name,
        record_preprocessor_script=None,  # we are not using record preprocessor in this example
        post_analytics_processor_script=None,  # we are not using post analytics processor in this example
        output_s3_uri=data_drift_monitoring_results_s3_uri,
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=schedule_cron_expression,
        enable_cloudwatch_metrics=True,
    )

    return None


def extract_captured_data(
    s3_client: Any,
    s3_bucket_name: str,
    data_capture_prefix: str,
    endpoint_name: str,
) -> tuple[pd.DataFrame, str | None]:
    # get objects in the captured data uri
    current_endpoint_capture_prefix = "{}/{}".format(data_capture_prefix, endpoint_name)
    result = s3_client.list_objects_v2(
        Bucket=s3_bucket_name, Prefix=current_endpoint_capture_prefix
    )
    captured_files = [file.get("Key") for file in result.get("Contents")]

    # handle the case when no data has been captured by the endpoint
    if not captured_files:
        return pd.DataFrame(), "Error: no data found in the captured dir"

    def get_obj_body(obj_key):
        return (
            s3_client.get_object(Bucket=s3_bucket_name, Key=obj_key)
            .get("Body")
            .read()
            .decode("utf-8")
        )

    features = []
    targets = []

    for obj_key in captured_files:
        body = json.loads(get_obj_body(obj_key).split("\n")[0])
        features.extend(
            json.loads(body["captureData"]["endpointInput"]["data"])["Input"]
        )
        targets.extend(
            json.loads(body["captureData"]["endpointOutput"]["data"])["Output"]
        )

    result = pd.DataFrame(data=features, columns=DataField.FEATURES)
    result = result.astype(float)

    result[DataField.TARGET] = targets

    return result, None


def create_data_drift_report(
    evidently_api_token: str,
    evidently_project_id: str,
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_metadata: dict[str, Any],
) -> None:
    ws = CloudWorkspace(token=evidently_api_token, url="https://app.evidently.cloud")

    data_report = Report(
        metrics=[
            DataDriftPreset(
                stattest="psi", stattest_threshold="0.3"
            ),  # a different test can be chosen
            DataQualityPreset(),
        ],
        timestamp=dt.now(),
        metadata=report_metadata,
    )

    data_report.run(reference_data=baseline_data, current_data=current_data)

    ws.add_report(evidently_project_id, data_report)

    return None


# SCENARIO: let us say we have BASELINE model drift monitoring  dataset
# in the s3 bucket; this data could be test data with the true
# labels. Let us also say we have another dataset in the s3 bucket.
# This data contains the MOST RECENT predictions from our model
# and the corresponding ground truths (we know them either from
# when the actual event that we predict happens or through
# our business logic). We will provide s3 uri of these two datasets
# to a function that will fetch them and return them for use
# in generating monitoring report.


def fetch_model_monitoring_data(
    baseline_data_s3_uri: str,
    current_data_s3_uri: str,
) -> tuple[pd.DataFrame, pd.DataFrame, None | str]:
    # here we fetch the baseline test or validation dataset,
    # make prediction on it using the endpoint in production;
    # the data with the most recent predictions and ground truths
    # is also fetched

    try:
        baseline_data = pd.read_csv(
            baseline_data_s3_uri
        )  # this contains the true targets and features
        current_data = pd.read_csv(
            current_data_s3_uri
        )  # this contains features, true targets and model predictions.

        err = None

    except Exception as e:
        baseline_data = pd.DataFrame()
        current_data = pd.DataFrame()
        err = f"error: {e}"

    return baseline_data, current_data, err


def create_model_drift_report(
    evidently_api_token: str,
    evidently_project_id: str,
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_metadata: dict[str, Any],
) -> None:
    ws = CloudWorkspace(token=evidently_api_token, url="https://app.evidently.cloud")

    model_report = Report(
        metrics=[ClassificationPreset()],
        timestamp=dt.now(),
        metadata=report_metadata,
    )

    model_report.run(reference_data=baseline_data, current_data=current_data)

    ws.add_report(evidently_project_id, model_report)

    return None
