import os
from enum import StrEnum, auto
from types import MappingProxyType

import boto3
from sagemaker.session import Session


class PipelineMode(StrEnum):
    TRAIN = auto()
    DEPLOY = auto()
    CLEAN = auto()
    INFERENCE = auto()
    DATADRIFT = auto()
    MODELDRIFT = auto()


class AppPredictionMode(StrEnum):
    SINGLE = "Single-sample prediction"
    BATCH = "Batch prediction"


class ModelAPprovalStatus(StrEnum):
    PENDING_MANUAL_APPROVAL = "PendingManualApproval"
    APPROVED = "Approved"
    REJECTED = "Rejected"


class DataField:
    FEATURES = (
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    )
    TARGET = "target"
    PREDICTION = "prediction"


class Shared:
    ROLE = os.getenv("ROLE")
    S3_BUCKET_NAME = os.getenv(
        "S3_BUCKET_NAME"
    )  # note the name must start with "sagemaker";
    # see https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-bucket.html

    PROJECT_S3_PREFIX = "iris_prediction"
    TRAIN_KEY_PREFIX = f"{PROJECT_S3_PREFIX}/data/train"
    TEST_KEY_PREFIX = f"{PROJECT_S3_PREFIX}/data/test"
    DATA_CAPTURE_S3_PREFIX = f"{PROJECT_S3_PREFIX}/monitoring/data_capture"

    TRAIN_DATA_S3_URI = f"s3://{S3_BUCKET_NAME}/{TRAIN_KEY_PREFIX}/train.csv"
    TEST_DATA_S3_URI = f"s3://{S3_BUCKET_NAME}/{TEST_KEY_PREFIX}/test.csv"
    EVIDENTLY_API_TOKEN = os.getenv("EVIDENTLY_API_TOKEN")
    EVIDENTLY_PROJECT_ID = os.getenv("EVIDENTLY_PROJECT_ID")

    SESSION = Session(default_bucket=S3_BUCKET_NAME)
    SM_CLIENT = boto3.client("sagemaker")
    S3_CLIENT = boto3.client("s3")

    PIPELINE_NAME = "iris-prediction-pipeline"

    INSTANCE_TYPE = "ml.m5.xlarge"
    INSTANCE_COUNT = 1

    MODEL_PACKAGE_GROUP_NAME = "iris-prediction-model-package"
    ENDPOINT_NAME = "iris-prediction-endpoint"


class Ingest:
    ARGS = MappingProxyType(
        dict(
            test_size=0.2,
            session=Shared.SESSION,
            s3_bucket_name=Shared.S3_BUCKET_NAME,
            project_s3_prefix=Shared.PROJECT_S3_PREFIX,
            train_key_prefix=Shared.TRAIN_KEY_PREFIX,
            test_key_prefix=Shared.TEST_KEY_PREFIX,
        )
    )


class Train:
    ARGS = MappingProxyType(
        dict(
            pipeline_name=Shared.PIPELINE_NAME,
            session=Shared.SESSION,
            role=Shared.ROLE,
            s3_bucket_name=Shared.S3_BUCKET_NAME,
            project_s3_prefix=Shared.PROJECT_S3_PREFIX,
            train_data_uri=Shared.TRAIN_DATA_S3_URI,
            test_data_uri=Shared.TEST_DATA_S3_URI,
            hyperparameters={
                "n-estimators": "100 200 300",
                "learning-rate": "0.1 0.01",
                "max-depth": "2 4 6",
                "train": Shared.TRAIN_DATA_S3_URI,
            },
            framework_version="1.2-1",
            instance_type=Shared.INSTANCE_TYPE,
            instance_count=Shared.INSTANCE_COUNT,
            model_package_group_name=Shared.MODEL_PACKAGE_GROUP_NAME,
            model_approval_status=ModelAPprovalStatus.PENDING_MANUAL_APPROVAL,
            register_accuracy_threshold=0.8,
            output_path=f"s3://{Shared.S3_BUCKET_NAME}/{Shared.PROJECT_S3_PREFIX}/pipeline_runs",
            code_location=f"s3://{Shared.S3_BUCKET_NAME}/{Shared.PROJECT_S3_PREFIX}/uploaded_codes",
        )
    )


class Deploy:
    ARGS = MappingProxyType(
        dict(
            role=Shared.ROLE,
            session=Shared.SESSION,
            sm_client=Shared.SM_CLIENT,
            endpoint_name=Shared.ENDPOINT_NAME,
            model_package_group_name=Shared.MODEL_PACKAGE_GROUP_NAME,
            instance_type=Shared.INSTANCE_TYPE,
            instance_count=Shared.INSTANCE_COUNT,
            data_capture_destination_uri=f"s3://{Shared.S3_BUCKET_NAME}/{Shared.DATA_CAPTURE_S3_PREFIX}",
            serverless_inference_config={
                "memory_size_in_mb": 1024,
                "max_concurrency": 10,
            },
        )
    )


class DataDrift:
    ARGS = MappingProxyType(
        dict(
            s3_clent=Shared.S3_CLIENT,
            s3_bucket_name=Shared.S3_BUCKET_NAME,
            data_capture_prefix=Shared.DATA_CAPTURE_S3_PREFIX,
            baseline_data_s3_uri=Shared.TRAIN_DATA_S3_URI,  # use train data  as baseline for data drift monitoring
            evidently_api_token=Shared.EVIDENTLY_API_TOKEN,
            evidently_propject_id=Shared.EVIDENTLY_PROJECT_ID,
        )
    )


class ModelDrift:
    ARGS = MappingProxyType(
        dict(
            baseline_data_s3_uri=Shared.TEST_DATA_S3_URI,  # use test data as baseline for model drift monitoring,
            evidently_api_token=Shared.EVIDENTLY_API_TOKEN,
            evidently_propject_id=Shared.EVIDENTLY_PROJECT_ID,
        )
    )


class Clean:
    ARGS = MappingProxyType(
        dict(
            sm_client=Shared.SM_CLIENT,
            s3_client=Shared.S3_CLIENT,
            package_group_name=Shared.MODEL_PACKAGE_GROUP_NAME,
            pipeline_name=Shared.PIPELINE_NAME,
            bucket_name=Shared.S3_BUCKET_NAME,
            prefix=Shared.PROJECT_S3_PREFIX,
            endpoint_find_key=Shared.ENDPOINT_NAME,
        )
    )


class Inference:
    ARGS = MappingProxyType(
        dict(
            session=Shared.SESSION,
            input_path_prefix=f"{Shared.PROJECT_S3_PREFIX}/data/inference",
            output_path_prefix=f"{Shared.PROJECT_S3_PREFIX}/data/predictions",
        )
    )
