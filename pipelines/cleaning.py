from typing import Any

from steps import (
    delete_endpoints,
    delete_model_package_group,
    delete_project_prefix_contents,
    delete_sagemaker_pipeline,
)
from utils.helper import get_logger

logger = get_logger(__name__)


def cleanup_pipeline(
    sm_client: Any,
    s3_client: Any,
    package_group_name: str,
    pipeline_name: str,
    bucket_name: str,
    prefix: str,
    endpoint_find_key: str,
) -> None:
    logger.info("Clean-up pipeline has started...")

    delete_endpoints(
        sm_client=sm_client,
        endpoint_find_key=endpoint_find_key,
    )  # delete all endpoints and associated monitoring schedules

    delete_model_package_group(
        sm_client=sm_client,
        package_group_name=package_group_name,
    )

    delete_sagemaker_pipeline(
        sm_client=sm_client,
        pipeline_name=pipeline_name,
    )

    delete_project_prefix_contents(
        s3_client=s3_client,
        bucket_name=bucket_name,
        prefix=prefix,
    )

    logger.info("Clean-up pipeline finished sucessfully.")

    return None
