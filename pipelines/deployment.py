from typing import Any

from sagemaker.session import Session

from steps import deploy_model, fetch_model
from utils.helper import get_logger

logger = get_logger(__name__)


def deployment_pipeline(
    role: str,
    session: Session,
    sm_client: Any,
    endpoint_name: str,
    model_package_group_name: str,
    instance_type: str,
    instance_count: int,
    data_capture_destination_uri: str,
    serverless: bool,
    serverless_inference_config: dict,
) -> None:
    logger.info("Model deployment pipeline has started.")

    latest_model = fetch_model(
        role=role,
        session=session,
        sm_client=sm_client,
        model_package_group_name=model_package_group_name,
    )

    deploy_model(
        latest_model=latest_model,
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        instance_count=instance_count,
        data_capture_destination_uri=data_capture_destination_uri,
        serverless=serverless,
        serverless_inference_config=serverless_inference_config,
    )

    logger.info("Model deployment pipeline finished successfuly.")

    return None
