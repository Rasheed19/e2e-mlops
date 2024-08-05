from typing import Any
from sagemaker.session import Session

from steps import model_deployer
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
    serverless: bool,
    serverless_inference_config: dict,
) -> None:

    logger.info("Model deployment pipeline has started.")
    # might want to write the latest model description somewhere!
    latest_model_description = model_deployer(
        role=role,
        session=session,
        sm_client=sm_client,
        endpoint_name=endpoint_name,
        model_package_group_name=model_package_group_name,
        instance_type=instance_type,
        instance_count=instance_count,
        serverless=serverless,
        serverless_inference_config=serverless_inference_config,
    )

    logger.info("Model deployment pipeline finished successfuly.")

    return None
