from typing import Any
import json
from datetime import datetime as dt

from sagemaker import ModelPackage
from sagemaker.session import Session
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig

from utils.helper import get_logger

logger = get_logger(__name__)


def model_deployer(
    role: str,
    session: Session,
    sm_client: Any,
    endpoint_name: str,
    model_package_group_name: str,
    instance_type: str,
    instance_count: int,
    serverless: bool,
    serverless_inference_config: dict,
) -> dict:

    logger.info("Reading all registered models...")

    registered_models = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name
    )
    latest_version_number = registered_models["ModelPackageSummaryList"][0][
        "ModelPackageVersion"
    ]
    latest_model_package_arn = registered_models["ModelPackageSummaryList"][0][
        "ModelPackageArn"
    ]
    latest_model_status = registered_models["ModelPackageSummaryList"][0][
        "ModelApprovalStatus"
    ]
    latest_model_description = sm_client.describe_model_package(
        ModelPackageName=latest_model_package_arn
    )

    latest_model_hyperparameters = json.loads(
        latest_model_description["ModelCard"]["ModelCardContent"]
    )["training_details"]["training_job_details"]["hyper_parameters"]
    hyperparameters_needed = {}
    for param in latest_model_hyperparameters:
        name = param["name"]
        if name in [
            "sagemaker_submit_directory",
            "sagemaker_region",
            "sagemaker_program",
        ]:
            hyperparameters_needed[name] = (
                param["value"].replace("'", "").replace('"', "")
            )
    model_data = latest_model_description["InferenceSpecification"]["Containers"][0][
        "ModelDataUrl"
    ]
    sagemaker_default_invocations_accept = latest_model_description[
        "InferenceSpecification"
    ]["SupportedContentTypes"]

    model = ModelPackage(
        role=role,
        model_data=model_data,
        model_package_arn=latest_model_package_arn,
        sagemaker_session=session,
        env={
            "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": sagemaker_default_invocations_accept[
                0
            ],
            "SAGEMAKER_USE_NGINX": "True",
            "SAGEMAKER_WORKER_CLASS_TYPE": "gevent",
            "SAGEMAKER_KEEP_ALIVE_SEC": "60",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
            "SAGEMAKER_PROGRAM": hyperparameters_needed[
                "sagemaker_program"
            ],  # name of the entrypoint to the model trainer script
            "SAGEMAKER_REGION": hyperparameters_needed["sagemaker_region"],
            "SAGEMAKER_SUBMIT_DIRECTORY": hyperparameters_needed[
                "sagemaker_submit_directory"
            ],
        },
    )
    endpoint_name = f"{endpoint_name}-{dt.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    if serverless:
        logger.info(
            "Deploying model in serverless mode with the following configuration:\n"
            f"{serverless_inference_config}"
        )
        serverless_inference_config = ServerlessInferenceConfig(
            **serverless_inference_config
        )
        model.deploy(
            endpoint_name=endpoint_name,
            serverless_inference_config=serverless_inference_config,
            wait=False,
        )
    else:
        logger.info(
            f"Deploying model in real-time mode with intance type '{instance_type}' and "
            f"initial instance count {instance_count}."
        )
        model.deploy(
            endpoint_name=endpoint_name,
            initial_instance_count=instance_count,
            instance_type=instance_type,
            wait=False,
        )

    logger.info(
        f"Latest model with version '{latest_version_number}' and "
        f"status '{latest_model_status}' has been successfully "
        "submitted for deployment. Check the Endpoint section "
        "of the amazon sagemaker studio for progress and details."
    )

    return latest_model_description
