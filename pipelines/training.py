from sagemaker.workflow.pipeline import Pipeline
from sagemaker.session import Session

from steps import (
    model_trainer,
    model_evaluator,
    model_registerer,
    model_registry_condition,
)
from utils.helper import get_logger

logger = get_logger(__name__)


def training_pipeline(
    pipeline_name: str,
    session: Session,
    role: str,
    s3_bucket_name: str,
    project_s3_prefix: str,
    output_path: str,
    code_location: str,
    train_data_uri: str,
    test_data_uri: str,
    hyperparameters: dict,
    framework_version: str,
    instance_type: str,
    instance_count: int,
    model_package_group_name: str,
    model_approval_status: str,
    register_accuracy_threshold: float,
) -> None:

    logger.info("Starting training pipeline...")

    sklearn_estimator, model_training_step = model_trainer(
        role=role,
        sagemaker_session=session,
        train_data_uri=train_data_uri,
        hyperparameters=hyperparameters,
        framework_version=framework_version,
        instance_type=instance_type,
        instance_count=instance_count,
        output_path=output_path,
        code_location=code_location,
    )

    evaluation_report, model_evaluation_step = model_evaluator(
        framework_version=framework_version,
        instance_type=instance_type,
        instance_count=instance_count,
        sagemaker_session=session,
        role=role,
        s3_bucket_name=s3_bucket_name,
        project_s3_prefix=project_s3_prefix,
        model_training_step=model_training_step,
        test_data_uri=test_data_uri,
        code_location=code_location,
    )

    model_registering_step = model_registerer(
        model_training_step=model_training_step,
        model_evaluation_step=model_evaluation_step,
        estimator=sklearn_estimator,
        model_package_group_name=model_package_group_name,
        model_approval_status=model_approval_status,
    )

    model_registering_condition_step = model_registry_condition(
        model_evaluation_step=model_evaluation_step,
        evaluation_report=evaluation_report,
        model_registering_step=model_registering_step,
        register_accuracy_threshold=register_accuracy_threshold,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        sagemaker_session=session,
        steps=[
            model_training_step,
            model_evaluation_step,
            model_registering_condition_step,
        ],  # Note that the order of execution is determined from each step's dependencies on other steps,
        # not on the order they are passed in below.
    )

    # Submit pipline
    pipeline.upsert(role_arn=role)
    pipeline.start()

    logger.info("Training pipeline submitted to sagemaker for execution.")

    return None
