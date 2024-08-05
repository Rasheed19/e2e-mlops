from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.functions import Join
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn import SKLearn


def model_registerer(
    model_training_step: TrainingStep,
    model_evaluation_step: ProcessingStep,
    estimator: SKLearn,
    model_package_group_name: str,
    model_approval_status: str,
) -> RegisterModel:

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    model_evaluation_step.arguments["ProcessingOutputConfig"][
                        "Outputs"
                    ][0]["S3Output"]["S3Uri"],
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    model_registering_step = RegisterModel(
        name="model-registering-step",
        estimator=estimator,
        model_data=model_training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/x-npy"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge", "ml.m5.large"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    return model_registering_step
