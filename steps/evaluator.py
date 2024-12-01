from sagemaker.processing import FrameworkProcessor
from sagemaker.session import Session
from sagemaker.sklearn import SKLearn
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingStep,
    TrainingStep,
)


def model_evaluator(
    framework_version: str,
    instance_type: str,
    instance_count: int,
    sagemaker_session: Session,
    role: str,
    s3_bucket_name: str,
    project_s3_prefix: str,
    model_training_step: TrainingStep,
    test_data_uri: str,
    code_location: str,
) -> tuple[PropertyFile, ProcessingStep]:
    sklearn_processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        framework_version=framework_version,
        code_location=code_location,
        sagemaker_session=PipelineSession(
            default_bucket=sagemaker_session.default_bucket(),
            default_bucket_prefix=project_s3_prefix,
        ),
    )

    step_args = sklearn_processor.run(
        code="evaluate.py",
        source_dir="./scripts",
        inputs=[
            ProcessingInput(
                source=model_training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",  # this is just to create a destination path in the processing container
            ),
            ProcessingInput(
                source=test_data_uri,
                destination="/opt/ml/processing/test",  # this is just to create a destination path in the processing container
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=Join(
                    on="/",
                    values=[
                        "s3://{}".format(s3_bucket_name),
                        project_s3_prefix,
                        "evaluation_report",  # this name can be changed to suit your preference
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                    ],
                ),
            ),
        ],
        logs=False,
    )

    # A PropertyFile is for referencing outputs from a processing step, for instance to use in a condition step.
    # Find more info at https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
    evaluation_report = PropertyFile(
        name="evaluation-report-property-file",
        output_name="evaluation",
        path="evaluation.json",
    )

    model_evaluation_step = ProcessingStep(
        name="model-evaluation-step",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    return evaluation_report, model_evaluation_step
