from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.sklearn import SKLearn
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn


def model_trainer(
    role: str,
    sagemaker_session: Session,
    train_data_uri: str,
    hyperparameters: dict,
    framework_version: str,
    instance_type: str,
    instance_count: int,
    output_path: str,
    code_location: str,
) -> tuple[SKLearn, TrainingStep]:

    sklearn_estimator = SKLearn(
        role=role,
        sagemaker_session=sagemaker_session,
        source_dir="./scripts",
        entry_point="train.py",
        framework_version=framework_version,
        py_version="py3",
        instance_type=instance_type,
        instance_count=instance_count,
        hyperparameters=hyperparameters,
        output_path=output_path,
        code_location=code_location,
    )

    model_training_step = TrainingStep(
        name="model-training-step",
        estimator=sklearn_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=train_data_uri,
                content_type="text/csv",
            ),
        },
    )

    return sklearn_estimator, model_training_step
