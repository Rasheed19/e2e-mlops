import os

import boto3
import click
from dotenv import load_dotenv
from sagemaker.session import Session

from pipelines import (
    cleanup_pipeline,
    data_ingestion_pipeline,
    deployment_pipeline,
    inference_pipeline,
    training_pipeline,
)
from utils.constants import PipelineMode
from utils.helper import load_yaml_file

load_dotenv()


@click.command(
    help="""
Entry point for running pipelines.
"""
)
@click.option(
    "--pipeline",
    default=PipelineMode.TRAIN,
    help=f"""
    The kind of pipeline to run. Valid options are {", ".join([mode.value for mode in PipelineMode])}.
    """,
)
@click.option(
    "--force-upload",
    is_flag=True,
    default=False,
    help=f"""
    If this flag is given, train and test
    data will be forced to upload even if they
    already exist in the s3 bucket. This flag
    must be used when pipeline is set to {PipelineMode.TRAIN}.
    """,
)
@click.option(
    "--serverless",
    is_flag=True,
    default=False,
    help=f"""
    If this flag is given, model will be
    deployed to an endpoint in severless mode
    else real-time mode. This flag
    must be used when pipeline is set to {PipelineMode.DEPLOY}.
    """,
)
@click.option(
    "--inference-file-name",
    type=click.STRING,
    help=f"""
    Name of the inference data file in the
    s3 bucket prefix. Must be of the form "*.csv". This argument
    must be used when pipeline is set to {PipelineMode.INFERENCE}.
    """,
)
@click.option(
    "--deployed-endpoint-name",
    type=click.STRING,
    help=f"""
    Name of the deployed sagemaker endpoint. This argument
    must be used when pipeline is set to {PipelineMode.DEPLOY}.
    """,
)
def main(
    pipeline: str,
    inference_file_name: str,
    deployed_endpoint_name: str,
    force_upload: bool = False,
    serverless: bool = False,
) -> None:
    ROLE: str = os.getenv("ROLE")
    S3_BUCKET_NAME: str = os.getenv(
        "S3_BUCKET_NAME"
    )  # note the name must start with "sagemaker";
    # see https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-bucket.html
    infrastructure_config = load_yaml_file(path="./configs/infrastructure_config.yaml")
    model_config = load_yaml_file(path="./configs/model_config.yaml")
    inference_config = load_yaml_file(path="./configs/inference_config.yaml")

    session = Session(default_bucket=S3_BUCKET_NAME)

    sm_client = boto3.client("sagemaker")
    s3_client = boto3.client("s3")

    if pipeline == PipelineMode.TRAIN:
        # upload data to s3; will be skipped if data already exists unless force_upload is True
        data_ingestion_pipeline(
            test_size=model_config["test_size"],
            session=session,
            s3_bucket_name=S3_BUCKET_NAME,
            project_s3_prefix=model_config["project_s3_prefix"],
            train_key_prefix=model_config["train_key_prefix"],
            test_key_prefix=model_config["test_key_prefix"],
            force_upload=force_upload,
        )

        # train model
        hyperparameters = model_config["param_grid"]
        hyperparameters["train"] = (
            f"s3://{S3_BUCKET_NAME}/{model_config['train_key_prefix']}/train.csv"
        )
        training_pipeline(
            pipeline_name=model_config["pipeline_name"],
            session=session,
            role=ROLE,
            s3_bucket_name=S3_BUCKET_NAME,
            project_s3_prefix=model_config["project_s3_prefix"],
            train_data_uri=f"s3://{S3_BUCKET_NAME}/{model_config['train_key_prefix']}/train.csv",
            test_data_uri=f"s3://{S3_BUCKET_NAME}/{model_config['test_key_prefix']}/test.csv",
            hyperparameters=hyperparameters,
            framework_version=model_config["framework_version"],
            instance_type=infrastructure_config["instance_type"],
            instance_count=infrastructure_config["instance_count"],
            model_package_group_name=model_config["model_package_group_name"],
            model_approval_status=model_config["model_approval_status"],
            register_accuracy_threshold=model_config["register_accuracy_threshold"],
            output_path=f"s3://{S3_BUCKET_NAME}/{model_config['output_path_prefix']}",
            code_location=f"s3://{S3_BUCKET_NAME}/{model_config['code_location_prefix']}",
        )

    elif pipeline == PipelineMode.DEPLOY:
        deployment_pipeline(
            role=ROLE,
            session=session,
            sm_client=sm_client,
            endpoint_name=model_config["endpoint_name"],
            model_package_group_name=model_config["model_package_group_name"],
            instance_type=infrastructure_config["instance_type"],
            instance_count=infrastructure_config["instance_count"],
            serverless=serverless,
            serverless_inference_config=infrastructure_config[
                "serverless_inference_config"
            ],
        )

    elif pipeline == PipelineMode.CLEAN:
        cleanup_pipeline(
            sm_client=sm_client,
            s3_client=s3_client,
            package_group_name=model_config["model_package_group_name"],
            pipeline_name=model_config["pipeline_name"],
            bucket_name=S3_BUCKET_NAME,
            prefix=model_config["project_s3_prefix"],
            endpoint_find_key=model_config["endpoint_name"],
        )

    elif pipeline == PipelineMode.INFERENCE:
        inference_pipeline(
            session=session,
            deployed_endpoint_name=deployed_endpoint_name,
            input_path_prefix=inference_config["input_path_prefix"],
            inference_file_name=inference_file_name,
            output_path_prefix=inference_config["output_path_prefix"],
        )

    else:
        raise ValueError(
            f"--pipeline arg must take one of  {', '.join([mode.value for mode in PipelineMode])}; but {pipeline} was given."
        )

    return None


if __name__ == "__main__":
    main()
