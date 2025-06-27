import click
from dotenv import load_dotenv

from pipelines import (
    cleanup_pipeline,
    data_ingestion_pipeline,
    deployment_pipeline,
    evidently_data_drift_monitoring_pipeline,
    evidently_model_drift_monitoring_pipeline,
    inference_pipeline,
    training_pipeline,
)
from utils.constants import (
    Clean,
    DataDrift,
    Deploy,
    Inference,
    Ingest,
    ModelDrift,
    Train,
)

load_dotenv()


@click.group()
def cli() -> None:
    """A cli for running various pipelines in the e2e-mlops project.

    Author: Rasheed Ibraheem.
    """
    pass


@cli.command()
def clean() -> None:
    """
    Delete sagemaker resources and artifacts.
    """

    if click.confirm(
        """Do you want to continue?  The pipeline run inputs and outputs, models
        in the model registry, and deployed endpoints will be DELETED. Note that this pipeline will
        recursively delete all the contents of the S3 bucket you created during
        the infrastructure set up.""",
        abort=True,
    ):
        click.echo("Deleting sagemaker resources and artifacts...")
        cleanup_pipeline(**Clean.ARGS)

    return None


@cli.command()
@click.option(
    "--force-upload",
    is_flag=True,
    default=False,
    help="""
    If this flag is given, train, test and baseline monitoring
    data will be forced to upload even if they
    already exist in the s3 bucket.
    """,
)
def train(force_upload: bool) -> None:
    """
    Ingest data, train, evaluate and register model.
    """

    # upload data to s3; will be skipped if data already exists unless force_upload is True
    run_args = {}
    run_args["force_upload"] = force_upload
    run_args.update(Ingest.ARGS)

    data_ingestion_pipeline(**run_args)
    training_pipeline(**Train.ARGS)

    return None


@cli.command()
@click.option(
    "--serverless",
    is_flag=True,
    default=False,
    help="""
    If this flag is given, model will be
    deployed to an endpoint in severless mode
    else real-time mode.
    """,
)
def deploy(serverless: bool) -> None:
    """
    Deploy the latest registered model in the model registry.
    """
    run_args = {}
    run_args["serverless"] = serverless
    run_args.update(Deploy.ARGS)

    deployment_pipeline(**run_args)

    return None


@cli.command()
@click.argument(
    "endpoint-name",
    type=click.STRING,
)
def datadrift(endpoint_name: str) -> None:
    """
    Run data drift monitoring pipeline against the ENDPOINT_NAME.

    ENDPOINT_NAME: name of the deployed sagemaker endpoint.
    """

    run_args = {}
    run_args["endpoint_name"] = endpoint_name
    run_args.update(DataDrift.ARGS)

    evidently_data_drift_monitoring_pipeline(**run_args)

    return None


@cli.command()
@click.argument(
    "endpoint-name",
    type=click.STRING,
)
@click.argument(
    "current-data-s3-uri",
    type=click.STRING,
)
def modeldrift(endpoint_name: str, current_data_s3_uri: str) -> None:
    """
    Run model drift monitoring pipeline against the ENDPOINT_NAME.

    ENDPOINT_NAME: name of the deployed sagemaker endpoint.

    CURRENT_DATA_S3_URI: the complete s3 uri to the  current data that contains predictions from the
    deployed endpoint and the ground truth.
    """

    run_args = {}
    run_args["endpoint_name"] = endpoint_name
    run_args["current_data_s3_uri"] = current_data_s3_uri
    run_args.update(ModelDrift.ARGS)

    evidently_model_drift_monitoring_pipeline(**run_args)

    return None


@cli.command()
@click.argument(
    "endpoint-name",
    type=click.STRING,
)
@click.argument(
    "inference-file-name",
    type=click.STRING,
)
def inference(endpoint_name: str, inference_file_name: str) -> None:
    """
    Run model drift monitoring pipeline against the ENDPOINT_NAME.

    ENDPOINT_NAME: name of the deployed sagemaker endpoint.

    INFERENCE_FILE_NAME: name of the inference data file in the
    s3 bucket prefix. Must be of the form "*.csv".
    """

    run_args = {}
    run_args["deployed_endpoint_name"] = endpoint_name
    run_args["inference_file_name"] = inference_file_name
    run_args.update(Inference.ARGS)

    inference_pipeline(**run_args)

    return None
