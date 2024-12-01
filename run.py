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
    DataField,
    Deploy,
    Inference,
    Ingest,
    ModelDrift,
    PipelineMode,
    Train,
)

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
    must be used when pipeline is set to {PipelineMode.DEPLOY} or {PipelineMode.DATADRIFT}.
    """,
)
@click.option(
    "--current-data-s3-uri",
    type=click.STRING,
    help=f"""
    The complete s3 uri to the  current data that contains predictions from the
    deployed endpoint and the ground truth. Note that the column that contains
    predictions must be named {DataField.PREDICTION} and the column with the
    ground truth must be named {DataField.TARGET} . Other columns must contain
    the features with appropriate names (in order): {DataField.FEATURES}. This argument
    must be used when pipeline is set to {PipelineMode.MODELDRIFT}.
    """,
)
@click.option(
    "--force-upload",
    is_flag=True,
    default=False,
    help=f"""
    If this flag is given, train, test and baseline monitoring
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
def main(
    pipeline: str,
    inference_file_name: str,
    deployed_endpoint_name: str,
    current_data_s3_uri: str,
    force_upload: bool = False,
    serverless: bool = False,
) -> None:
    run_args = {}

    if pipeline == PipelineMode.TRAIN:
        # upload data to s3; will be skipped if data already exists unless force_upload is True
        run_args["force_upload"] = force_upload
        run_args.update(Ingest.ARGS)
        data_ingestion_pipeline(**run_args)
        training_pipeline(**Train.ARGS)

    elif pipeline == PipelineMode.DEPLOY:
        run_args["serverless"] = serverless
        run_args.update(Deploy.ARGS)
        deployment_pipeline(**run_args)

    elif pipeline == PipelineMode.DATADRIFT:
        run_args["endpoint_name"] = deployed_endpoint_name
        run_args.update(DataDrift.ARGS)
        evidently_data_drift_monitoring_pipeline(**run_args)

    elif pipeline == PipelineMode.MODELDRIFT:
        run_args["endpoint_name"] = deployed_endpoint_name
        run_args["current_data_s3_uri"] = current_data_s3_uri
        run_args.update(ModelDrift.ARGS)
        evidently_model_drift_monitoring_pipeline(**run_args)

    elif pipeline == PipelineMode.CLEAN:
        cleanup_pipeline(**Clean.ARGS)

    elif pipeline == PipelineMode.INFERENCE:
        run_args["deployed_endpoint_name"] = deployed_endpoint_name
        run_args["inference_file_name"] = inference_file_name
        run_args.update(Inference.ARGS)
        inference_pipeline(**run_args)

    else:
        raise ValueError(
            f"--pipeline arg must take one of  {', '.join([mode.value for mode in PipelineMode])}; but {pipeline} was given."
        )

    return None


if __name__ == "__main__":
    main()
