from sagemaker.session import Session

from steps import data_loader, train_data_splitter, data_uploader
from utils.helper import get_logger

logger = get_logger(__name__)


def data_ingestion_pipeline(
    test_size: int,
    session: Session,
    s3_bucket_name: str,
    project_s3_prefix: str,
    train_key_prefix: str,
    test_key_prefix: str,
    force_upload: bool,
) -> None:

    logger.info("Data ingestion pipeline has started.")

    dataset, target_name = data_loader()

    path_to_train_data, path_to_test_data = train_data_splitter(
        dataset=dataset, target_name=target_name, test_size=test_size
    )

    data_uploader(
        session=session,
        s3_bucket_name=s3_bucket_name,
        project_s3_prefix=project_s3_prefix,
        path_to_train_data=path_to_train_data,
        path_to_test_data=path_to_test_data,
        train_key_prefix=train_key_prefix,
        test_key_prefix=test_key_prefix,
        force_upload=force_upload,
    )

    logger.info("Data ingestion pipeline finished successfully.")

    return None
