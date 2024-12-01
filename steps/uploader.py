from sagemaker.s3 import S3Downloader
from sagemaker.session import Session

from utils.helper import get_logger

logger = get_logger(__name__)


def data_uploader(
    session: Session,
    s3_bucket_name: str,
    project_s3_prefix: str,
    path_to_train_data: str,
    path_to_test_data: str,
    train_key_prefix: str,
    test_key_prefix: str,
    force_upload: bool,
) -> None:
    s3_data_dir_contents = S3Downloader.list(
        s3_uri=f"s3://{s3_bucket_name}/{project_s3_prefix}/data"
    )
    s3_data_dir_contents = set(s3_data_dir_contents)
    issubset = {
        f"s3://{s3_bucket_name}/{train_key_prefix}/{path_to_train_data.rstrip('/')[0]}",
        f"s3://{s3_bucket_name}/{test_key_prefix}/{path_to_test_data.rstrip('/')[0]}",
    }.issubset(s3_data_dir_contents)

    if issubset:
        if force_upload:
            logger.warning(
                "Train and test data already exist in s3 bucket but forcing upload "
                "as 'force_upload' is set to True."
            )
            pass

        else:
            logger.info(
                "Skipping data upload as train and test data already exist in s3 bucket."
            )
            return None

    train_data_uri = session.upload_data(
        path=path_to_train_data,
        key_prefix=train_key_prefix,
    )
    test_data_uri = session.upload_data(
        path=path_to_test_data,
        key_prefix=test_key_prefix,
    )

    print("Test data set uploaded to ", test_data_uri)
    print("Train data set uploaded to ", train_data_uri)

    return None
