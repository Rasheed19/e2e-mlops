import time
from typing import Any


def delete_model_package_group(sm_client: Any, package_group_name: str):
    try:
        model_versions = sm_client.list_model_packages(
            ModelPackageGroupName=package_group_name
        )

    except Exception as e:
        print("{} \n".format(e))

        return None

    for model_version in model_versions["ModelPackageSummaryList"]:
        try:
            sm_client.delete_model_package(
                ModelPackageName=model_version["ModelPackageArn"]
            )
        except Exception as e:
            print("{} \n".format(e))
        time.sleep(0.5)  # Ensure requests aren't throttled

    try:
        sm_client.delete_model_package_group(ModelPackageGroupName=package_group_name)
        print("{} model package group deleted".format(package_group_name))
    except Exception as e:
        print("{} \n".format(e))

    return None


def delete_sagemaker_pipeline(sm_client: Any, pipeline_name: str):
    try:
        sm_client.delete_pipeline(
            PipelineName=pipeline_name,
        )
        print("{} pipeline deleted".format(pipeline_name))
    except Exception as e:
        print("{} \n".format(e))

    return None


def delete_project_prefix_contents(
    s3_client: Any, bucket_name: str, prefix: str
) -> None:
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    try:
        for object in response["Contents"]:
            print("Deleting", object["Key"])
            s3_client.delete_object(Bucket=bucket_name, Key=object["Key"])

    except Exception as e:
        print("{} \n".format(e))

    return None


def delete_monitoring_schedule(sm_client: Any, schedule_name: str) -> None:
    try:
        sm_client.delete_monitoring_schedule(MonitoringScheduleName=schedule_name)
        print("{} monitoring schedule deleted".format(schedule_name))
    except Exception as e:
        print("{} \n".format(e))

    return None


def delete_endpoints(sm_client: Any, endpoint_find_key: str) -> None:
    try:
        response = sm_client.list_endpoints(
            NameContains=endpoint_find_key,
            # StatusEquals="OutOfService"
            # | "Creating"
            # | "Updating"
            # | "SystemUpdating"
            # | "RollingBack"
            # | "InService"
            # | "Deleting"
            # | "Failed",
        )

        for endpoint in response["Endpoints"]:
            # list monitoring schedules
            # schedules = sm_client.list_monitoring_schedules(
            #     EndpointName=endpoint["EndpointName"]
            # )

            # for schedule in schedules["MonitoringScheduleSummaries"]:
            #     delete_monitoring_schedule(
            #         sm_client, schedule["MonitoringScheduleName"]
            #     )

            print("Deleting", endpoint["EndpointName"])
            sm_client.delete_endpoint(EndpointName=endpoint["EndpointName"])

    except Exception as e:
        print("{} \n".format(e))

    return None
