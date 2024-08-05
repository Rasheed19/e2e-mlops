import boto3
import json


def predict_iris(
    request_body: dict[str, list[list[float]]],
    endpoint_name: str,
) -> list[int]:
    """
    example of a request body:
        request_body = {"Input": [[0.09178, 0.12, 4.05, 0.60], [0.09178, 0.560, 1.05, 2.0]]}
    """

    runtime_client = boto3.client("sagemaker-runtime")
    content_type = "application/json"
    data = json.loads(json.dumps(request_body))
    payload = json.dumps(data)

    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=payload
    )
    result = json.loads(response["Body"].read().decode())["Output"]

    return result
