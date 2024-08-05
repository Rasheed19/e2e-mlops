import pandas as pd
from pathlib import Path
from shiny.types import ImgData
from shiny import ui

from steps import predict_iris
from utils.helper import get_iris_dictionary


def get_merged_prediction_data(
    prepared_data: pd.DataFrame, endpoint_name: str
) -> pd.DataFrame:

    request_body = {"Input": prepared_data.values.tolist()}
    predictions = predict_iris(
        request_body=request_body,
        endpoint_name=endpoint_name,
    )

    prepared_data["predicted iris"] = predictions
    prepared_data["predicted iris"] = prepared_data["predicted iris"].map(
        get_iris_dictionary()
    )

    return prepared_data


def load_image(image_path: Path) -> ImgData:

    img: ImgData = {
        "src": image_path,
        "height": "200px",
        "width": "350px",
    }
    return img


def github_text() -> ui.Tag:
    return ui.markdown(
        """
        _The source code for this dashboard can be found in 
        this [link](https://github.com/Rasheed19/e2e-mlops).
        This project makes use of the [Amazon Sagemaker](https://github.com/aws/sagemaker-python-sdk) machine learning 
        oprations (MLOps) structure to develop both the model steps
        and pipelines. The dashboard is built using the [shiny](https://shiny.posit.co/py/) Python
        framework._
        """
    )


def about_prediction_service() -> ui.Tag:
    return ui.markdown(
        """This prediction service is obtained from 
            training the gradient boost model on the Iris
            dataset downloaded from the scikit-learn library.
            The data contains the lengths and widths of the 
            sepal and petal of three irises namely setosa, versicolor,
            and virginica. More information about this dataset
            can be found [here](https://en.wikipedia.org/wiki/Iris_flower_data_set).
            The model is served using the Amazon Sagemaker endpoint.
            """
    )
