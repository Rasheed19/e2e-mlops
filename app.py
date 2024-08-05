from shiny import App, Inputs, Outputs, Session, render, ui, reactive
import pandas as pd
from shiny.types import FileInfo
from pathlib import Path
from datetime import datetime as dt

from utils.helper import prepare_single_prediction_data, prepare_batch_prediction_data
from apps.server import (
    get_merged_prediction_data,
    load_image,
    github_text,
    about_prediction_service,
)

# define your endpoint name here
ENPOINT_NAME: str = "iris-prediction-endpoint-2024-08-05-18-07-08"

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            "prediction_type",
            "Select prediction type",
            choices=["Single-sample prediction", "Batch prediction"],
            multiple=False,
        ),
        ui.output_ui("text_input_upload_ui"),
        ui.input_action_button(
            id="get_prediction_btn",
            label="Get prediction",
            icon=None,
            class_="btn btn-primary",
        ),
        ui.card(ui.card_header("About"), github_text()),
        width=550,
    ),
    ui.output_ui("prediction_table"),
    title="Iris prediction dashboard",
    fillable=True,
)


def server(input: Inputs, output: Outputs, session: Session):

    @reactive.calc
    def parsed_file():
        file: list[FileInfo] | None = input.batch_csv()
        if file is None:
            return pd.DataFrame()
        return pd.read_csv(file[0]["datapath"])

    @reactive.calc
    @reactive.event(input.get_prediction_btn)
    def get_prepared_data():

        if input.prediction_type() == "Single-sample prediction":
            prepared_data = prepare_single_prediction_data(
                sepal_length=input.sepal_length(),
                sepal_width=input.sepal_width(),
                petal_length=input.petal_length(),
                petal_width=input.petal_width(),
            )

        elif input.prediction_type() == "Batch prediction":
            prepared_data = prepare_batch_prediction_data(uploaded_data=parsed_file())

        return prepared_data

    @render.data_frame
    def summary():
        df = parsed_file()

        if df.empty:
            return pd.DataFrame()

        row_count = df.shape[0]
        column_count = df.shape[1]
        names = df.columns.tolist()
        column_names = ", ".join(str(name) for name in names)

        info_df = pd.DataFrame(
            {
                "Row Count": [row_count],
                "Column Count": [column_count],
                "Column Names": [column_names],
            }
        )

        return render.DataGrid(data=info_df.loc[:, input.stats()])

    @render.ui
    def text_input_upload_ui():

        if input.prediction_type() == "Single-sample prediction":
            return (
                ui.card(
                    ui.row(
                        ui.column(
                            6,
                            ui.input_text(
                                id="sepal_length",
                                label="Sepal length",
                                value="",
                                width=None,
                                placeholder="Enter the sepal length in cm",
                                autocomplete="off",
                                spellcheck=None,
                            ),
                            ui.input_text(
                                id="sepal_width",
                                label="Sepal width",
                                value="",
                                width=None,
                                placeholder="Enter the sepal width in cm",
                                autocomplete="off",
                                spellcheck=None,
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_text(
                                id="petal_length",
                                label="Petal length",
                                value="",
                                width=None,
                                placeholder="Enter the petal length in cm",
                                autocomplete="off",
                                spellcheck=None,
                            ),
                            ui.input_text(
                                id="petal_width",
                                label="Petal width",
                                value="",
                                width=None,
                                placeholder="Enter the petal width in cm",
                                autocomplete="off",
                                spellcheck=None,
                            ),
                        ),
                    ),
                ),
            )
        elif input.prediction_type() == "Batch prediction":
            return (
                ui.card(
                    ui.input_file(
                        "batch_csv",
                        "Upload a CSV file",
                        accept=[".csv"],
                        multiple=False,
                    ),
                    ui.input_checkbox_group(
                        "stats",
                        "Summary Stats",
                        choices=["Row Count", "Column Count", "Column Names"],
                        selected=["Row Count", "Column Count", "Column Names"],
                    ),
                    ui.output_data_frame("summary"),
                ),
            )

        return ui.markdown("Select a prediction type to proceed.")

    @render.ui
    def prediction_table():
        prepared_data = get_prepared_data()

        if isinstance(prepared_data, pd.DataFrame):
            with ui.Progress(min=0, max=1) as p:
                p.set(message="Getting predictions", detail="This may take a while...")
                p.inc(amount=0.5, message="Predicting...")
                merged_prediction_data = get_merged_prediction_data(
                    prepared_data=prepared_data,
                    endpoint_name=ENPOINT_NAME,
                )
                p.inc(amount=0.9, message="Predicting...")
                p.close()

            @render.data_frame
            def _render_table():
                return render.DataGrid(data=merged_prediction_data, width=1000)

            @render.image
            def _iris_virginica():
                return load_image(Path().resolve() / "assets" / "iris_virginica.jpg")

            @render.image
            def _iris_versicolor():
                return load_image(Path().resolve() / "assets" / "iris_versicolor.jpg")

            @render.image
            def _iris_setosa():
                return load_image(Path().resolve() / "assets" / "iris_setosa.jpg")

            @render.download(
                filename=lambda: f"iris-prediction-{dt.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv",
                media_type="text/csv",
            )
            def _download_handler():
                return (prepared_data.to_csv().encode("utf-8"),)

            return ui.tags.div(
                ui.row(
                    ui.column(
                        10,
                        ui.card(
                            ui.card_header(
                                "Uploaded/input data with predicted iris type"
                            ),
                            ui.output_data_frame("_render_table"),
                        ),
                    ),
                    ui.column(
                        2,
                        ui.download_button(
                            "_download_handler",
                            "Download data as CSV",
                            class_="btn btn-primary",
                        ),
                    ),
                    ui.card(
                        ui.card_header("About prediction service"),
                        about_prediction_service(),
                        ui.row(
                            *[
                                ui.column(
                                    4,
                                    ui.card(
                                        ui.card_header(h),
                                        ui.output_image(
                                            id=id_,
                                            height="200px",
                                        ),
                                    ),
                                )
                                for h, id_ in zip(
                                    [
                                        "Iris Virginica",
                                        "Iris Versicolor",
                                        "Iris Setosa",
                                    ],
                                    [
                                        "_iris_virginica",
                                        "_iris_versicolor",
                                        "_iris_setosa",
                                    ],
                                )
                            ]
                        ),
                    ),
                )
            )
        ui.notification_show(
            prepared_data,
            type="error",
            duration=5,
        )


app = App(app_ui, server)
