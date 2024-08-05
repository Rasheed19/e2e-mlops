from steps.loader import data_loader
from steps.splitter import train_data_splitter
from steps.uploader import data_uploader
from steps.trainer import model_trainer
from steps.evaluator import model_evaluator
from steps.registerer import model_registerer
from steps.condition import model_registry_condition
from steps.cleaner import (
    delete_model_package_group,
    delete_sagemaker_pipeline,
    delete_project_prefix_contents,
    delete_endpoints,
)
from steps.deployer import model_deployer
from steps.predictor import predict_iris
