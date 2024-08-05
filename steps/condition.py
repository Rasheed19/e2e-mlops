from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel


def model_registry_condition(
    model_evaluation_step: ProcessingStep,
    evaluation_report: PropertyFile,
    model_registering_step: RegisterModel,
    register_accuracy_threshold: float,
) -> ConditionStep:

    register_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=model_evaluation_step.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value",
        ),
        right=register_accuracy_threshold,
    )

    model_registering_condition_step = ConditionStep(
        name="model-registry-condition-step",
        conditions=[register_condition],
        if_steps=[model_registering_step],
        else_steps=[],
    )

    return model_registering_condition_step
