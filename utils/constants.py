from enum import StrEnum, auto


class PipelineMode(StrEnum):
    TRAIN = auto()
    DEPLOY = auto()
    CLEAN = auto()
    INFERENCE = auto()
