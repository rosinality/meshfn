from enum import Enum


class ParallelMode(Enum):
    GLOBAL = "global"
    DATA = "data"
    MODEL = "model"
    PIPELINE = "pipe"
    TENSOR = "tensor"
    TENSOR_1D = "tensor_1d"
