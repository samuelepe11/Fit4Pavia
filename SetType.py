# Import packages
from enum import Enum


# Class
class SetType(Enum):
    TRAINING = "train"
    VALIDATION = "validation"
    TEST = "test"
    EXT_TEST = "external_test"
