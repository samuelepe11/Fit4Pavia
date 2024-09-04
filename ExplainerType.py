# Import packages
from enum import Enum


# Class
class ExplainerType(Enum):
    GC = "GradCAM"
    GCref = "GradCAM_REF"
    HRC = "HiResCAM"
    HRCref = "HiResCAM_REF"
    VC = "VanillaCAM"
    LIME = "LIME"
    SHAP = "SHAP"
