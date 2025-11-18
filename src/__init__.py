"""
Module __init__.py pour le package src
"""

__version__ = "1.0.0"
__author__ = "Influencer-or-Observer Team"

# Imports des modules principaux
from . import data_loader
from . import preprocessing
from . import feature_engineering
from . import models
from . import evaluation
from . import utils

__all__ = [
    "data_loader",
    "preprocessing",
    "feature_engineering",
    "models",
    "evaluation",
    "utils"
]
