"""
SPARC: Multimodal Learning Framework

A framework for multimodal representation learning with support for various 
datasets, model architectures, and evaluation metrics.
"""

__version__ = "0.1.0"
__author__ = "SPARC Team"

# Core modules
from . import model
from . import evaluation  
from . import feature_extract
from . import utils
from . import datasets
from . import kernels
from . import loss
from . import post_analysis

# Key classes and functions for easy access
from .model import *
from .utils import seed_worker, set_seed
from .datasets import HDF5FeatureDataset
from .loss import autoencoder_loss, normalized_mean_squared_error, normalized_L1_loss

__all__ = [
    "model",
    "evaluation", 
    "feature_extract",
    "utils",
    "datasets", 
    "kernels",
    "loss",
    "post_analysis",
    "seed_worker",
    "set_seed",
    "HDF5FeatureDataset",
    "autoencoder_loss", 
    "normalized_mean_squared_error",
    "normalized_L1_loss",
] 