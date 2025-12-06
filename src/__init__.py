"""
FaultFormer: Transformer-Based Bearing Fault Diagnosis
======================================================

A deep learning framework for cross-domain bearing fault diagnosis
using self-supervised pretraining and transfer learning.

Modules:
    - data: Dataset loaders and preprocessing pipelines
    - models: Neural network architectures and layers
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.models.faultformer import build_faultformer
from src.data.pipeline import BearingDatasetPipeline
