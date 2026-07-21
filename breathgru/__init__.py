"""
BreathGRU

A semi-supervised Bidirectional GRU framework for speech-breath segmentation.
"""

__version__ = "1.0.0"
__author__ = "Sania Fatima Sayed"

from .model import BreathGRUSegmenter
from .inference import BreathGRU
from .features import extract_features

__all__ = [
    "BreathGRUSegmenter",
    "BreathGRU",
    "extract_features",
]
