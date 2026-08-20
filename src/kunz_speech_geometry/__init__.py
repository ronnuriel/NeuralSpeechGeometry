"""Tools for exploratory neural-geometry analysis of the Kunz speech dataset."""

from .schema import NeuralDataset
from .synthetic import make_synthetic_dataset

__all__ = ["NeuralDataset", "make_synthetic_dataset"]
__version__ = "0.1.0"
