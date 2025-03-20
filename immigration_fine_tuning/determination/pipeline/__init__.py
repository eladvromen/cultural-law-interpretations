"""
Determination Extraction Pipeline package.

This package provides a modular pipeline for extracting legal determinations 
from immigration case documents using multiple extraction models.
"""

from .determination_pipeline import (
    DeterminationPipeline,
    PrecisionOptimizedPipeline,
    RecallOptimizedPipeline
)

from .model_loader import (
    ModelLoader,
    load_models
)

from .run_pipeline import (
    run_pipeline,
    evaluate_results
)

__all__ = [
    'DeterminationPipeline',
    'PrecisionOptimizedPipeline',
    'RecallOptimizedPipeline',
    'ModelLoader',
    'load_models',
    'run_pipeline',
    'evaluate_results'
]

__version__ = '0.1.0' 