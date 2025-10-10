"""
Semi-automated Data Labeling System
Package initialization module
"""

__version__ = "1.0.0"
__author__ = "Semi-automated Labeling Team"
__description__ = "A comprehensive system for semi-automated data labeling with multi-layer validation"

# Import main functions for easy access
from .utils import load_data, save_results, setup_logging
from .llm_generator import generate_labels
from .rule_checker import check_rules
from .model_validator import validate_labels
from .human_review import review_labels

__all__ = [
    'load_data',
    'save_results', 
    'setup_logging',
    'generate_labels',
    'check_rules',
    'validate_labels',
    'review_labels'
]