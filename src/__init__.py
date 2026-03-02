"""
Track-MDP: Object Tracking with Reinforcement Learning

A modular framework for training RL agents to track moving objects in grid environments.
"""

__version__ = "1.0.0"
__author__ = "Track-MDP Team"

# Core functionality
from .core import GridEnvironment, TrackingLearner, TrackingEnv

# Training functionality
from .training import train, save_environment

# Evaluation functionality
from .evaluation import (
    evaluate_policy, 
    evaluate_policy_detailed, 
    print_evaluation_summary,
    visualize_policy_interactive,
    evaluate_and_visualize,
    find_latest_checkpoint
)

# Visualization functionality
from .visualization import TrackingRenderer

__all__ = [
    # Core
    'GridEnvironment',
    'TrackingLearner', 
    'TrackingEnv',
    # Training
    'train',
    'save_environment',
    # Evaluation
    'evaluate_policy',
    'evaluate_policy_detailed',
    'print_evaluation_summary',
    'visualize_policy_interactive',
    'evaluate_and_visualize',
    'find_latest_checkpoint',
    # Visualization
    'TrackingRenderer'
]
