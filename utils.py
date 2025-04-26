"""
utils.py — Utilities for reproducibility, logging, and visualization
"""
import random
import numpy as np
import torch
import logging

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging():
    """
    Configure logging format for main.py.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
