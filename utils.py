"""
utils.py — Утилиты для воспроизводимости, логгирования и визуализации
"""
import torch
import numpy as np
import random
import logging

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
