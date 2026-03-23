import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Фиксирует seed для всех основных источников случайности, чтобы обеспечить воспроизводимость.

    Parameters:
        seed (int): Значение seed.
    """
    random.seed(seed)  # Python
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # Torch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    