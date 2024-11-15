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
    torch.cuda.manual_seed(seed)  # Torch (GPU)
    torch.cuda.manual_seed_all(seed)  # Для всех GPU
    torch.backends.cudnn.deterministic = True  # Для воспроизводимости в CuDNN
    torch.backends.cudnn.benchmark = False  # Отключить эвристики для скорости
    