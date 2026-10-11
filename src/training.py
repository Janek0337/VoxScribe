from src.config import DATA_DIR

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

import matplotlib.pyplot as plt

torch.random.manual_seed(0)