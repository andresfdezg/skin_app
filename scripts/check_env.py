"""Check that PyTorch sees the GPU."""
import sys

import torch

print(f"Python  {sys.version.split()[0]}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     {torch.cuda.get_device_name(0)}")
