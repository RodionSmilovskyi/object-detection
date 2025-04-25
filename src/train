#!/usr/bin/env python
#pylint: skip-file
import sys
import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
    else:
        print("No CUDA-enabled GPU is available.")

    print("SUCCESS")
    sys.exit(0)