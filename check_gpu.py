"""
GPU Availability Check Script for HyperNetworks

This script checks if CUDA is available and displays information about the GPU
environment for the HyperNetworks training. It is designed to be run in the
sae_eeg conda environment.

Usage:
    conda run -n sae_eeg python check_gpu.py
"""

import torch
import sys

def check_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Please check your PyTorch installation.")
        sys.exit(1)
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    print("\nCUDA current device:", torch.cuda.current_device())
    print("CUDA default generator seed:", torch.cuda.initial_seed())
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
    print("cuDNN benchmark:", torch.backends.cudnn.benchmark)
    
    # Create a simple tensor and move it to GPU to verify operations work
    try:
        x = torch.rand(5, 5).cuda()
        y = torch.rand(5, 5).cuda()
        z = x @ y  # Matrix multiplication
        print("\nGPU tensor operations test: SUCCESS")
    except Exception as e:
        print(f"\nGPU tensor operations test: FAILED")
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== HyperNetworks GPU Environment Check ===\n")
    check_cuda()
    print("\n=== Check Complete ===")
