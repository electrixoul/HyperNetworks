import torch
import sys
import os
from primary_net import PrimaryNetwork

def run_simple_gpu_test():
    """A simple test to verify GPU functionality with the HyperNetwork model"""
    
    # Print Python and PyTorch versions for debugging
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Running on {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Current Device: {torch.cuda.current_device()}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Running on CPU.")
    
    # Initialize network
    print("\nInitializing PrimaryNetwork...")
    net = PrimaryNetwork()
    net = net.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test with random input
    print("\nTesting forward pass with random input...")
    batch_size = 16
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # Measure inference time
    import time
    
    net.eval()
    start_time = time.time()
    with torch.no_grad():
        output = net(dummy_input)
    
    if device == torch.device("cuda"):
        torch.cuda.synchronize()  # Wait for GPU operations to complete
    
    inference_time = time.time() - start_time
    print(f"Forward pass completed successfully!")
    print(f"Output shape: {output.shape}")
    print(f"Inference time for batch of {batch_size}: {inference_time*1000:.2f} ms")
    print(f"Inference time per image: {inference_time*1000/batch_size:.2f} ms")
    
    # Memory usage after forward pass
    if device == torch.device("cuda"):
        print(f"\nCUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    run_simple_gpu_test()
