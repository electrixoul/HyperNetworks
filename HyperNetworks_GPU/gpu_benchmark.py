import torch
import time
import os
import numpy as np
import argparse
from primary_net import PrimaryNetwork

def benchmark_model():
    """Run a comprehensive benchmark of the HyperNetwork model on GPU"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Benchmark HyperNetwork GPU performance')
    parser.add_argument('--max_batch_size', type=int, default=256, 
                        help='Maximum batch size to test')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations to run for each batch size')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations')
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running benchmark on {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Running on CPU.")
        print(f"PyTorch Version: {torch.__version__}")
    
    # Initialize network
    print("\nInitializing PrimaryNetwork...")
    net = PrimaryNetwork()
    net = net.to(device)
    net.eval()  # Set to evaluation mode
    
    # Print model parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Batch sizes to test (powers of 2)
    batch_sizes = [2**i for i in range(0, int(np.log2(args.max_batch_size))+1)]
    
    # Run benchmarks
    print("\n" + "="*60)
    print(f"{'Batch Size':<15}{'Avg Latency (ms)':<20}{'Throughput (img/s)':<20}")
    print("="*60)
    
    for batch_size in batch_sizes:
        # Generate random input
        dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
        
        # Warmup
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = net(dummy_input)
        
        # Synchronize before timing
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
        
        # Measure performance
        latencies = []
        for _ in range(args.iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = net(dummy_input)
            
            if device == torch.device("cuda"):
                torch.cuda.synchronize()
            
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        throughput = batch_size * 1000 / avg_latency  # images/second
        
        print(f"{batch_size:<15}{avg_latency:<20.2f}{throughput:<20.2f}")
    
    print("="*60)
    
    # Memory usage
    if device == torch.device("cuda"):
        print(f"\nCUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    benchmark_model()
