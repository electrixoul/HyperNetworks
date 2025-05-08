import torch
import torchvision
import torchvision.transforms as transforms
import time
import argparse
import os
import numpy as np
import warnings

# Ignore specific warnings from torchvision
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

from primary_net import PrimaryNetwork

def test_gpu_performance():
    """Test GPU performance with the HyperNetwork model"""
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"CUDA is available. Running on {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Current Device: {torch.cuda.current_device()}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test HyperNetworks GPU performance')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for testing')
    parser.add_argument('--model_path', type=str, default='./hypernetworks_cifar_gpu.pth', 
                        help='path to the model checkpoint')
    args = parser.parse_args()
    
    # Data transformations
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Test dataset
    print("Loading CIFAR-10 test dataset...")
    try:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform_test)
    except AttributeError as e:
        # Handle compatibility issues with newer PyTorch versions
        print(f"Compatibility warning: {e}")
        print("Using alternative method to load CIFAR-10...")
        
        # Alternative loading method - completely standalone implementation
        from PIL import Image
        import pickle
        from typing import Any, Callable, Optional, Tuple
        import numpy as np
        
        class CIFAR10Compatible:
            def __init__(
                self,
                root: str,
                train: bool = True,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                download: bool = False,
            ) -> None:
                self.root = root
                self.transform = transform
                self.target_transform = target_transform
                self.train = train
                self.base_folder = "cifar-10-batches-py"
                
                if download:
                    from torchvision.datasets.utils import download_and_extract_archive
                    import os
                    
                    if not os.path.exists(os.path.join(root, self.base_folder)):
                        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
                        download_and_extract_archive(url, root, filename="cifar-10-python.tar.gz")
                
                if self.train:
                    downloaded_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
                else:
                    downloaded_list = ["test_batch"]
                
                self.data = []
                self.targets = []
                
                # Load the data
                for file_name in downloaded_list:
                    file_path = os.path.join(root, self.base_folder, file_name)
                    with open(file_path, "rb") as f:
                        entry = pickle.load(f, encoding="latin1")
                        self.data.append(entry["data"])
                        self.targets.extend(entry["labels"])
                
                self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
                self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC format
            
            def __getitem__(self, index: int) -> Tuple[Any, Any]:
                img, target = self.data[index], self.targets[index]
                img = Image.fromarray(img)
                
                if self.transform is not None:
                    img = self.transform(img)
                
                if self.target_transform is not None:
                    target = self.target_transform(target)
                
                return img, target
            
            def __len__(self) -> int:
                return len(self.data)
        
        testset = CIFAR10Compatible(root='./data', train=False, download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=2, pin_memory=True)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Initialize network
    print("Initializing network...")
    net = PrimaryNetwork()
    net = net.to(device)
    
    # Load model if available
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        net.load_state_dict(checkpoint['net'])
        print(f"Model loaded with accuracy: {checkpoint.get('acc', 'N/A')}%")
    else:
        print(f"No model found at {args.model_path}, using randomly initialized weights")
    
    # Print model parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Measure inference time
    print("\nMeasuring inference performance...")
    net.eval()
    
    # Warm-up
    print("Warming up...")
    dummy_input = torch.randn(10, 3, 32, 32, device=device)
    with torch.no_grad():
        _ = net(dummy_input)
    
    # Single image inference time
    print("\nSingle image inference time:")
    latencies = []
    with torch.no_grad():
        for _ in range(100):  # Run 100 times to get average
            single_input = torch.randn(1, 3, 32, 32, device=device)
            start_time = time.time()
            _ = net(single_input)
            if device == torch.device("cuda"):
                torch.cuda.synchronize()  # Wait for GPU to finish
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
    
    # Remove outliers (top and bottom 10%)
    latencies = sorted(latencies)[10:-10]
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"Average latency (ms): {avg_latency:.2f}")
    print(f"P95 latency (ms): {p95_latency:.2f}")
    print(f"P99 latency (ms): {p99_latency:.2f}")
    
    # Batch inference time
    print("\nBatch inference time (batch size: {})".format(args.batch_size))
    batch_latencies = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, _ = data
            images = images.to(device)
            
            start_time = time.time()
            _ = net(images)
            if device == torch.device("cuda"):
                torch.cuda.synchronize()
            latency = (time.time() - start_time) * 1000  # Convert to ms
            batch_latencies.append(latency)
            
            if i >= 10:  # Test with 10 batches
                break
    
    avg_batch_latency = np.mean(batch_latencies)
    imgs_per_sec = args.batch_size * 1000 / avg_batch_latency
    
    print(f"Average batch latency (ms): {avg_batch_latency:.2f}")
    print(f"Images per second: {imgs_per_sec:.2f}")
    
    # Run evaluation on test set if model is loaded
    if os.path.exists(args.model_path):
        print("\nEvaluating model on test set...")
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        print(f'Overall accuracy: {100 * correct / total:.2f}%')
        
        print('\nPer-class accuracy:')
        for i in range(10):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
            else:
                accuracy = 0
            print(f'{classes[i]}: {accuracy:.2f}%')
    
    # Print GPU memory usage after inference
    if device == torch.device("cuda"):
        print(f"\nFinal CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Final CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    test_gpu_performance()
