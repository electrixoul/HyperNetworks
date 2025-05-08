"""
HyperNetworks GPU Training Script

This script implements GPU-accelerated training for the HyperNetworks model on CIFAR-10.
Designed to be run in the 'conda mod' environment with GPU support.

Original Paper: "HyperNetworks" by Ha, Dai and Schmidhuber (2016)
https://arxiv.org/abs/1609.09106

Requirements:
- PyTorch 2.x+
- CUDA support
- conda mod environment

To run:
  conda run -n mod python train.py [options]

Options:
  --resume, -r      Resume from checkpoint
  --batch_size      Batch size (default: 128)
  --epochs          Number of epochs (default: 200)
  --lr              Learning rate (default: 0.002)
  --weight_decay    Weight decay (default: 0.0005)
  --checkpoint_path Path for checkpoint (default: ./hypernetworks_cifar_gpu.pth)
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
import warnings

# Ignore specific warnings from torchvision
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

from primary_net import PrimaryNetwork

# Argument Parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with HyperNetworks on GPU')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--checkpoint_path', type=str, default='./hypernetworks_cifar_gpu.pth', help='path to save checkpoint')
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Data loaders
print("Loading CIFAR-10 dataset...")

def get_cifar10_datasets():
    try:
        # Attempt to use standard CIFAR10 loader
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)
        return trainset, testset
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
        
        trainset = CIFAR10Compatible(root='./data', train=True, download=True, transform=transform_train)
        testset = CIFAR10Compatible(root='./data', train=False, download=True, transform=transform_test)
        return trainset, testset

trainset, testset = get_cifar10_datasets()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=4, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Initialize network
print("Initializing network...")
net = PrimaryNetwork()
net = net.to(device)

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    net = nn.DataParallel(net)

best_accuracy = 0.0

# Resume from checkpoint if specified
if args.resume:
    print("Resuming from checkpoint...")
    checkpoint = torch.load(args.checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_accuracy = checkpoint['acc']
    print(f"Loaded checkpoint with accuracy: {best_accuracy:.2f}%")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Learning rate scheduler
milestones = [int(args.epochs * 0.3), int(args.epochs * 0.6), int(args.epochs * 0.8)]
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer, milestones=milestones, gamma=0.5)

# Training loop
print("Starting training...")
total_parameters = sum(p.numel() for p in net.parameters())
print(f"Total parameters in model: {total_parameters:,}")

for epoch in range(args.epochs):
    start_time = time.time()
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(trainloader)}], "
                  f"Loss: {running_loss/50:.4f}, Acc: {100.*correct/total:.2f}%")
            running_loss = 0.0
    
    # Apply learning rate scheduler
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Evaluate on test set
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * test_correct / test_total
    epoch_time = time.time() - start_time
    
    print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s | "
          f"Train Acc: {100.*correct/total:.2f}% | Test Acc: {accuracy:.2f}% | "
          f"Learning Rate: {current_lr:.6f}")
    
    # Save checkpoint if accuracy improved
    if accuracy > best_accuracy:
        print('Saving model...')
        state = {
            'net': net.state_dict(),
            'acc': accuracy,
            'epoch': epoch,
        }
        torch.save(state, args.checkpoint_path)
        best_accuracy = accuracy

print('Finished Training')
print(f'Best accuracy: {best_accuracy:.2f}%')
