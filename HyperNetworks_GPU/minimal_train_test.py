import torch
import torch.nn as nn
import torch.optim as optim

from primary_net import PrimaryNetwork

def test_minimal_training():
    """Test a minimal training loop with the HyperNetwork model"""
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Running on {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Running on CPU.")
    
    # Initialize network
    print("\nInitializing PrimaryNetwork...")
    net = PrimaryNetwork()
    net = net.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Create a small batch of dummy data
    batch_size = 8
    print(f"\nCreating dummy batch of {batch_size} images...")
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
    dummy_labels = torch.randint(0, 10, (batch_size,), device=device)
    
    # Execute one forward and backward pass
    print("\nExecuting forward pass...")
    optimizer.zero_grad()
    outputs = net(dummy_input)
    print(f"Output shape: {outputs.shape}")
    
    print("\nCalculating loss...")
    loss = criterion(outputs, dummy_labels)
    print(f"Initial loss: {loss.item():.4f}")
    
    print("\nExecuting backward pass...")
    loss.backward()
    
    print("\nUpdating weights...")
    optimizer.step()
    
    # Check if the model parameters were updated
    print("\nVerifying parameters were updated...")
    optimizer.zero_grad()
    new_outputs = net(dummy_input)
    new_loss = criterion(new_outputs, dummy_labels)
    print(f"New loss: {new_loss.item():.4f}")
    
    # Print whether the loss changed after the update
    if new_loss.item() != loss.item():
        print("\nSuccess! Model parameters were updated correctly.")
    else:
        print("\nWarning: Loss hasn't changed. There might be an issue with parameter updates.")
    
    # Memory usage
    if device == torch.device("cuda"):
        print(f"\nCUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print("\nMinimal training test completed.")

if __name__ == "__main__":
    test_minimal_training()
