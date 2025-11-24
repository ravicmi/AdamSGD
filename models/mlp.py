"""
Multi-Layer Perceptron (MLP) model for MNIST classification.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with ReLU activations and dropout.
    Used for Section 6.2 experiments.
    
    Args:
        input_size (int): Size of input features (784 for flattened 28x28 MNIST)
        hidden_sizes (list): List of hidden layer sizes
        num_classes (int): Number of output classes (10 for MNIST)
        dropout (float): Dropout probability. Default: 0.5
    """
    
    def __init__(self, input_size=784, hidden_sizes=[1000, 1000], num_classes=10, dropout=0.5):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        out = self.network(x)
        return out

