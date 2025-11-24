"""
Logistic Regression model for MNIST classification.
"""

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    Simple logistic regression model (single linear layer).
    Used for Section 6.1 experiments.
    
    Args:
        input_size (int): Size of input features (784 for flattened 28x28 MNIST)
        num_classes (int): Number of output classes (10 for MNIST)
    """
    
    def __init__(self, input_size=784, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
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
        
        out = self.linear(x)
        return out

