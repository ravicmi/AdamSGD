"""
Convolutional Neural Network (CNN) model for CIFAR-10 classification.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional neural network for image classification.
    Used for Section 6.3 experiments.
    
    Architecture:
    - Multiple convolutional layers with ReLU and max pooling
    - Fully connected layers with dropout
    - Output layer
    
    Args:
        num_classes (int): Number of output classes (10 for CIFAR-10)
        conv_channels (list): List of channel sizes for conv layers. Default: [32, 64, 128]
        fc_hidden (int): Hidden size for fully connected layer. Default: 512
        dropout (float): Dropout probability. Default: 0.5
    """
    
    def __init__(self, num_classes=10, conv_channels=[32, 64, 128], fc_hidden=512, dropout=0.5):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels[0], conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels[1], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels[2], conv_channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        )
        
        # Calculate size after convolutions
        # CIFAR-10: 32x32 -> 16x16 -> 8x8 -> 4x4
        conv_output_size = conv_channels[2] * 4 * 4
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return x

